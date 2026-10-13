"""
autoencoder.py — Module 2: Transformer Autoencoder for Log Anomaly Detection

Architecture: bottleneck encoder-decoder, not masked-language-model-style
denoising. A learned [POOL] token's encoder output is projected down to a
small bottleneck vector; the decoder reconstructs the FULL original window
from just that vector via cross-attention. Reconstruction "loss" = average
per-position cross-entropy against the true template IDs — this is the
anomaly score (higher = less like anything seen during training on normal
data).

Vocabulary note: index 0 is reserved as the OOV/UNK slot for any template_id
encountered at eval/inference time that never appeared in training (a
genuinely novel event type). This is why vocab_size must be
(number of templates seen in training) + 1, and any id >= that number gets
clamped to 0 before being fed to the model.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class LogAutoencoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        window_size: int = 20,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 128,
        bottleneck_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len=window_size + 1)

        # learned [POOL] token, prepended to every window — its encoder output
        # becomes the compressed representation the whole window is judged by
        self.pool_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.to_bottleneck = nn.Linear(d_model, bottleneck_dim)
        self.from_bottleneck = nn.Linear(bottleneck_dim, d_model)

        # learned per-position query embeddings the decoder uses to "ask"
        # the bottleneck memory what belonged at each position
        self.decoder_pos_queries = nn.Parameter(torch.randn(1, window_size, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.output_head = nn.Linear(d_model, vocab_size)

    def encode(self, template_ids: torch.Tensor) -> torch.Tensor:
        """template_ids: (batch, window_size) -> bottleneck: (batch, bottleneck_dim)"""
        batch_size = template_ids.size(0)
        embedded = self.token_embedding(template_ids)  # (B, W, d_model)
        pool = self.pool_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        seq = torch.cat([pool, embedded], dim=1)  # (B, 1+W, d_model)
        seq = self.pos_encoding(seq)
        encoded = self.encoder(seq)  # (B, 1+W, d_model)
        pooled = encoded[:, 0, :]  # the [POOL] token's output
        return self.to_bottleneck(pooled)  # (B, bottleneck_dim)

    def decode(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """bottleneck: (batch, bottleneck_dim) -> logits: (batch, window_size, vocab_size)"""
        batch_size = bottleneck.size(0)
        memory = self.from_bottleneck(bottleneck).unsqueeze(1)  # (B, 1, d_model) — single memory token
        queries = self.decoder_pos_queries.expand(batch_size, -1, -1)  # (B, W, d_model)
        decoded = self.decoder(tgt=queries, memory=memory)  # (B, W, d_model)
        return self.output_head(decoded)  # (B, W, vocab_size)

    def forward(self, template_ids: torch.Tensor) -> torch.Tensor:
        bottleneck = self.encode(template_ids)
        return self.decode(bottleneck)

    def position_losses(self, template_ids: torch.Tensor) -> torch.Tensor:
        """
        Per-ROW reconstruction loss, shape (batch, window_size): cross-entropy
        of each position against its true template ID. This is the raw
        material for everything else, and it's what tells you WHICH row of a
        flagged window is the surprising one (useful for explanation / SOAR).
        """
        logits = self.forward(template_ids)  # (B, W, V)
        losses = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), template_ids.reshape(-1), reduction="none"
        )
        return losses.reshape(template_ids.size(0), template_ids.size(1))

    def reconstruction_loss(self, template_ids: torch.Tensor) -> torch.Tensor:
        """Scalar mean cross-entropy over all rows. This is the TRAINING objective
        only — it is not the anomaly score (see anomaly_score)."""
        return self.position_losses(template_ids).mean()

    def anomaly_score(self, template_ids: torch.Tensor, mode: str = "max") -> torch.Tensor:
        """
        One anomaly score per window, shape (batch,).

        mode="max"  (default): the worst single row in the window. A lone
            malicious request hiding among 19 normal ones (e.g. a reverse
            shell in otherwise ordinary browsing) is diluted away by
            averaging: on real AIT-LDS data one such row had loss 11.8 yet its
            window averaged 2.5, under the 3.16 threshold. Max-pooling catches
            it.
        mode="mean": average over rows. Kept for comparison in the results.
        """
        pl = self.position_losses(template_ids)
        if mode == "max":
            return pl.max(dim=1).values
        if mode == "mean":
            return pl.mean(dim=1)
        raise ValueError(f"unknown mode: {mode}")
