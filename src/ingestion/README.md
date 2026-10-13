# Module 1 — Ingestion, Anonymization & Sliding-Window Sequencing

Status: **built and verified** against real LogHub samples (HDFS_2k, BGL_2k).

## Files

- `anonymizer.py` — regex-based masking, always run FIRST on raw logs.
  Masks: IPv4 (with/without port), HDFS block IDs, HPC/BGL node identifiers,
  user/uid/username fields, token/session/jwt fields, MAC addresses.
- `preprocessor.py` — Drain3-based template parsing + sliding-window sequencing.
  Turns anonymized log lines into windows of template IDs — the unit the
  autoencoder and classifier will consume in the next build steps.

## How to run

```bash
pip install -r requirements.txt

# Step 1: anonymize a raw log file
python3 -c "from anonymizer import anonymize_file; anonymize_file('raw.log', 'raw.masked.log')"

# Step 2: parse + window the anonymized file
python3 preprocessor.py raw.masked.log
```

## Verified results (this session)

| Dataset   | Raw lines | Unique templates found | Windows (size 20, no overlap) | Leakage check |
|-----------|-----------|-------------------------|--------------------------------|---------------|
| HDFS_2k   | 2000      | 16                       | 100                             | 0 IPs, 0 block IDs found in masked output |
| BGL_2k    | 2000      | 91                       | 100                             | node IDs correctly masked |

Two different LogHub systems parsed with the *same* code and no
system-specific tuning — this is what will let us later assign different
LogHub systems to different simulated federated clients for the
qualitative heterogeneity demo.

## Design notes / things to remember later

- `template_id` (not the raw template string) is what feeds the autoencoder —
  it's stable, compact, and doesn't leak the original tokens.
- `stride < window_size` in `make_windows()` gives overlapping windows —
  this is reserved for Phase 7 (live replay / low-latency detection), not
  used for training.
- Drain3 template strings only reach their final generalized form (`<*>`
  wildcards in the right places) after enough lines have been absorbed into
  a cluster — don't trust a template's text at first-occurrence time, use
  `final_templates` from `parse_lines()` instead.
- Dates are intentionally NOT masked (needed for windowing/temporal
  analysis and not considered sensitive on their own) — only IPs, block/node
  IDs, and explicit user/token fields are masked.

## Next step (not yet built)

Module 2 — Transformer autoencoder, trained on the `template_id` windows
from *normal* traffic only, to produce the reconstruction-loss anomaly score.
