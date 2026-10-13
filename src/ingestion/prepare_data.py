"""
prepare_data.py — builds windows from a real (log, labels) pair.

Usage:
    python3 prepare_data.py --data access.log --labels access-labels.log --name access

Design decisions (each found necessary by testing on real AIT-LDS data):

1. URL decomposition. Apache access lines cram method+path+query into one
   space-containing quoted field, which Drain's word-level tokenizer treats as
   a single token and wildcards away — collapsing 148k lines into 6
   meaningless templates. We split method/path/query/protocol/status/UA into
   separate tokens BEFORE Drain, giving ~91 templates, over half of them
   attack-only. Lines that don't match the Apache format (syslog, auth.log,
   ...) fall back to the raw masked line, so this script works for them too.

2. Two ground-truth columns. AIT labels every line twice: column 1 is
   time-based (any line near an attack step gets the label, malicious or
   not); column 2 is content-based. A line is a CONFIRMED attack only if both
   columns agree on the same non-zero label. Lines where only column 1 fires
   are "ambiguous" and are excluded from evaluation (they are usually benign
   requests that merely happened during an attack).

3. Drain is fit on ALL lines (it is only a structural parser). The
   autoencoder's VOCABULARY is restricted to templates seen on clean normal
   lines (both columns '0'); anything else maps to the UNK slot (index 0).
"""
import argparse
import json
import re
import sys

sys.path.insert(0, "src/ingestion")
from anonymizer import anonymize_line
from preprocessor import build_template_miner, save_miner

WINDOW_SIZE = 20
STRIDE = 20

_ACCESS_RE = re.compile(
    r'^(?P<ip>\S+) \S+ \S+ \[(?P<ts>[^\]]+)\] "(?P<method>\S+) (?P<path>\S*) (?P<proto>[^"]+)" '
    r'(?P<status>\d+|-) (?P<size>\d+|-) "(?P<referer>[^"]*)" "(?P<ua>[^"]*)"'
)


def decompose_access_line(masked_line: str) -> str:
    """Expose method/path/query sub-tokens to Drain instead of one opaque field."""
    m = _ACCESS_RE.search(masked_line)
    if not m:
        return masked_line
    raw_path = m.group("path")
    path, query = raw_path.split("?", 1) if "?" in raw_path else (raw_path, "")
    path_tokens = " ".join(t for t in path.split("/") if t)
    query_tokens = " ".join(re.split(r"[&=+]", query)) if query else ""
    return (
        f'{m.group("method")} PATH {path_tokens} QUERY {query_tokens} '
        f'PROTO {m.group("proto")} STATUS {m.group("status")} UA {m.group("ua")}'
    )


def load_and_parse(data_path: str, labels_path: str):
    miner = build_template_miner()
    template_ids, col1, col2 = [], [], []
    with open(data_path, "r", encoding="utf-8", errors="replace") as fd, open(
        labels_path, "r", encoding="utf-8", errors="replace"
    ) as fl:
        for data_line, label_line in zip(fd, fl):
            decomposed = decompose_access_line(anonymize_line(data_line))
            template_ids.append(miner.add_log_message(decomposed)["cluster_id"])
            parts = label_line.strip().split(",")
            col1.append(parts[0])
            col2.append(parts[1] if len(parts) > 1 else "0")
    final_templates = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
    return template_ids, col1, col2, final_templates, miner


def make_windows(template_ids, col1, col2, window_size=WINDOW_SIZE, stride=STRIDE):
    windows = []
    for i in range(0, len(template_ids) - window_size + 1, stride):
        seg1, seg2 = col1[i : i + window_size], col2[i : i + window_size]
        confirmed_positions = [p for p, (a, b) in enumerate(zip(seg1, seg2)) if a != "0" and a == b]
        confirmed_types = sorted({seg1[p] for p in confirmed_positions})
        any_label = any(a != "0" or b != "0" for a, b in zip(seg1, seg2))
        windows.append(
            {
                "start": i,
                "end": i + window_size,
                "template_ids": template_ids[i : i + window_size],
                "is_pure_normal": not any_label,
                "confirmed_types": confirmed_types,
                "confirmed_positions": confirmed_positions,
                "ambiguous": bool(any_label and not confirmed_types),
            }
        )
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--name", default="access", help="prefix for files written to data/artifacts/")
    args = ap.parse_args()

    print(f"Parsing {args.data} (~1-2 min per 150k lines)...")
    tids, c1, c2, templates, miner = load_and_parse(args.data, args.labels)
    print(f"Parsed {len(tids)} lines into {len(templates)} templates.")

    windows = make_windows(tids, c1, c2)
    n_norm = sum(w["is_pure_normal"] for w in windows)
    n_conf = sum(bool(w["confirmed_types"]) for w in windows)
    n_amb = sum(w["ambiguous"] for w in windows)
    print(f"{len(windows)} windows: {n_norm} clean-normal | {n_conf} content-confirmed attack | "
          f"{n_amb} ambiguous (time-label only, excluded from evaluation)")

    normal_tids = {t for t, a, b in zip(tids, c1, c2) if a == "0" and b == "0"}
    vocab_map = {tid: i + 1 for i, tid in enumerate(sorted(normal_tids))}  # 0 = UNK
    print(f"Autoencoder vocabulary: {len(vocab_map)} templates (+1 UNK slot)")

    out = f"data/artifacts/{args.name}"
    save_miner(miner, f"{out}_drain_miner.pkl")
    json.dump({str(k): v for k, v in vocab_map.items()}, open(f"{out}_vocab_map.json", "w"))
    json.dump(windows, open(f"{out}_windows.json", "w"))
    json.dump({str(k): v for k, v in templates.items()}, open(f"{out}_templates.json", "w"))
    print(f"Saved artifacts -> {out}_*.json / .pkl")


if __name__ == "__main__":
    main()
