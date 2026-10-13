"""
preprocessor.py — Module 1b: Parsing, Template Extraction & Sliding-Window Sequencing

Adds Drain-state persistence vs. the earlier version: the TemplateMiner fitted
during training MUST be reused (never refit) when parsing evaluation or live
data, or template IDs stop meaning the same thing between runs. save_miner() /
load_miner() handle this.
"""
import pickle
from dataclasses import dataclass, field
from typing import List, Optional

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


@dataclass
class ParsedLine:
    raw: str
    template_id: int


@dataclass
class LogWindow:
    window_id: int
    template_ids: List[int]
    start_line: int
    end_line: int


def build_template_miner() -> TemplateMiner:
    config = TemplateMinerConfig()
    config.drain_sim_th = 0.4
    config.drain_depth = 4
    config.drain_max_children = 100
    config.drain_max_clusters = 1024
    config.profiling_enabled = False
    return TemplateMiner(config=config)


def parse_lines(masked_lines: List[str], miner: Optional[TemplateMiner] = None):
    """
    Parse lines with the given miner (fit further if new patterns appear), or a
    fresh one if none is passed. ALWAYS pass the training miner back in when
    parsing validation/eval/live data — never let this default to a fresh miner
    outside of initial training, or template IDs will not correspond across runs.
    """
    if miner is None:
        miner = build_template_miner()
    parsed = []
    for raw in masked_lines:
        result = miner.add_log_message(raw)
        parsed.append(ParsedLine(raw=raw, template_id=result["cluster_id"]))
    final_templates = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
    return parsed, final_templates, miner


def make_windows(parsed: List[ParsedLine], window_size: int = 20, stride: int = 20) -> List[LogWindow]:
    windows = []
    wid = 0
    i = 0
    while i + window_size <= len(parsed):
        chunk = parsed[i : i + window_size]
        windows.append(
            LogWindow(
                window_id=wid,
                template_ids=[c.template_id for c in chunk],
                start_line=i,
                end_line=i + window_size,
            )
        )
        wid += 1
        i += stride
    return windows


def save_miner(miner: TemplateMiner, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(miner, f)


def load_miner(path: str) -> TemplateMiner:
    with open(path, "rb") as f:
        return pickle.load(f)
