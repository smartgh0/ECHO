#!/usr/bin/env python3
"""Build a small clean SFT mix for the 2B checkpoint.

Plain code (no think/tool), one-shot tools (no fake tool_result),
capped identity, light general, plus gold turns for the eval battery.
"""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "pipeline" / "input" / "distill_only"
GOLD = ROOT / "pipeline" / "input" / "sft_gold"
OUT = ROOT / "pipeline" / "input" / "sft_clean"

HTML_RE = re.compile(r"<!doctype|<html|<script|<body", re.I)
CODE_HINT = re.compile(r"```|^\s*def |\bSELECT\b|\bJOIN\b", re.I | re.M)

IDENTITY_REPEATS = 20
CODE_ALPACA_CAP = 12000
CODE_SO_CAP = 2500
GENERAL_QA_CAP = 2500
GENERAL_SFT_CAP = 1500
TOOL_MAX_CHARS = 900
CODE_MAX_CHARS = 3500


def split_turns(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    turns: list[list[str]] = []
    cur: list[str] = []
    for line in lines:
        if line.lstrip().startswith("user:") and cur:
            turns.append(cur)
            cur = [line]
        else:
            cur.append(line)
    if cur:
        turns.append(cur)
    return ["".join(block).strip() + "\n" for block in turns if "".join(block).strip()]


def dedupe(turns: list[str]) -> list[str]:
    seen = set()
    out = []
    for turn in turns:
        key = hashlib.md5(turn.encode("utf-8", errors="ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(turn)
    return out


def write_turns(path: Path, turns: list[str]) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(turn.rstrip() + "\n" for turn in turns if turn.strip())
    path.write_text(text, encoding="utf-8")
    return len(turns), len(text.encode("utf-8"))


def is_plain_code(turn: str) -> bool:
    if "user:" not in turn or "echo:" not in turn:
        return False
    if HTML_RE.search(turn):
        return False
    if "think:" in turn or "tool:" in turn or "tool_result" in turn:
        return False
    if len(turn) > CODE_MAX_CHARS:
        return False
    return bool(CODE_HINT.search(turn))


def clean_tool_turn(turn: str) -> str | None:
    if "user:" not in turn or HTML_RE.search(turn):
        return None
    keep = []
    seen_tool = False
    for line in turn.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("tool_result"):
            break
        if stripped.startswith("echo:") and seen_tool:
            break
        if stripped.startswith("tool:"):
            if seen_tool:
                break
            seen_tool = True
        keep.append(line)
    text = "\n".join(keep).strip() + "\n"
    if "tool:" not in text:
        return None
    if len(text) > TOOL_MAX_CHARS:
        return None
    return text


def load_src(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8", errors="ignore")


def main():
    rng = random.Random(42)
    if OUT.exists():
        for old in OUT.glob("*.txt"):
            old.unlink()
    else:
        OUT.mkdir(parents=True, exist_ok=True)

    rows = []

    code_turns = []
    for name, cap in (
        ("newdata_gpt4_coding.txt", None),
        ("newdata_code_alpaca.txt", CODE_ALPACA_CAP),
        ("newdata_stackoverflow.txt", CODE_SO_CAP),
    ):
        picked = [turn for turn in split_turns(load_src(name)) if is_plain_code(turn)]
        picked = dedupe(picked)
        if cap is not None and len(picked) > cap:
            rng.shuffle(picked)
            picked = picked[:cap]
        code_turns.extend(picked)
        print(f"code source {name}: {len(picked)}", flush=True)
    gold_code = split_turns((GOLD / "sft_code_gold.txt").read_text(encoding="utf-8"))
    code_turns = gold_code * 40 + dedupe(code_turns)
    n, nbytes = write_turns(OUT / "sft_code_alpaca.txt", code_turns)
    rows.append(("sft_code_alpaca.txt", n, nbytes))

    tool_turns = []
    for name in ("agent_data1.txt", "agent_data_batch2.txt", "tool_calling_data.txt"):
        cleaned = [clean_tool_turn(turn) for turn in split_turns(load_src(name))]
        cleaned = [turn for turn in cleaned if turn]
        tool_turns.extend(cleaned)
        print(f"tools source {name}: {len(cleaned)}", flush=True)
    gold_tools = split_turns((GOLD / "tool_sft_gold.txt").read_text(encoding="utf-8"))
    tool_turns = gold_tools * 50 + dedupe(tool_turns)
    n, nbytes = write_turns(OUT / "tool_sft_clean.txt", tool_turns)
    rows.append(("tool_sft_clean.txt", n, nbytes))

    identity = []
    for name in ("echo_identity_and_everyday.txt", "foundations_identity.txt"):
        identity.extend(split_turns(load_src(name)))
    identity = [turn for turn in dedupe(identity) if "echo:" in turn and "think:" not in turn]
    n, nbytes = write_turns(OUT / "echo_identity_and_everyday.txt", identity)
    rows.append(("echo_identity_and_everyday.txt", n, nbytes))

    boost = split_turns(load_src("echo_identity_boost.txt"))
    boost = [turn for turn in dedupe(boost) if "echo:" in turn and "think:" not in turn]
    who = (
        "user: who are you\n"
        "echo: I'm Echo, a domain-focused AI model built primarily for coding "
        "and agentic tool calling, with general-purpose knowledge on top of that. "
        "I was trained by Solomon Nyamekye.\n"
    )
    boost = [who] + boost
    n, nbytes = write_turns(OUT / "echo_identity_boost.txt", boost * IDENTITY_REPEATS)
    rows.append(("echo_identity_boost.txt", n, nbytes))

    general = []
    for name, cap in (
        ("newdata_output_qa.txt", GENERAL_QA_CAP),
        ("newdata_sft_conversations.txt", GENERAL_SFT_CAP),
    ):
        picked = [
            turn for turn in split_turns(load_src(name))
            if "echo:" in turn and "think:" not in turn and "tool:" not in turn
            and not HTML_RE.search(turn) and len(turn) < 2000
        ]
        picked = dedupe(picked)
        if len(picked) > cap:
            rng.shuffle(picked)
            picked = picked[:cap]
        general.extend(picked)
    for name in (
        "knowledge_science.txt",
        "knowledge_math.txt",
        "knowledge_general.txt",
        "foundations_practical.txt",
    ):
        src = SRC / name
        if src.exists():
            general.extend(split_turns(src.read_text(encoding="utf-8", errors="ignore")))
    n, nbytes = write_turns(OUT / "newdata_output_qa.txt", dedupe(general))
    rows.append(("newdata_output_qa.txt", n, nbytes))

    total = sum(nbytes for _, _, nbytes in rows)
    print("\nSFT mix:")
    for name, count, nbytes in rows:
        print(f"  {name:40} {count:6} turns  {nbytes/1e6:6.2f} MB")
    print(f"TOTAL {total/1e6:.2f} MB  (~{total/4e6:.2f}M tokens rough) -> {OUT}")
    manifest = ["Echo 2B SFT clean corpus", f"total_bytes={total}", ""]
    manifest.extend(f"{name}\tturns={count}\tbytes={nbytes}" for name, count, nbytes in rows)
    (OUT / "MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
