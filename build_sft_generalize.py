#!/usr/bin/env python3
"""Build a breadth-first SFT mix aimed at generalization (not gold memorization).

Rules:
- Prefer unique turns over repeats.
- Keep gold light (policy anchor), not 40x dumps.
- Exclude a frozen held-out prompt set from all training text.
- One-shot tools only (no fake tool_result).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "pipeline" / "input" / "distill_only"
GOLD = ROOT / "pipeline" / "input" / "sft_gold"
OUT = ROOT / "pipeline" / "input" / "sft_generalize"
HELDOUT = ROOT / "pipeline" / "input" / "sft_heldout"

HTML_RE = re.compile(r"<!doctype|<html|<script|<body", re.I)
CODE_HINT = re.compile(r"```|^\s*def |\bSELECT\b|\bJOIN\b", re.I | re.M)

# Light anchors — enough to keep format, not enough to memorize the battery.
GOLD_CODE_REPEATS = 4
GOLD_TOOL_REPEATS = 5
IDENTITY_REPEATS = 8

CODE_ALPACA_CAP = 14000
CODE_SO_CAP = 3000
GENERAL_QA_CAP = 3000
GENERAL_SFT_CAP = 2000
TOOL_MAX_CHARS = 900
CODE_MAX_CHARS = 3500

# Frozen eval prompts — must never appear (substring) in training.
HELD_OUT_PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number. Just the code.",
    "Write count_vowels(s) that returns how many vowels are in s.",
    "Write a binary_search(arr, target) function. Just the code, no tools.",
    "Write a SQL query for the top 10 products by revenue from products(id,name) and sales(id,product_id,revenue).",
    "Create a file called memo.txt with content Ship by Friday.",
    "Delete the cache key 'refresh_token'.",
    "Show me what's in this folder using a tool.",
    "Solve for z: 7z - 3 = 18. Show steps.",
    "Write a Python function that merges two sorted lists. Just the code.",
    "Please remove cache key oauth_state.",
]


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


def user_text(turn: str) -> str:
    if "echo:" not in turn:
        return turn
    return turn.split("echo:", 1)[0]


def is_held_out(turn: str) -> bool:
    blob = user_text(turn).lower()
    for prompt in HELD_OUT_PROMPTS:
        # match core phrases so light rewordings still get excluded if present
        needle = prompt.lower()
        if needle in blob:
            return True
        # also catch short distinctive fragments
        for frag in (
            "nth fibonacci",
            "count_vowels",
            "binary_search(arr, target)",
            "memo.txt",
            "refresh_token",
            "show me what's in this folder",
            "7z - 3 = 18",
            "merges two sorted lists",
            "oauth_state",
            "top 10 products by revenue",
        ):
            if frag in blob:
                return True
    return False


def write_turns(path: Path, turns: list[str]) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(turn.rstrip() + "\n" for turn in turns if turn.strip())
    path.write_text(text, encoding="utf-8")
    return len(turns), len(text.encode("utf-8"))


def is_plain_code(turn: str) -> bool:
    if "user:" not in turn or "echo:" not in turn:
        return False
    if HTML_RE.search(turn) or is_held_out(turn):
        return False
    if "think:" in turn or "tool:" in turn or "tool_result" in turn:
        return False
    if len(turn) > CODE_MAX_CHARS:
        return False
    return bool(CODE_HINT.search(turn))


def clean_tool_turn(turn: str) -> str | None:
    if "user:" not in turn or HTML_RE.search(turn) or is_held_out(turn):
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


def structural_tool_variants() -> list[str]:
    """Many unique keys/files — never reuse held-out names."""
    turns = []
    files = [
        ("alpha.txt", "first note"),
        ("beta.md", "draft"),
        ("config.json", '{"ok": true}'),
        ("out.log", "ready"),
        ("plan.txt", "next sprint"),
        ("data.csv", "a,b\n1,2"),
        ("summary.txt", "done"),
        ("err.log", "none"),
    ]
    for name, content in files:
        turns.append(
            f"user: Create a new file called {name} with the content '{content}'.\n"
            f"echo: think: Write {name}.\ntool: file_write filename={name} content={content}\n"
        )
        turns.append(
            f"user: Write '{content}' into {name}.\n"
            f"echo: think: Use file_write.\ntool: file_write filename={name} content={content}\n"
        )
    keys = [
        "temp_data",
        "session_token",
        "auth_cache",
        "draft_key",
        "user_prefs",
        "rate_limit",
        "job_queue",
        "feature_flag",
        "invite_code",
        "csrf_nonce",
    ]
    for key in keys:
        turns.append(
            f"user: Delete the cache key '{key}'.\n"
            f"echo: think: Remove that cache key.\ntool: cache_delete key={key}\n"
        )
        turns.append(
            f"user: Please delete cache key {key}.\n"
            f"echo: think: Delete the key instead of setting it to null.\ntool: cache_delete key={key}\n"
        )
    for phrasing, cmd in (
        ("List files in the current directory using a tool.", 'ls -la'),
        ("Use a tool to list the current directory.", "ls"),
        ("List the working directory with a shell tool.", "ls -1"),
        ("What files are here? Use a tool.", "ls -la"),
    ):
        turns.append(
            f"user: {phrasing}\n"
            f'echo: think: List the working directory.\ntool: shell_run command="{cmd}"\n'
        )
    return turns


def structural_code_variants() -> list[str]:
    """Extra unique code tasks — not the frozen held-out set."""
    samples = [
        (
            "Write a Python function that reverses a string. Just the code.",
            "def reverse_string(s):\n    return s[::-1]",
        ),
        (
            "Write is_even(n) that returns True if n is even.",
            "def is_even(n):\n    return n % 2 == 0",
        ),
        (
            "Write is_odd(n) that returns True when n is odd. Just the code.",
            "```python\ndef is_odd(n):\n    return n % 2 == 1\n```",
        ),
        (
            "Write a Python function that returns the factorial of n. Just the code.",
            "```python\ndef factorial(n):\n    if n < 0:\n        raise ValueError(\"n must be >= 0\")\n"
            "    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n```",
        ),
        (
            "Write max_of_two(a, b). Just the code, no tools.",
            "```python\ndef max_of_two(a, b):\n    return a if a >= b else b\n```",
        ),
        (
            "Write sum_list(nums) that returns the sum of a list. Just the code.",
            "```python\ndef sum_list(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total\n```",
        ),
        (
            "Write clamp(x, lo, hi). Just the code.",
            "```python\ndef clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n```",
        ),
        (
            "Write title_case(s) that capitalizes each word. Just the code.",
            "```python\ndef title_case(s):\n    return \" \".join(w[:1].upper() + w[1:].lower() for w in s.split())\n```",
        ),
    ]
    return [f"user: {u}\necho: {e}\n" for u, e in samples]


def main():
    rng = random.Random(7)
    if OUT.exists():
        for old in OUT.glob("*.txt"):
            old.unlink()
    else:
        OUT.mkdir(parents=True, exist_ok=True)
    HELDOUT.mkdir(parents=True, exist_ok=True)

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
    gold_code = [
        t for t in split_turns((GOLD / "sft_code_gold.txt").read_text(encoding="utf-8"))
        if not is_held_out(t)
    ]
    code_turns = (
        gold_code * GOLD_CODE_REPEATS
        + structural_code_variants() * 3
        + dedupe(code_turns)
    )
    code_turns = [t for t in code_turns if not is_held_out(t)]
    n, nbytes = write_turns(OUT / "sft_code_alpaca.txt", code_turns)
    rows.append(("sft_code_alpaca.txt", n, nbytes))

    tool_turns = []
    for name in ("agent_data1.txt", "agent_data_batch2.txt", "tool_calling_data.txt"):
        cleaned = [clean_tool_turn(turn) for turn in split_turns(load_src(name))]
        cleaned = [turn for turn in cleaned if turn]
        tool_turns.extend(cleaned)
        print(f"tools source {name}: {len(cleaned)}", flush=True)
    gold_tools = [
        t for t in split_turns((GOLD / "tool_sft_gold.txt").read_text(encoding="utf-8"))
        if not is_held_out(t)
    ]
    tool_turns = (
        gold_tools * GOLD_TOOL_REPEATS
        + structural_tool_variants() * 4
        + dedupe(tool_turns)
    )
    tool_turns = [t for t in tool_turns if not is_held_out(t)]
    n, nbytes = write_turns(OUT / "tool_sft_clean.txt", tool_turns)
    rows.append(("tool_sft_clean.txt", n, nbytes))

    identity = []
    for name in ("echo_identity_and_everyday.txt", "foundations_identity.txt"):
        src = SRC / name
        if src.exists():
            identity.extend(split_turns(src.read_text(encoding="utf-8", errors="ignore")))
    identity = [
        turn for turn in dedupe(identity)
        if "echo:" in turn and "think:" not in turn and not is_held_out(turn)
    ]
    n, nbytes = write_turns(OUT / "echo_identity_and_everyday.txt", identity)
    rows.append(("echo_identity_and_everyday.txt", n, nbytes))

    boost = []
    src_boost = SRC / "echo_identity_boost.txt"
    if src_boost.exists():
        boost = split_turns(src_boost.read_text(encoding="utf-8", errors="ignore"))
    boost = [
        turn for turn in dedupe(boost)
        if "echo:" in turn and "think:" not in turn and not is_held_out(turn)
    ]
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
        src = SRC / name
        if not src.exists():
            continue
        picked = [
            turn for turn in split_turns(src.read_text(encoding="utf-8", errors="ignore"))
            if "echo:" in turn and "think:" not in turn and "tool:" not in turn
            and not HTML_RE.search(turn) and len(turn) < 2000 and not is_held_out(turn)
        ]
        picked = dedupe(picked)
        if len(picked) > cap:
            rng.shuffle(picked)
            picked = picked[:cap]
        general.extend(picked)
        print(f"general source {name}: {len(picked)}", flush=True)
    n, nbytes = write_turns(OUT / "newdata_output_qa.txt", dedupe(general))
    rows.append(("newdata_output_qa.txt", n, nbytes))

    total = sum(nbytes for _, _, nbytes in rows)
    print("\nGeneralize mix:")
    for name, count, nbytes in rows:
        print(f"  {name:40} {count:6} turns  {nbytes/1e6:6.2f} MB")
    print(f"TOTAL {total/1e6:.2f} MB -> {OUT}")
    (OUT / "MANIFEST.txt").write_text(
        "Echo 2B generalization SFT\n"
        + f"total_bytes={total}\n"
        + "\n".join(f"{n}\tturns={c}\tbytes={b}" for n, c, b in rows)
        + "\n",
        encoding="utf-8",
    )

    (HELDOUT / "held_out_prompts.json").write_text(
        json.dumps(
            {
                "note": "Never include these user prompts in training. Use for eval only.",
                "prompts": HELD_OUT_PROMPTS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Held-out prompts -> {HELDOUT / 'held_out_prompts.json'}")


if __name__ == "__main__":
    main()
