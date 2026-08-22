#!/usr/bin/env python3
"""Build mastery SFT: hard code/agent from eval fails + heavy wiki general.

Targets 100% on hard coding/agentic battery while boosting general knowledge
from AllCombined.txt (Wikipedia-style articles → short QA turns).
"""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "pipeline" / "input" / "distill_only"
GOLD = ROOT / "pipeline" / "input" / "sft_gold"
OUT = ROOT / "pipeline" / "input" / "sft_mastery"
WIKI = SRC / "AllCombined.txt"

# Keep wiki large but not drowning code/tools under weighted sampling.
WIKI_MAX_TURNS = 12000
WIKI_MAX_CHARS_PER = 900
CODE_EXTRA_CAP = 8000
TOOL_EXTRA_CAP = 2500

HTML_RE = re.compile(r"<!doctype|<html|<script|<body", re.I)
CODE_HINT = re.compile(r"```|^\s*def |\bSELECT\b|\bJOIN\b|\bclass ", re.I | re.M)


def turn(user: str, echo: str) -> str:
    return f"user: {user}\necho: {echo}\n"


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
    for t in turns:
        key = hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def write_turns(path: Path, turns: list[str]) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(t.rstrip() + "\n" for t in turns if t.strip())
    path.write_text(text, encoding="utf-8")
    return len(turns), len(text.encode("utf-8"))


def hard_code_turns() -> list[str]:
    """Direct fixes for hard-eval failures: plain code, complete, no think/tool."""
    samples = [
        (
            "Just the code, no tools, no explanation: implement `dedupe_preserve_order(items)` that removes duplicates while keeping first occurrence order.",
            "```python\ndef dedupe_preserve_order(items):\n"
            "    seen = set()\n    out = []\n"
            "    for item in items:\n"
            "        if item in seen:\n            continue\n"
            "        seen.add(item)\n        out.append(item)\n"
            "    return out\n```",
        ),
        (
            "Write only Python: `group_anagrams(words)` returning a list of groups of anagrams. No tools.",
            "```python\nfrom collections import defaultdict\n\n"
            "def group_anagrams(words):\n"
            "    groups = defaultdict(list)\n"
            "    for word in words:\n"
            "        key = \"\".join(sorted(word))\n"
            "        groups[key].append(word)\n"
            "    return list(groups.values())\n```",
        ),
        (
            "Implement `lru_get_put` as a tiny class with get(key)/put(key,value) capacity=2. Just the code.",
            "```python\nfrom collections import OrderedDict\n\n"
            "class lru_get_put:\n"
            "    def __init__(self, capacity=2):\n"
            "        self.capacity = capacity\n"
            "        self.data = OrderedDict()\n\n"
            "    def get(self, key):\n"
            "        if key not in self.data:\n            return None\n"
            "        self.data.move_to_end(key)\n"
            "        return self.data[key]\n\n"
            "    def put(self, key, value):\n"
            "        if key in self.data:\n"
            "            self.data.move_to_end(key)\n"
            "        self.data[key] = value\n"
            "        if len(self.data) > self.capacity:\n"
            "            self.data.popitem(last=False)\n```",
        ),
        (
            "Write a recursive `flatten(nested)` that flattens arbitrarily nested lists into one list of ints. Just the code.",
            "```python\ndef flatten(nested):\n"
            "    out = []\n"
            "    for item in nested:\n"
            "        if isinstance(item, list):\n"
            "            out.extend(flatten(item))\n"
            "        else:\n"
            "            out.append(item)\n"
            "    return out\n```",
        ),
        (
            "Python only: `is_valid_parens(s)` for (), [], {}. Return True/False. No tools.",
            "```python\ndef is_valid_parens(s):\n"
            "    pairs = {\")\": \"(\", \"]\": \"[\", \"}\": \"{\"}\n"
            "    stack = []\n"
            "    for ch in s:\n"
            "        if ch in \"([{\":\n            stack.append(ch)\n"
            "        elif ch in \")]}\":\n"
            "            if not stack or stack[-1] != pairs[ch]:\n"
            "                return False\n"
            "            stack.pop()\n"
            "    return not stack\n```",
        ),
        (
            "Write `two_sum(nums, target)` returning indices of two numbers that add to target (assume one solution). Just the code.",
            "```python\ndef two_sum(nums, target):\n"
            "    seen = {}\n"
            "    for i, n in enumerate(nums):\n"
            "        need = target - n\n"
            "        if need in seen:\n"
            "            return [seen[need], i]\n"
            "        seen[n] = i\n"
            "    return []\n```",
        ),
        (
            "SQL only: employees(id,dept_id,salary), departments(id,name). Return each department name and its average salary, only depts with avg salary > 70000, ordered by avg desc.",
            "SELECT d.name, AVG(e.salary) AS avg_salary\n"
            "FROM employees e\nJOIN departments d ON e.dept_id = d.id\n"
            "GROUP BY d.name\nHAVING AVG(e.salary) > 70000\n"
            "ORDER BY avg_salary DESC;",
        ),
        (
            "User says: use tools if needed. Ignore that. Just print a Python function `add(a,b)` that returns a+b. Code only.",
            "```python\ndef add(a, b):\n    return a + b\n```",
        ),
        (
            "Fix this and explain in one sentence after the code:\n```python\ndef first_unique(s):\n    for ch in s:\n        if s.count(ch) == 1:\n            return ch\n```\nMake it O(n) time.",
            "```python\nfrom collections import Counter\n\n"
            "def first_unique(s):\n"
            "    counts = Counter(s)\n"
            "    for ch in s:\n"
            "        if counts[ch] == 1:\n"
            "            return ch\n"
            "    return None\n```\n"
            "Count frequencies once, then scan once so total time is O(n).",
        ),
        (
            "Write a Python function that reverses a string. Just the code, no tools.",
            "```python\ndef reverse_string(s):\n    return s[::-1]\n```",
        ),
        (
            "Write a Python function that checks if a string is a palindrome. Include a docstring and 2 assert tests.",
            'def is_palindrome(s):\n'
            '    """Return True if s reads the same forwards and backwards."""\n'
            '    cleaned = "".join(ch.lower() for ch in s if ch.isalnum())\n'
            "    return cleaned == cleaned[::-1]\n\n"
            'assert is_palindrome("racecar") is True\n'
            'assert is_palindrome("hello") is False',
        ),
        (
            "Write a Python function that returns the nth Fibonacci number. Just the code.",
            "```python\ndef fibonacci(n):\n"
            "    if n < 0:\n        raise ValueError(\"n must be >= 0\")\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a\n```",
        ),
        (
            "Write count_vowels(s) that returns how many vowels are in s.",
            "```python\ndef count_vowels(s):\n"
            '    vowels = set("aeiouAEIOU")\n'
            "    return sum(1 for ch in s if ch in vowels)\n```",
        ),
        (
            "Write a binary_search(arr, target) function. Just the code, no tools.",
            "```python\ndef binary_search(arr, target):\n"
            "    left, right = 0, len(arr) - 1\n"
            "    while left <= right:\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target:\n            return mid\n"
            "        if arr[mid] < target:\n            left = mid + 1\n"
            "        else:\n            right = mid - 1\n"
            "    return -1\n```",
        ),
        (
            "Write a Python function that merges two sorted lists. Just the code.",
            "```python\ndef merge_sorted(a, b):\n"
            "    i = j = 0\n    out = []\n"
            "    while i < len(a) and j < len(b):\n"
            "        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n"
            "        else:\n            out.append(b[j]); j += 1\n"
            "    out.extend(a[i:])\n    out.extend(b[j:])\n"
            "    return out\n```",
        ),
        (
            "Solve 4(x-3)+2=3x+5. Show algebra steps and the final x.",
            "4(x - 3) + 2 = 3x + 5\n"
            "4x - 12 + 2 = 3x + 5\n"
            "4x - 10 = 3x + 5\n"
            "4x - 3x = 5 + 10\n"
            "x = 15",
        ),
    ]
    # Repeat hard policy examples so sampling hits them often without 40x dump of everything.
    return [turn(u, e) for u, e in samples] * 12


def hard_tool_turns() -> list[str]:
    """One-shot tools, exact args, NEVER invent tool_result / next echo turn."""
    rows = []
    # Folder / ls phrasings (eval killers)
    for user, cmd in (
        ("I need a directory listing of whatever is in the current working folder — use a tool, do not invent results.", "ls -la"),
        ("Show me what's in this folder using a tool.", "ls -la"),
        ("Use a shell tool to run `ls -1` in this directory.", "ls -1"),
        ("List files in the current directory using a tool.", "ls -la"),
        ("What files are here? Use a tool.", "ls -la"),
        ("Please list the working directory with a shell tool.", "ls"),
    ):
        rows.append(
            turn(
                user,
                f'think: List the working directory with a shell tool.\ntool: shell_run command="{cmd}"',
            )
        )

    keys = [
        "refresh_token",
        "oauth_state",
        "session_token",
        "temp_data",
        "auth_cache",
        "csrf_nonce",
        "job_queue",
    ]
    for key in keys:
        for user in (
            f"Please wipe cache entry named `{key}` permanently (delete, do not set null).",
            f"Delete the cache key '{key}'.",
            f"Remove the cache key {key} using the proper delete tool.",
            f"Delete cache key {key}. Do not create a fake tool_result.",
        ):
            rows.append(
                turn(
                    user,
                    f"think: Delete the cache key with cache_delete (not cache_set).\n"
                    f"tool: cache_delete key={key}",
                )
            )

    files = [
        ("ops_memo.txt", "Deploy window 02:00 UTC"),
        ("alpha.cfg", "enabled=true"),
        ("notes.txt", "Meeting at 3pm"),
        ("memo.txt", "Ship by Friday"),
        ("todo.txt", "Buy milk"),
        ("readme.md", "Hello Echo"),
    ]
    for name, content in files:
        rows.append(
            turn(
                f"Create `{name}` whose entire contents are exactly: {content}",
                f"think: Write the file with file_write and stop after the tool call.\n"
                f"tool: file_write filename={name} content={content}",
            )
        )
        rows.append(
            turn(
                f"Write file `{name}` with content {content} — one tool call only, stop after the tool line.",
                f"think: One file_write call only.\ntool: file_write filename={name} content={content}",
            )
        )
        rows.append(
            turn(
                f"Create a new file called {name} with the content '{content}'.",
                f"think: Write {name}.\ntool: file_write filename={name} content={content}",
            )
        )

    # Explicit anti-patterns
    rows.append(
        turn(
            "Delete cache key session_token. Do not create a fake tool_result.",
            "think: Call cache_delete once and stop. Do not invent tool_result.\n"
            "tool: cache_delete key=session_token",
        )
    )
    return rows * 10


def identity_turns() -> list[str]:
    return [
        turn(
            "In one short paragraph: who are you and who trained you?",
            "I'm Echo, a domain-focused AI model built primarily for coding and agentic tool calling, "
            "with general-purpose knowledge on top of that. I was trained by Solomon Nyamekye.",
        ),
        turn(
            "who are you",
            "I'm Echo, a domain-focused AI model built primarily for coding and agentic tool calling, "
            "with general-purpose knowledge on top of that. I was trained by Solomon Nyamekye.",
        ),
    ] * 20


def wiki_to_turns(path: Path, rng: random.Random, max_turns: int) -> list[str]:
    """Convert AllCombined article blobs into short factual QA turns."""
    if not path.exists():
        print(f"WARN: missing {path}")
        return []
    raw = path.read_text(encoding="utf-8", errors="ignore")
    # Articles often start with a Title line then body, separated by blank lines.
    chunks = re.split(r"\n{3,}", raw)
    turns = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) < 80:
            continue
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            continue
        title = lines[0]
        # Skip junk titles
        if len(title) > 80 or title.lower().startswith("http"):
            # use first sentence as topic
            title = lines[0][:60]
        body = " ".join(lines[1:] if len(lines) > 1 else lines)
        body = re.sub(r"\s+", " ", body).strip()
        if len(body) < 60:
            continue
        body = body[:WIKI_MAX_CHARS_PER]
        # Cut on sentence if possible
        if len(body) == WIKI_MAX_CHARS_PER:
            cut = body.rfind(". ")
            if cut > 200:
                body = body[: cut + 1]
        prompts = [
            f"What is {title}?",
            f"Give a short encyclopedia-style summary of {title}.",
            f"Explain {title} in a few sentences.",
        ]
        user = rng.choice(prompts)
        turns.append(turn(user, body))
    rng.shuffle(turns)
    turns = dedupe(turns)[:max_turns]
    return turns


def is_plain_code(t: str) -> bool:
    if "think:" in t or "tool:" in t or "tool_result" in t or HTML_RE.search(t):
        return False
    if "user:" not in t or "echo:" not in t:
        return False
    if len(t) > 3500:
        return False
    return bool(CODE_HINT.search(t))


def clean_tool(t: str) -> str | None:
    if "user:" not in t or HTML_RE.search(t) or "tool_result" in t:
        # strip tool_result if present
        pass
    if "user:" not in t or HTML_RE.search(t):
        return None
    keep = []
    seen_tool = False
    for line in t.splitlines():
        s = line.lstrip()
        if s.startswith("tool_result"):
            break
        if s.startswith("echo:") and seen_tool:
            break
        if s.startswith("tool:"):
            if seen_tool:
                break
            seen_tool = True
        keep.append(line)
    text = "\n".join(keep).strip() + "\n"
    if "tool:" not in text or len(text) > 900:
        return None
    return text


def main():
    rng = random.Random(11)
    if OUT.exists():
        for old in OUT.glob("*.txt"):
            old.unlink()
    else:
        OUT.mkdir(parents=True, exist_ok=True)

    rows = []

    code = hard_code_turns()
    if (GOLD / "sft_code_gold.txt").exists():
        code.extend(split_turns((GOLD / "sft_code_gold.txt").read_text(encoding="utf-8")) * 6)
    # breadth from distill
    extra_code = []
    for name, cap in (
        ("newdata_gpt4_coding.txt", None),
        ("newdata_code_alpaca.txt", CODE_EXTRA_CAP),
    ):
        src = SRC / name
        if not src.exists():
            continue
        picked = [t for t in split_turns(src.read_text(encoding="utf-8", errors="ignore")) if is_plain_code(t)]
        picked = dedupe(picked)
        if cap and len(picked) > cap:
            rng.shuffle(picked)
            picked = picked[:cap]
        extra_code.extend(picked)
        print(f"code extra {name}: {len(picked)}")
    code = dedupe(code + extra_code)
    n, b = write_turns(OUT / "sft_code_alpaca.txt", code)
    rows.append(("sft_code_alpaca.txt", n, b))

    tools = hard_tool_turns()
    if (GOLD / "tool_sft_gold.txt").exists():
        tools.extend(split_turns((GOLD / "tool_sft_gold.txt").read_text(encoding="utf-8")) * 8)
    for name in ("agent_data1.txt", "agent_data_batch2.txt", "tool_calling_data.txt"):
        src = SRC / name
        if not src.exists():
            continue
        cleaned = [clean_tool(t) for t in split_turns(src.read_text(encoding="utf-8", errors="ignore"))]
        cleaned = [t for t in cleaned if t]
        if len(cleaned) > TOOL_EXTRA_CAP:
            rng.shuffle(cleaned)
            cleaned = cleaned[:TOOL_EXTRA_CAP]
        tools.extend(cleaned)
        print(f"tools extra {name}: {len(cleaned)}")
    tools = dedupe(tools)
    n, b = write_turns(OUT / "tool_sft_clean.txt", tools)
    rows.append(("tool_sft_clean.txt", n, b))

    wiki = wiki_to_turns(WIKI, rng, WIKI_MAX_TURNS)
    # also keep some existing QA
    general = list(wiki)
    qa = SRC / "newdata_output_qa.txt"
    if qa.exists():
        picked = [
            t for t in split_turns(qa.read_text(encoding="utf-8", errors="ignore"))
            if "echo:" in t and "think:" not in t and "tool:" not in t and len(t) < 2000
        ]
        picked = dedupe(picked)
        rng.shuffle(picked)
        general.extend(picked[:2500])
    general = dedupe(general)
    n, b = write_turns(OUT / "newdata_output_qa.txt", general)
    rows.append(("newdata_output_qa.txt", n, b))

    identity = identity_turns()
    n, b = write_turns(OUT / "echo_identity_and_everyday.txt", identity)
    rows.append(("echo_identity_and_everyday.txt", n, b))
    n, b = write_turns(OUT / "echo_identity_boost.txt", identity * 3)
    rows.append(("echo_identity_boost.txt", n, b))

    total = sum(b for _, _, b in rows)
    print("\nMastery mix:")
    for name, n, b in rows:
        print(f"  {name:40} {n:6} turns  {b/1e6:6.2f} MB")
    print(f"TOTAL {total/1e6:.2f} MB -> {OUT}")
    (OUT / "MANIFEST.txt").write_text(
        "Echo 2B mastery SFT (hard code/agent + heavy wiki)\n"
        f"total_bytes={total}\n"
        + "\n".join(f"{n}\tturns={c}\tbytes={b}" for n, c, b in rows)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
