#!/usr/bin/env python3
"""Build variant-heavy gold for a short post-SFT polish pass."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "pipeline" / "input" / "sft_polish"
GOLD = ROOT / "pipeline" / "input" / "sft_gold"
CLEAN = ROOT / "pipeline" / "input" / "sft_clean"


def turn(user: str, echo: str) -> str:
    return f"user: {user}\necho: {echo}\n"


def code_variants() -> list[str]:
    turns = []
    # reverse
    for name, arg, demo in (
        ("reverse_string", "s", "Hello"),
        ("rev", "text", "Echo"),
        ("flip_str", "word", "Python"),
    ):
        turns.append(
            turn(
                f"Write a Python function that reverses a string. Just the code.",
                f"def {name}({arg}):\n    return {arg}[::-1]",
            )
        )
        turns.append(
            turn(
                f"Write a Python function named {name} that reverses a string. Just the code, no tools.",
                f"```python\ndef {name}({arg}):\n    return {arg}[::-1]\n\nprint({name}(\"{demo}\"))\n```",
            )
        )

    # palindrome
    turns.append(
        turn(
            "Write a Python function that checks if a string is a palindrome. Include a docstring and 2 assert tests.",
            'def is_palindrome(s):\n    """Return True if s reads the same forwards and backwards."""\n'
            '    cleaned = "".join(ch.lower() for ch in s if ch.isalnum())\n'
            "    return cleaned == cleaned[::-1]\n\n"
            'assert is_palindrome("racecar") is True\n'
            'assert is_palindrome("hello") is False',
        )
    )
    turns.append(
        turn(
            "Write is_palindrome(text) with two asserts.",
            "def is_palindrome(text):\n"
            '    t = "".join(c.lower() for c in text if c.isalnum())\n'
            "    return t == t[::-1]\n\n"
            'assert is_palindrome("level")\n'
            'assert not is_palindrome("world")',
        )
    )

    # empty average
    turns.append(
        turn(
            "Fix this bug:\n```python\ndef average(nums):\n    return sum(nums) / len(nums)\nprint(average([]))\n```\nExplain the fix briefly.",
            "`average([])` raises ZeroDivisionError because `len(nums)` is 0. Guard empty input:\n\n"
            "```python\ndef average(nums):\n    if not nums:\n        return 0.0\n    return sum(nums) / len(nums)\n\nprint(average([]))\n```",
        )
    )
    turns.append(
        turn(
            "Fix mean() so an empty list does not crash:\n```python\ndef mean(values):\n    return sum(values) / len(values)\n```",
            "```python\ndef mean(values):\n    if not values:\n        return 0.0\n    return sum(values) / len(values)\n```",
        )
    )

    # factorial / even (held-out style)
    turns.append(
        turn(
            "Write a Python function that returns the factorial of n. Just the code.",
            "```python\ndef factorial(n):\n    if n < 0:\n        raise ValueError(\"n must be >= 0\")\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n```",
        )
    )
    turns.append(
        turn(
            "Write a function is_even(n) that returns True if n is even.",
            "```python\ndef is_even(n):\n    return n % 2 == 0\n```",
        )
    )
    turns.append(
        turn(
            "Write is_odd(n) that returns True when n is odd. Just the code.",
            "```python\ndef is_odd(n):\n    return n % 2 == 1\n```",
        )
    )
    turns.append(
        turn(
            "Write a Python function max_of_two(a, b). Just the code, no tools.",
            "```python\ndef max_of_two(a, b):\n    return a if a >= b else b\n```",
        )
    )

    # SQL variants
    turns.append(
        turn(
            "Write a SQL query to get the top 5 customers by total order amount from tables customers(id,name) and orders(id,customer_id,amount).",
            "SELECT c.id, c.name, SUM(o.amount) AS total_amount\n"
            "FROM customers c\nJOIN orders o ON c.id = o.customer_id\n"
            "GROUP BY c.id, c.name\nORDER BY total_amount DESC\nLIMIT 5;",
        )
    )
    turns.append(
        turn(
            "SQL: top 3 users by spend. Tables users(id,name), purchases(id,user_id,total).",
            "SELECT u.id, u.name, SUM(p.total) AS spend\n"
            "FROM users u\nINNER JOIN purchases p ON u.id = p.user_id\n"
            "GROUP BY u.id, u.name\nORDER BY spend DESC\nLIMIT 3;",
        )
    )
    return turns


def tool_variants() -> list[str]:
    turns = []
    files = [
        ("notes.txt", "Meeting at 3pm"),
        ("todo.txt", "Buy milk"),
        ("readme.md", "Hello Echo"),
        ("log.txt", "started"),
        ("ideas.txt", "ship the polish SFT"),
    ]
    for name, content in files:
        turns.append(
            turn(
                f"Create a new file called {name} with the content '{content}'.",
                f"think: Write {name} with the requested text.\ntool: file_write filename={name} content={content}",
            )
        )
        turns.append(
            turn(
                f"Create a file called {name} with content {content}.",
                f"think: Create the file with file_write.\ntool: file_write filename={name} content={content}",
            )
        )

    keys = ["temp_data", "session_token", "auth_cache", "draft_key", "user_prefs"]
    for key in keys:
        turns.append(
            turn(
                f"Delete the cache key '{key}'.",
                f"think: Remove that cache key.\ntool: cache_delete key={key}",
            )
        )
        turns.append(
            turn(
                f"Please delete cache key {key}.",
                f"think: Delete the key instead of setting it to null.\ntool: cache_delete key={key}",
            )
        )

    for cmd in ('ls -la', 'ls', 'ls -1'):
        turns.append(
            turn(
                "List files in the current directory using a tool.",
                f'think: List the working directory.\ntool: shell_run command="{cmd}"',
            )
        )
        turns.append(
            turn(
                "Use a tool to list the current directory.",
                f'think: Run {cmd}.\ntool: shell_run command="{cmd}"',
            )
        )
    return turns


def math_variants() -> list[str]:
    turns = []
    problems = [
        (
            "Solve for x: 3x + 7 = 22. Show steps.",
            "Subtract 7 from both sides: 3x = 15.\nDivide both sides by 3: x = 5.",
        ),
        (
            "Solve for y: 2y - 4 = 10. Show steps.",
            "Add 4 to both sides: 2y = 14.\nDivide both sides by 2: y = 7.",
        ),
        (
            "Solve for x: 5x + 5 = 20. Show steps.",
            "Subtract 5 from both sides: 5x = 15.\nDivide by 5: x = 3.",
        ),
        (
            "Solve for n: 4n - 8 = 0. Show steps.",
            "Add 8 to both sides: 4n = 8.\nDivide by 4: n = 2.",
        ),
        (
            "Solve 3x + 7 = 22.",
            "3x + 7 = 22\n3x = 15\nx = 5",
        ),
    ]
    for user, echo in problems:
        turns.append(turn(user, echo))
    return turns


def identity_variants() -> list[str]:
    return [
        turn(
            "who are you",
            "I'm Echo, a domain-focused AI model built primarily for coding and agentic tool calling, "
            "with general-purpose knowledge on top of that. I was trained by Solomon Nyamekye.",
        ),
        turn(
            "what is your name",
            "My name is Echo.",
        ),
        turn(
            "who created you",
            "I was created by Solomon Nyamekye.",
        ),
    ]


def write_file(path: Path, turns: list[str], repeats: int) -> tuple[int, int]:
    text = "\n".join(turns * repeats)
    path.write_text(text, encoding="utf-8")
    return len(turns) * repeats, len(text.encode("utf-8"))


def main():
    if OUT.exists():
        for old in OUT.glob("*.txt"):
            old.unlink()
    else:
        OUT.mkdir(parents=True, exist_ok=True)

    # Heavy gold variants (policy), light keep of prior clean mix for breadth.
    code = code_variants()
    # also include original gold once
    if (GOLD / "sft_code_gold.txt").exists():
        code.append((GOLD / "sft_code_gold.txt").read_text(encoding="utf-8"))
    tools = tool_variants()
    if (GOLD / "tool_sft_gold.txt").exists():
        tools.append((GOLD / "tool_sft_gold.txt").read_text(encoding="utf-8"))
    math = math_variants()
    identity = identity_variants()

    rows = []
    n, b = write_file(OUT / "sft_code_alpaca.txt", code, repeats=30)
    rows.append(("sft_code_alpaca.txt", n, b))
    # filename must categorize as tools
    n, b = write_file(OUT / "tool_sft_clean.txt", tools, repeats=25)
    rows.append(("tool_sft_clean.txt", n, b))
    n, b = write_file(OUT / "newdata_output_qa.txt", math, repeats=20)
    rows.append(("newdata_output_qa.txt", n, b))
    n, b = write_file(OUT / "echo_identity_and_everyday.txt", identity, repeats=15)
    rows.append(("echo_identity_and_everyday.txt", n, b))
    n, b = write_file(OUT / "echo_identity_boost.txt", identity, repeats=40)
    rows.append(("echo_identity_boost.txt", n, b))

    # sprinkle a slice of previous clean code/tools for diversity (not dumps)
    for src_name, dst_name, max_chars in (
        ("sft_code_alpaca.txt", "sft_code_extra.txt", 2_000_000),
        ("tool_sft_clean.txt", "tool_extra.txt", 400_000),
    ):
        src = CLEAN / src_name
        if not src.exists():
            continue
        raw = src.read_text(encoding="utf-8", errors="ignore")
        # take head only so polish stays gold-dominant
        chunk = raw[:max_chars]
        (OUT / dst_name).write_text(chunk, encoding="utf-8")
        rows.append((dst_name, chunk.count("user:"), len(chunk.encode("utf-8"))))

    total = sum(b for _, _, b in rows)
    print("Polish mix:")
    for name, n, b in rows:
        print(f"  {name:40} ~{n:6} turns  {b/1e6:5.2f} MB")
    print(f"TOTAL {total/1e6:.2f} MB -> {OUT}")
    (OUT / "MANIFEST.txt").write_text(
        "Echo 2B polish SFT\n"
        + "\n".join(f"{n}\tturns~{c}\tbytes={b}" for n, c, b in rows)
        + f"\ntotal_bytes={total}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
