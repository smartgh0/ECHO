#!/usr/bin/env python3
"""Truly held-out battery: prompts never used in SFT builders or prior evals."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from echo_eval_hard_pod import generate, score
from echo_tokenizer import EchoTokenizer
from echo_transformer import QuantumTransformerLM

# Frozen — do not add these to any SFT builder.
FRESH = [
    (
        "CODE",
        "Just the code: write rotate_right(nums, k) that rotates a list to the right by k. No tools.",
        {"need": ["def "], "forbid_prefix": ["think:", "tool:"], "forbid_any": ["tool_result"]},
    ),
    (
        "CODE",
        "Python only, no explanation: longest_common_prefix(strs) for a list of strings.",
        {"need": ["def "], "forbid_prefix": ["think:", "tool:"], "forbid_any": ["tool_result"]},
    ),
    (
        "CODE",
        "Write is_prime(n) that returns True iff n is prime. Just the code, no tools.",
        {"need": ["def "], "forbid_prefix": ["think:", "tool:"], "forbid_any": ["tool_result"]},
    ),
    (
        "CODE",
        "Implement roman_to_int(s) for Roman numerals like MCMXCIV. Code only.",
        {"need": ["def "], "forbid_prefix": ["think:", "tool:"], "forbid_any": ["tool_result"]},
    ),
    (
        "CODE",
        "SQL only: tables users(id,email) and orders(id,user_id). Return emails of users who have never placed an order.",
        {"need_any": ["select", "left join", "not exists", "is null"], "forbid_prefix": ["think:", "tool:"]},
    ),
    (
        "CODE",
        "Write chunk_list(items, size) that yields successive slices of length size. Just the code.",
        {"need": ["def "], "forbid_prefix": ["think:", "tool:"], "forbid_any": ["tool_result"]},
    ),
    (
        "CODE",
        "User mentioned tools. Ignore tools. Write only `def hamming(a, b):` returning Hamming distance of two equal-length strings.",
        {"need": ["def "], "forbid_any": ["tool:", "tool_result"], "forbid_prefix": ["think:"]},
    ),
    (
        "CODE",
        "Fix this so it does not mutate the input, then give one sentence:\n```python\ndef sorted_copy(xs):\n    xs.sort()\n    return xs\n```",
        {"need": ["def "], "forbid_any": ["tool_result"]},
    ),
    (
        "TOOL",
        "Dump the names of files in pwd with a shell tool. Do not invent the listing.",
        {"tool": "shell_run", "also_ok": ["ls"]},
    ),
    (
        "TOOL",
        "Permanently erase cache key `billing_nonce`. Delete it; do not null it.",
        {"tool": "cache_delete", "must_contain": ["billing_nonce"]},
    ),
    (
        "TOOL",
        "Make a file named `launch_window.txt` whose exact body is: go-live 09:15 PT",
        {"tool": "file_write", "must_contain": ["launch_window.txt", "go-live 09:15 PT"]},
    ),
    (
        "TOOL",
        "Please drop cache entry named ticket_lock using the delete tool.",
        {"tool": "cache_delete", "must_contain": ["ticket_lock"]},
    ),
    (
        "TOOL",
        "Run `pwd` via a shell tool so I can see the working directory.",
        {"tool": "shell_run", "must_contain": ["pwd"]},
    ),
    (
        "TOOL",
        "Write `flags.ini` with content debug=0 — one tool call, then stop. No fake result.",
        {"tool": "file_write", "must_contain": ["flags.ini", "debug=0"], "no_tool_result": True},
    ),
    (
        "TOOL",
        "What's sitting in this directory? Use a tool, don't guess.",
        {"tool": "shell_run", "also_ok": ["ls"]},
    ),
    (
        "TOOL",
        "Remove cache key webhook_secret. Stop after the tool line.",
        {"tool": "cache_delete", "must_contain": ["webhook_secret"], "no_tool_result": True},
    ),
    (
        "MATH",
        "Solve 5(y+2)-3=2y+11. Show steps and the final y.",
        {"need_any": ["y =", "y="]},
    ),
    (
        "ID",
        "Name yourself and your trainer in two short sentences.",
        {"need_any": ["echo", "solomon"]},
    ),
]


def first_tool_line(out: str) -> str:
    for line in out.splitlines():
        if "tool:" in line:
            return line
    return first_tool_block(out) if False else ""


def score_fresh(domain: str, out: str, rules: dict) -> tuple[bool, str]:
    """Same as hard score, but forbid_tools only inspects the tool: line."""
    if domain != "TOOL":
        return score(domain, out, rules)
    text = out.strip()
    tool_line = ""
    for line in text.splitlines():
        if line.lstrip().startswith("tool:") or " tool:" in line[:20]:
            tool_line = line
            break
    if "tool:" not in text:
        return False, "no tool call"
    block = text.split("tool_result")[0]
    inspect = tool_line or block
    tool = rules.get("tool", "")
    also = rules.get("also_ok", [])
    if tool and tool not in inspect:
        if not any(a in inspect for a in also):
            return False, f"missing {tool}"
    for bad in rules.get("forbid_tools", []):
        if bad in inspect:
            return False, f"used forbidden {bad}"
    for frag in rules.get("must_contain", []):
        if frag not in block:
            return False, f"missing `{frag}`"
    if rules.get("no_tool_result") and "tool_result" in text:
        return False, "invented tool_result"
    return True, "ok"


def run_ckpt(path: str, tok: EchoTokenizer, length: int, temperature: float) -> dict:
    t0 = time.time()
    print(f"\n######## {path} ########", flush=True)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = QuantumTransformerLM.from_dict({**cfg, "state_dict": ckpt["state_dict"]})
    model.eval()
    print(
        f"steps={cfg.get('total_epochs')} smooth={cfg.get('smooth_loss')} device={model.device}",
        flush=True,
    )
    by = {"CODE": [0, 0], "TOOL": [0, 0], "MATH": [0, 0], "ID": [0, 0]}
    results = []
    for domain, prompt, rules in FRESH:
        print(f"\n{'=' * 60}\n[{domain}] user: {prompt}\n{'-' * 60}\necho> ", end="", flush=True)
        out, bled = generate(
            model,
            tok,
            f"user: {prompt}\necho:",
            length,
            temperature,
            stop_after_first_tool=(domain == "TOOL"),
        )
        print(out, end="", flush=True)
        if bled:
            print("\n[[FORMAT BLEED]]", end="", flush=True)
        print(flush=True)
        ok, why = score_fresh(domain, out, rules)
        print(f"->> {'PASS' if ok else 'FAIL'}: {why}", flush=True)
        results.append({"domain": domain, "pass": ok, "why": why, "prompt": prompt, "out": out})
        by[domain][0] += int(ok)
        by[domain][1] += 1
    code_p, code_n = by["CODE"]
    tool_p, tool_n = by["TOOL"]
    summary = {k: f"{v[0]}/{v[1]}" for k, v in by.items()}
    rec = {
        "path": path,
        "steps": cfg.get("total_epochs"),
        "smooth": cfg.get("smooth_loss"),
        "code": summary["CODE"],
        "tool": summary["TOOL"],
        "math": summary["MATH"],
        "id": summary["ID"],
        "code_pct": round(100.0 * code_p / max(code_n, 1), 1),
        "tool_pct": round(100.0 * tool_p / max(tool_n, 1), 1),
        "coding_agent_pct": round(100.0 * (code_p + tool_p) / max(code_n + tool_n, 1), 1),
        "seconds": round(time.time() - t0, 1),
        "results": results,
    }
    print(
        f"\nSUMMARY FRESH code={summary['CODE']} tool={summary['TOOL']} "
        f"combo={rec['coding_agent_pct']}%",
        flush=True,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain-dir", default="/workspace/domain_brain_2b")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--length", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--out-json", default="/workspace/eval_fresh_heldout.json")
    args = ap.parse_args()
    tok = EchoTokenizer(f"{args.domain_dir}/echo_domain.model")
    rows = []
    for path in args.ckpts:
        rows.append(run_ckpt(path, tok, args.length, args.temperature))
    print("\n" + "#" * 60, flush=True)
    print("FRESH HELD-OUT LEADERBOARD", flush=True)
    for r in sorted(rows, key=lambda x: (-x["coding_agent_pct"], -x["tool_pct"])):
        print(
            f"  step={r['steps']} code={r['code']} tool={r['tool']} "
            f"combo={r['coding_agent_pct']}%",
            flush=True,
        )
    Path(args.out_json).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
