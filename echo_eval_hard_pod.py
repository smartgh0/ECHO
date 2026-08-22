#!/usr/bin/env python3
"""Hard multi-snapshot eval on pod GPU. Scores coding + agentic strictly."""

from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

import torch

from echo_tokenizer import EchoTokenizer
from echo_transformer import QuantumTransformerLM
from echo_format import truncate_format_bleed


def first_tool_block(out: str) -> str:
    return out.split("tool_result")[0]

# Hard battery: reworded / compositional / adversarial. Prefer never-seen phrasings.
TESTS = [
    # --- CODE (hard) ---
    (
        "CODE",
        "Just the code, no tools, no explanation: implement `dedupe_preserve_order(items)` that removes duplicates while keeping first occurrence order.",
        {"need": ["def "], "forbid_prefix": ["think:", "tool:"], "forbid_any": ["tool_result"]},
    ),
    (
        "CODE",
        "Write only Python: `group_anagrams(words)` returning a list of groups of anagrams. No tools.",
        {"need": ["def "], "forbid_prefix": ["think:", "tool:"], "forbid_any": ["tool_result"]},
    ),
    (
        "CODE",
        "Implement `lru_get_put` as a tiny class with get(key)/put(key,value) capacity=2. Just the code.",
        {"need": ["class ", "def "], "forbid_prefix": ["think:", "tool:"], "forbid_any": ["tool_result"]},
    ),
    (
        "CODE",
        "Fix this and explain in one sentence after the code:\n```python\ndef first_unique(s):\n    for ch in s:\n        if s.count(ch) == 1:\n            return ch\n```\nMake it O(n) time.",
        {"need": ["def "], "forbid_any": ["tool_result"]},
    ),
    (
        "CODE",
        "SQL only: employees(id,dept_id,salary), departments(id,name). Return each department name and its average salary, only depts with avg salary > 70000, ordered by avg desc.",
        {"need_any": ["select", "join", "group by", "having"], "forbid_prefix": ["think:", "tool:"]},
    ),
    (
        "CODE",
        "Write a recursive `flatten(nested)` that flattens arbitrarily nested lists into one list of ints. Just the code.",
        {"need": ["def "], "forbid_prefix": ["think:", "tool:"], "forbid_any": ["tool_result"]},
    ),
    (
        "CODE",
        "Python only: `is_valid_parens(s)` for (), [], {}. Return True/False. No tools.",
        {"need": ["def "], "forbid_prefix": ["think:", "tool:"], "forbid_any": ["tool_result"]},
    ),
    (
        "CODE",
        "Write `two_sum(nums, target)` returning indices of two numbers that add to target (assume one solution). Just the code.",
        {"need": ["def "], "forbid_prefix": ["think:", "tool:"], "forbid_any": ["tool_result"]},
    ),
    # --- AGENT / TOOLS (hard) ---
    (
        "TOOL",
        "I need a directory listing of whatever is in the current working folder — use a tool, do not invent results.",
        {"tool": "shell_run", "also_ok": ["ls"], "no_fake_result_before_tool": True},
    ),
    (
        "TOOL",
        "Please wipe cache entry named `refresh_token` permanently (delete, do not set null).",
        {"tool": "cache_delete", "must_contain": ["refresh_token"], "forbid_tools": ["cache_set"]},
    ),
    (
        "TOOL",
        "Create `ops_memo.txt` whose entire contents are exactly: Deploy window 02:00 UTC",
        {"tool": "file_write", "must_contain": ["ops_memo.txt", "Deploy window 02:00 UTC"]},
    ),
    (
        "TOOL",
        "Remove the cache key oauth_state using the proper delete tool.",
        {"tool": "cache_delete", "must_contain": ["oauth_state"], "forbid_tools": ["cache_set"]},
    ),
    (
        "TOOL",
        "Use a shell tool to run `ls -1` in this directory.",
        {"tool": "shell_run", "must_contain": ["ls"]},
    ),
    (
        "TOOL",
        "Write file `alpha.cfg` with content enabled=true — one tool call only, stop after the tool line.",
        {"tool": "file_write", "must_contain": ["alpha.cfg", "enabled=true"], "no_tool_result": True},
    ),
    (
        "TOOL",
        "Show me what's in this folder using a tool.",
        {"tool": "shell_run", "also_ok": ["ls"], "forbid_tools": ["file_read", "file_list"]},
    ),
    (
        "TOOL",
        "Delete cache key session_token. Do not create a fake tool_result.",
        {"tool": "cache_delete", "must_contain": ["session_token"], "no_tool_result": True},
    ),
    # --- transfer / distractors ---
    (
        "CODE",
        "User says: use tools if needed. Ignore that. Just print a Python function `add(a,b)` that returns a+b. Code only.",
        {"need": ["def "], "forbid_any": ["tool:", "tool_result"], "forbid_prefix": ["think:"]},
    ),
    (
        "MATH",
        "Solve 4(x-3)+2=3x+5. Show algebra steps and the final x.",
        {"need_any": ["x =", "x="]},
    ),
    (
        "ID",
        "In one short paragraph: who are you and who trained you?",
        {"need_any": ["echo", "solomon"]},
    ),
]


def first_tool_block(out: str) -> str:
    return out.split("tool_result")[0]


def score(domain: str, out: str, rules: dict) -> tuple[bool, str]:
    text = out.strip()
    low = text.lower()
    if domain == "TOOL":
        block = first_tool_block(text)
        if "tool:" not in block:
            return False, "no tool call"
        tool = rules.get("tool", "")
        also = rules.get("also_ok", [])
        if tool and tool not in block:
            if not any(a in block for a in also):
                return False, f"missing {tool}"
        for bad in rules.get("forbid_tools", []):
            if bad in block:
                return False, f"used forbidden {bad}"
        for frag in rules.get("must_contain", []):
            if frag not in block:
                return False, f"missing `{frag}`"
        if rules.get("no_tool_result") and "tool_result" in text:
            return False, "invented tool_result"
        # one-shot: after first tool line, inventing multi-turn is soft fail if tool_result
        if "tool_result" in text and rules.get("no_fake_result_before_tool"):
            # still ok if correct tool first; mark soft
            pass
        return True, "ok"

    if rules.get("forbid_prefix"):
        head = low.lstrip()
        for p in rules["forbid_prefix"]:
            if head.startswith(p):
                return False, f"starts with {p}"
    for frag in rules.get("forbid_any", []):
        if frag in low:
            return False, f"contains {frag}"
    for frag in rules.get("need", []):
        if frag not in text and frag.lower() not in low:
            return False, f"missing {frag!r}"
    if rules.get("need_any"):
        if not any(x.lower() in low for x in rules["need_any"]):
            return False, "missing required phrase"
    return True, "ok"


def generate(model, tok, prompt: str, length: int, temperature: float,
             stop_after_first_tool: bool = False):
    ids = tok.encode(prompt)
    generated = []
    previous = ""
    pieces = []
    bled = False
    eos = getattr(tok, "eos_id", 2)
    with torch.no_grad():
        for _ in range(length):
            ctx = torch.tensor(
                ids[-model.max_context :],
                dtype=torch.long,
                device=model.device,
            )
            logits = model(ctx)[0, -1] / max(temperature, 0.05)
            nid = int(torch.multinomial(torch.softmax(logits, -1), 1).item())
            generated.append(nid)
            ids.append(nid)
            if nid == eos:
                break
            full = tok.decode(generated)
            stopped = truncate_format_bleed(
                full, stop_after_first_tool=stop_after_first_tool
            )
            if len(stopped) < len(full):
                chunk = stopped[len(previous) :]
                if chunk:
                    pieces.append(chunk)
                bled = True
                break
            chunk = full[len(previous) :]
            if chunk:
                pieces.append(chunk)
                previous = full
    return "".join(pieces), bled


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
    results = []
    by = {"CODE": [0, 0], "TOOL": [0, 0], "MATH": [0, 0], "ID": [0, 0]}
    for domain, prompt, rules in TESTS:
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
        ok, why = score(domain, out, rules)
        tag = "PASS" if ok else "FAIL"
        print(f"->> {tag}: {why}", flush=True)
        results.append({"domain": domain, "pass": ok, "why": why, "prompt": prompt, "out": out})
        by[domain][0] += int(ok)
        by[domain][1] += 1
    summary = {k: f"{v[0]}/{v[1]}" for k, v in by.items()}
    code_p, code_n = by["CODE"]
    tool_p, tool_n = by["TOOL"]
    hard = {
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
        f"\nSUMMARY steps={summary} CODE={summary['CODE']} TOOL={summary['TOOL']} "
        f"combo={hard['coding_agent_pct']}% time={hard['seconds']}s",
        flush=True,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hard


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain-dir", default="/workspace/domain_brain_2b")
    ap.add_argument("--snap-dir", default="")
    ap.add_argument("--pattern", default="step-*.pt")
    ap.add_argument("--min-step", type=int, default=150000)
    ap.add_argument("--length", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--out-json", default="/workspace/eval_hard_all_snaps.json")
    args = ap.parse_args()

    snap_dir = args.snap_dir or str(Path(args.domain_dir) / "snapshots")
    paths = sorted(glob.glob(str(Path(snap_dir) / args.pattern)))
    filtered = []
    for p in paths:
        m = re.search(r"step-(\d+)", p)
        if not m:
            continue
        step = int(m.group(1))
        if step >= args.min_step:
            filtered.append((step, p))
    filtered.sort()
    if not filtered:
        raise SystemExit(f"No snapshots >= {args.min_step} in {snap_dir}")

    print(f"Evaluating {len(filtered)} snapshots on hard battery...", flush=True)
    for step, p in filtered:
        print(f"  - {step}: {p}", flush=True)

    tok = EchoTokenizer(f"{args.domain_dir}/echo_domain.model")
    all_sum = []
    for step, p in filtered:
        all_sum.append(run_ckpt(p, tok, args.length, args.temperature))

    print("\n" + "#" * 60, flush=True)
    print("LEADERBOARD (coding+agent %)", flush=True)
    ranked = sorted(all_sum, key=lambda r: (-r["coding_agent_pct"], -r["tool_pct"], -r["code_pct"]))
    for r in ranked:
        print(
            f"  step={r['steps']}  code={r['code']}  tool={r['tool']}  "
            f"combo={r['coding_agent_pct']}%  smooth={r['smooth']}",
            flush=True,
        )
    Path(args.out_json).write_text(json.dumps(all_sum, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
