#!/usr/bin/env python3
"""Contamination-resistant capability, context, and quantum-routing evaluation."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import torch
from torch.nn import functional as F

from echo_eval_fresh_heldout import run_ckpt as run_fresh
from echo_eval_hard_pod import generate
from echo_tokenizer import EchoTokenizer
from echo_transformer import QuantumTransformerLM


CODE_TASKS = (
    (
        "count_islands",
        "Python only: define count_islands(grid), returning the number of 4-connected "
        "groups of 1 values in a rectangular list of lists. Do not mutate grid.",
        (
            ("count_islands([[1,1,0],[0,1,0],[1,0,1]])", 3),
            ("count_islands([[0,0],[0,0]])", 0),
            ("count_islands([[1]])", 1),
        ),
    ),
    (
        "merge_intervals",
        "Code only: define merge_intervals(intervals) which merges overlapping closed "
        "integer intervals and returns them sorted. Do not use tools.",
        (
            ("merge_intervals([[1,3],[2,6],[8,10],[10,12]])", [[1, 6], [8, 12]]),
            ("merge_intervals([])", []),
            ("merge_intervals([[4,5],[1,2]])", [[1, 2], [4, 5]]),
        ),
    ),
    (
        "balanced_split",
        "Write only Python for balanced_split(nums): return an index i from 1 through "
        "len(nums)-1 where sum(nums[:i]) equals sum(nums[i:]), or -1 if absent.",
        (
            ("balanced_split([1,2,3,0,6])", 3),
            ("balanced_split([2,1,1,2])", 2),
            ("balanced_split([1,2,4])", -1),
        ),
    ),
)

KNOWLEDGE_TASKS = (
    (
        "math",
        "A tank is 3/5 full. After 24 liters are removed it is 1/3 full. "
        "Find the tank's capacity and show the equation.",
        ("90",),
    ),
    (
        "math",
        "A fair six-sided die is rolled twice. What is the exact probability that "
        "the sum is at least 10? Simplify the fraction.",
        ("1/6",),
    ),
    (
        "science",
        "Why does increasing atmospheric carbon dioxide warm the lower atmosphere? "
        "Answer using infrared absorption and energy balance, not slogans.",
        ("infrared", "absorb", "energy"),
    ),
    (
        "science",
        "Distinguish DNA replication from transcription in template, product, and "
        "main polymerase. Keep the comparison factual.",
        ("dna", "rna", "polymerase"),
    ),
    (
        "general",
        "Explain the difference between fiscal policy and monetary policy, naming "
        "the main institution responsible for each.",
        ("government", "central bank"),
    ),
)

SUMMARY_SOURCE = (
    "In 2024 the city installed heat pumps in twelve public buildings. The project "
    "cost $4.2 million, including $1.1 million from a state grant. Metered energy use "
    "fell by 18 percent over the following year, while maintenance calls fell from "
    "74 to 51. Officials cautioned that the winter was 9 percent warmer than the "
    "ten-year average, so they commissioned a weather-normalized audit before "
    "expanding the program. The council will vote in October on retrofits for eight "
    "additional buildings."
)


def load_model(path: str) -> tuple[QuantumTransformerLM, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = QuantumTransformerLM.from_dict(
        {**config, "state_dict": checkpoint["state_dict"]}
    )
    model.eval()
    return model, config


def extract_function(output: str, expected_name: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", output, flags=re.I | re.S)
    candidates = blocks + [output]
    for candidate in candidates:
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == expected_name:
                unsafe = (
                    ast.Import,
                    ast.ImportFrom,
                    ast.Attribute,
                    ast.Global,
                    ast.Nonlocal,
                    ast.With,
                    ast.AsyncWith,
                )
                if any(isinstance(child, unsafe) for child in ast.walk(node)):
                    return None
                if any(
                    isinstance(child, ast.Name)
                    and child.id.startswith("__")
                    for child in ast.walk(node)
                ):
                    return None
                return ast.unparse(node)
    return None


def execute_tests(code: str, tests: tuple[tuple[str, object], ...]) -> tuple[bool, str]:
    runner = textwrap.dedent(
        f"""
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
        safe = {{
            "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
            "enumerate": enumerate, "float": float, "int": int, "len": len,
            "list": list, "max": max, "min": min, "range": range, "reversed": reversed,
            "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
            "zip": zip, "True": True, "False": False, "None": None,
        }}
        ns = {{"__builtins__": safe}}
        exec(compile({code!r}, "<candidate>", "exec"), ns, ns)
        tests = {tests!r}
        for expression, expected in tests:
            actual = eval(expression, ns, ns)
            if actual != expected:
                raise AssertionError(f"{{expression}}: {{actual!r}} != {{expected!r}}")
        """
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", runner],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        return False, detail[-1][:240] if detail else "execution failed"
    return True, "all hidden tests passed"


@torch.no_grad()
def route_probability(model: QuantumTransformerLM, tok: EchoTokenizer, prompt: str) -> float:
    ids = tok.encode(prompt)[-min(model.max_context, 512) :]
    tensor = torch.tensor(ids, dtype=torch.long, device=model.device)
    model(tensor)
    values = []
    for projection in model.iter_quantum_linears():
        if projection._last_mix is not None:
            values.append(float(projection._last_mix.float().mean().cpu()))
    return sum(values) / max(len(values), 1)


def branch_cosines(model: QuantumTransformerLM) -> list[float]:
    values = []
    for projection in model.iter_quantum_linears():
        cosine = F.cosine_similarity(
            projection.weight_a.detach().float().reshape(-1),
            projection.weight_b.detach().float().reshape(-1),
            dim=0,
        )
        values.append(float(cosine.cpu()))
    return values


def build_retrieval_prompt(tok: EchoTokenizer, target_tokens: int, key: str, value: str) -> str:
    prefix = (
        "Read the notes and answer the question using the exact stored value. "
        f"Critical note: the value for {key} is {value}.\n"
    )
    filler_line = (
        "Routine archive note: sensors were checked, labels were reconciled, and "
        "ordinary records were retained without changing the critical note.\n"
    )
    text = prefix
    while len(tok.encode(text)) < target_tokens:
        text += filler_line
    return text + f"\nQuestion: What exact value was stored for {key}?\necho:"


def evaluate_capabilities(
    path: str,
    tok: EchoTokenizer,
    generation_length: int,
    temperature: float,
) -> dict:
    model, config = load_model(path)
    rows = []

    for name, prompt, tests in CODE_TASKS:
        output, bled = generate(
            model,
            tok,
            f"user: {prompt}\necho:",
            generation_length,
            temperature,
        )
        code = extract_function(output, name)
        passed, reason = (False, "no safe parseable function")
        if code is not None:
            passed, reason = execute_tests(code, tests)
        rows.append(
            {
                "domain": "code_exec",
                "name": name,
                "pass": passed,
                "reason": reason,
                "format_bleed": bled,
                "output": output,
            }
        )

    for domain, prompt, required in KNOWLEDGE_TASKS:
        output, bled = generate(
            model,
            tok,
            f"user: {prompt}\necho:",
            min(generation_length, 220),
            temperature,
        )
        lowered = output.lower()
        passed = all(term.lower() in lowered for term in required)
        rows.append(
            {
                "domain": domain,
                "pass": passed,
                "required": required,
                "format_bleed": bled,
                "output": output,
            }
        )

    summary_prompt = (
        "Summarize this source in two sentences. Preserve the numeric results and "
        "the stated caveat; do not add outside facts.\n\n" + SUMMARY_SOURCE
    )
    output, bled = generate(
        model,
        tok,
        f"user: {summary_prompt}\necho:",
        min(generation_length, 220),
        temperature,
    )
    summary_required = ("18", "warmer", "audit")
    summary_forbidden = ("solar", "2025", "twenty buildings")
    lowered = output.lower()
    rows.append(
        {
            "domain": "summary",
            "pass": all(term in lowered for term in summary_required)
            and not any(term in lowered for term in summary_forbidden),
            "required": summary_required,
            "format_bleed": bled,
            "output": output,
        }
    )

    for target, key, value in (
        (1024, "archive cobalt code", "R7-Q4-M2"),
        (1850, "archive amber code", "V9-K3-P6"),
    ):
        prompt = build_retrieval_prompt(tok, target, key, value)
        output, bled = generate(model, tok, prompt, 72, temperature)
        rows.append(
            {
                "domain": "long_context",
                "target_tokens": target,
                "prompt_tokens": len(tok.encode(prompt)),
                "pass": value.lower() in output.lower(),
                "format_bleed": bled,
                "output": output,
            }
        )

    symbolic_prompts = (
        "user: Implement binary search in Python and analyze its complexity.\necho:",
        "user: Solve 3x + 7 = 31 and verify the answer.\necho:",
        "user: Call the cache deletion tool for key sample_key_782.\necho:",
    )
    prose_prompts = (
        "user: Explain how photosynthesis transfers energy.\necho:",
        "user: Summarize the causes of urban heat islands.\necho:",
        "user: Introduce yourself and describe your broad knowledge.\necho:",
    )
    symbolic_route = sum(route_probability(model, tok, p) for p in symbolic_prompts) / len(
        symbolic_prompts
    )
    prose_route = sum(route_probability(model, tok, p) for p in prose_prompts) / len(
        prose_prompts
    )
    cosines = branch_cosines(model)
    quantum = {
        **model.quantum_stats(),
        "symbolic_branch_a": symbolic_route,
        "prose_branch_a": prose_route,
        "signed_domain_separation": symbolic_route - prose_route,
        "absolute_domain_separation": abs(symbolic_route - prose_route),
        "branch_cosine_mean": sum(cosines) / max(len(cosines), 1),
        "branch_cosine_abs_mean": sum(abs(x) for x in cosines) / max(len(cosines), 1),
    }

    passed = sum(int(row["pass"]) for row in rows)
    result = {
        "path": path,
        "steps": config.get("total_epochs"),
        "smooth_loss": config.get("smooth_loss"),
        "score": passed,
        "total": len(rows),
        "score_pct": round(100.0 * passed / max(len(rows), 1), 1),
        "by_domain": {},
        "quantum": quantum,
        "results": rows,
    }
    for domain in sorted({row["domain"] for row in rows}):
        selected = [row for row in rows if row["domain"] == domain]
        result["by_domain"][domain] = {
            "passed": sum(int(row["pass"]) for row in selected),
            "total": len(selected),
        }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-dir", default="/workspace/domain_brain_2b")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--length", type=int, default=360)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--out-json", default="/workspace/eval_capability.json")
    args = parser.parse_args()

    tok = EchoTokenizer(f"{args.domain_dir}/echo_domain.model")
    reports = []
    for path in (args.baseline, args.candidate):
        print(f"\nCAPABILITY EVAL: {path}", flush=True)
        capability = evaluate_capabilities(path, tok, args.length, args.temperature)
        fresh = run_fresh(path, tok, args.length, args.temperature)
        reports.append({"capability": capability, "fresh": fresh})
        print(
            f"score={capability['score']}/{capability['total']} "
            f"fresh={fresh['coding_agent_pct']}% quantum={capability['quantum']}",
            flush=True,
        )

    baseline, candidate = reports
    base_cap = baseline["capability"]
    cand_cap = candidate["capability"]
    base_fresh = baseline["fresh"]
    cand_fresh = candidate["fresh"]
    def passed_count(score: str) -> int:
        return int(score.split("/", 1)[0])

    identity_ok = passed_count(cand_fresh["id"]) >= passed_count(base_fresh["id"])
    tool_ok = cand_fresh["tool_pct"] >= base_fresh["tool_pct"] - 5.0
    semantic_ok = cand_cap["score_pct"] > base_cap["score_pct"]
    routing_ok = (
        cand_cap["quantum"]["absolute_domain_separation"]
        >= base_cap["quantum"]["absolute_domain_separation"]
        and cand_cap["quantum"]["branch_cosine_abs_mean"] < 0.98
    )
    decision = {
        "continue": bool(identity_ok and tool_ok and semantic_ok and routing_ok),
        "identity_preserved": identity_ok,
        "tool_preserved": tool_ok,
        "semantic_improved": semantic_ok,
        "routing_improved": routing_ok,
        "baseline_capability_pct": base_cap["score_pct"],
        "candidate_capability_pct": cand_cap["score_pct"],
        "baseline_fresh_pct": base_fresh["coding_agent_pct"],
        "candidate_fresh_pct": cand_fresh["coding_agent_pct"],
    }
    payload = {"reports": reports, "decision": decision}
    Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\nDECISION " + json.dumps(decision, indent=2), flush=True)
    print(f"Wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
