#!/usr/bin/env python3
"""Gold vs frozen held-out battery for generalization checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from echo_tokenizer import EchoTokenizer
from echo_transformer import QuantumTransformerLM

STOP = ("\nuser:", "\nuser :", "\necho:", "\necho :")

GOLD = [
    ("GOLD-CODE", "Write a Python function that checks if a string is a palindrome. Include a docstring and 2 assert tests."),
    ("GOLD-CODE", "Fix this bug:\n```python\ndef average(nums):\n    return sum(nums) / len(nums)\nprint(average([]))\n```\nExplain the fix briefly."),
    ("GOLD-CODE", "Write a SQL query to get the top 5 customers by total order amount from tables customers(id,name) and orders(id,customer_id,amount)."),
    ("GOLD-CODE", "Write a Python function that reverses a string. Just the code, no tools."),
    ("GOLD-TOOL", "Create a new file called notes.txt with the content 'Meeting at 3pm'."),
    ("GOLD-TOOL", "Delete the cache key 'temp_data'."),
    ("GOLD-TOOL", "List files in the current directory using a tool."),
    ("GOLD-MATH", "Solve for x: 3x + 7 = 22. Show steps."),
    ("GOLD-ID", "who are you"),
]

DEFAULT_HELDOUT = [
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

TRANSFER = [
    ("SCIENCE", "Explain photosynthesis in 3 sentences for a high school student."),
    ("DEBUG", "What does this error mean and how do I fix it?\nTypeError: 'NoneType' object is not subscriptable"),
    ("EVERYDAY", "How do I make scrambled eggs?"),
]


def trunc(text: str) -> str:
    cut = len(text)
    for marker in STOP:
        index = text.find(marker)
        if index != -1:
            cut = min(cut, index)
    return text[:cut].rstrip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--domain-dir", default="/workspace/domain_brain_2b")
    ap.add_argument("--heldout-json", default="")
    ap.add_argument("--length", type=int, default=140)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    tok = EchoTokenizer(f"{args.domain_dir}/echo_domain.model")
    print(f"Loading {args.ckpt}...", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = QuantumTransformerLM.from_dict({**cfg, "state_dict": ckpt["state_dict"]})
    model.eval()
    print(
        f"steps={cfg.get('total_epochs')} smooth={cfg.get('smooth_loss')} device={model.device}",
        flush=True,
    )

    held = list(DEFAULT_HELDOUT)
    if args.heldout_json:
        data = json.loads(Path(args.heldout_json).read_text(encoding="utf-8"))
        held = list(data.get("prompts", held))

    tests = list(GOLD)
    tests.extend(("HELD-OUT", p) for p in held)
    tests.extend(TRANSFER)

    def generate(prompt: str):
        ids = tok.encode(prompt)
        generated = []
        previous = ""
        with torch.no_grad():
            for _ in range(args.length):
                ctx = torch.tensor(
                    ids[-min(model.max_context, 512) :],
                    dtype=torch.long,
                    device=model.device,
                )
                logits = model(ctx)[0, -1] / max(args.temperature, 0.05)
                nid = int(torch.multinomial(torch.softmax(logits, -1), 1).item())
                generated.append(nid)
                ids.append(nid)
                if nid == 2:
                    break
                full = tok.decode(generated)
                stopped = trunc(full)
                if len(stopped) < len(full):
                    chunk = stopped[len(previous) :]
                    if chunk:
                        yield chunk, True
                    return
                chunk = full[len(previous) :]
                if chunk:
                    yield chunk, False
                    previous = full

    for domain, prompt in tests:
        print(f"\n{'=' * 60}\n[{domain}] user: {prompt}\n{'-' * 60}\necho> ", end="", flush=True)
        bled = False
        for chunk, did_bleed in generate(f"user: {prompt}\necho:"):
            print(chunk, end="", flush=True)
            bled = bled or did_bleed
        if bled:
            print("\n[[FORMAT BLEED stopped]]", end="", flush=True)
        print(flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
