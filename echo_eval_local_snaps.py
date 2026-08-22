#!/usr/bin/env python3
"""Compare generalize-SFT snapshots locally (GPU when available)."""

from __future__ import annotations

import argparse
import os
import time

import torch

from echo_tokenizer import EchoTokenizer
from echo_transformer import QuantumTransformerLM
from echo_format import truncate_format_bleed

ROOT = os.path.dirname(os.path.abspath(__file__))
DOMAIN = os.path.join(ROOT, "domain_brain_2b")

TESTS = [
    ("GOLD-CODE", "Write a Python function that reverses a string. Just the code, no tools."),
    ("GOLD-CODE", "Write a Python function that checks if a string is a palindrome. Include a docstring and 2 assert tests."),
    ("GOLD-TOOL", "Delete the cache key 'temp_data'."),
    ("GOLD-TOOL", "Create a new file called notes.txt with the content 'Meeting at 3pm'."),
    ("GOLD-MATH", "Solve for x: 3x + 7 = 22. Show steps."),
    ("GOLD-ID", "who are you"),
    ("HELD-CODE", "Write a Python function that returns the nth Fibonacci number. Just the code."),
    ("HELD-CODE", "Write count_vowels(s) that returns how many vowels are in s."),
    ("HELD-CODE", "Write a binary_search(arr, target) function. Just the code, no tools."),
    ("HELD-TOOL", "Create a file called memo.txt with content Ship by Friday."),
    ("HELD-TOOL", "Delete the cache key 'refresh_token'."),
    ("HELD-TOOL", "Show me what's in this folder using a tool."),
    ("HELD-MATH", "Solve for z: 7z - 3 = 18. Show steps."),
    ("TRANSFER", "Explain photosynthesis in 3 sentences for a high school student."),
]


def run_ckpt(path: str, length: int = 400, temperature: float = 0.3) -> None:
    print(f"\n######## {path} ########", flush=True)
    t0 = time.time()
    tok = EchoTokenizer(os.path.join(DOMAIN, "echo_domain.model"))
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = QuantumTransformerLM.from_dict({**cfg, "state_dict": ckpt["state_dict"]})
    model.eval()
    print(
        f"steps={cfg.get('total_epochs')} smooth={cfg.get('smooth_loss')} "
        f"device={model.device} load={time.time() - t0:.1f}s",
        flush=True,
    )
    eos = getattr(tok, "eos_id", 2)

    def generate(prompt: str, stop_after_first_tool: bool = False):
        ids = tok.encode(prompt)
        generated = []
        previous = ""
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
                        yield chunk, True
                    return
                chunk = full[len(previous) :]
                if chunk:
                    yield chunk, False
                    previous = full

    for domain, prompt in TESTS:
        print(f"\n{'=' * 60}\n[{domain}] user: {prompt}\n{'-' * 60}\necho> ", end="", flush=True)
        bled = False
        for chunk, did_bleed in generate(
            f"user: {prompt}\necho:",
            stop_after_first_tool=("TOOL" in domain),
        ):
            print(chunk, end="", flush=True)
            bled = bled or did_bleed
        if bled:
            print("\n[[FORMAT BLEED]]", end="", flush=True)
        print(flush=True)
    print(f"\nDONE {path} in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "ckpts",
        nargs="*",
        default=[
            os.path.join(DOMAIN, "snapshots", "step-156000.pt"),
        ],
    )
    ap.add_argument("--length", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()
    for path in args.ckpts:
        if not os.path.isfile(path):
            print(f"SKIP missing {path}", flush=True)
            continue
        run_ckpt(path, length=args.length, temperature=args.temperature)


if __name__ == "__main__":
    main()
