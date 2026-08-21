#!/usr/bin/env python3
"""Interactive chat for Echo's subword domain-pretrained checkpoint."""

import os
import sys
import argparse

import torch

from echo_tokenizer import EchoTokenizer
from echo_transformer import QuantumTransformerLM


ROOT = os.path.dirname(os.path.abspath(__file__))
DOMAIN_DIR = os.environ.get("ECHO_DOMAIN_DIR", os.path.join(ROOT, "domain_brain"))


def load_domain_model(checkpoint_path):
    tokenizer_path = os.path.join(DOMAIN_DIR, "echo_domain.model")
    if not os.path.exists(checkpoint_path) or not os.path.exists(tokenizer_path):
        print("Domain model not found.")
        print("Train it first with: python3 train_domain.py --steps 10000")
        raise SystemExit(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = QuantumTransformerLM.from_dict({
        **checkpoint["config"],
        "state_dict": checkpoint["state_dict"],
    })
    model.eval()
    return model, EchoTokenizer(tokenizer_path)


# Stop when the model starts hallucinating the next training turn.
STOP_MARKERS = ("\nuser:", "\nuser :", "\necho:", "\necho :")


def _truncate_at_stop(text):
    """Cut generated text at the first fake next-turn marker, if any."""
    cut = len(text)
    for marker in STOP_MARKERS:
        index = text.find(marker)
        if index != -1:
            cut = min(cut, index)
    return text[:cut].rstrip()


def generate(model, tokenizer, prompt, length=400, temperature=0.3):
    """Yield text chunks as tokens are sampled, so callers can print incrementally."""
    token_ids = tokenizer.encode(prompt)
    generated = []
    previous_text = ""
    model.eval()
    with torch.no_grad():
        for _ in range(length):
            context = torch.tensor(token_ids[-model.max_context:], dtype=torch.long, device=model.device)
            logits = model(context)[0, -1]
            if temperature <= 1e-5:
                next_id = int(torch.argmax(logits).item())
            else:
                logits = logits / max(temperature, 0.05)
                next_id = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
            generated.append(next_id)
            token_ids.append(next_id)
            if next_id == 2:
                break
            # Re-decode the whole run so far; SentencePiece needs full context to place spaces correctly.
            full_text = tokenizer.decode(generated)
            stopped = _truncate_at_stop(full_text)
            if len(stopped) < len(full_text):
                chunk = stopped[len(previous_text):]
                if chunk:
                    yield chunk
                break
            chunk = full_text[len(previous_text):]
            if chunk:
                yield chunk
                previous_text = full_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint path; defaults to the latest model.pt")
    parser.add_argument("--max-new-tokens", type=int, default=200,
                        help="maximum tokens to generate per reply")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="sampling temperature (lower = more deterministic; 0 = greedy)")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint or os.path.join(DOMAIN_DIR, "model.pt")
    model, tokenizer = load_domain_model(checkpoint_path)
    print("ECHO DOMAIN MODEL")
    print(f"  checkpoint:  {checkpoint_path}")
    print(f"  trained at:  step {model.total_epochs:,}")
    print(f"  parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"  tokenizer:  {tokenizer.vocab_size:,} subword pieces")
    print(f"  device:     {model.device}")
    print(f"  temperature:{args.temperature}")
    print("Type a message. Commands: :info, :quit")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == ":quit":
            break
        if text == ":info":
            print(model.info())
            continue
        answer = generate(
            model,
            tokenizer,
            f"user: {text}\necho:",
            length=args.max_new_tokens,
            temperature=args.temperature,
        )
        print("echo> ", end="", flush=True)
        got_output = False
        for chunk in answer:
            got_output = True
            print(chunk, end="", flush=True)
        print("..." if not got_output else "")


if __name__ == "__main__":
    main()
