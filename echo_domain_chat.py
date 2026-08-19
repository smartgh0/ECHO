#!/usr/bin/env python3
"""Interactive chat for Echo's subword domain-pretrained checkpoint."""

import os
import sys

import torch

from echo_tokenizer import EchoTokenizer
from echo_transformer import QuantumTransformerLM


ROOT = os.path.dirname(os.path.abspath(__file__))
DOMAIN_DIR = os.environ.get("ECHO_DOMAIN_DIR", os.path.join(ROOT, "domain_brain"))


def load_domain_model():
    checkpoint_path = os.path.join(DOMAIN_DIR, "model.pt")
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


def generate(model, tokenizer, prompt, length=120, temperature=0.7):
    token_ids = tokenizer.encode(prompt)
    generated = []
    model.eval()
    with torch.no_grad():
        for _ in range(length):
            context = torch.tensor(token_ids[-model.max_context:], dtype=torch.long, device=model.device)
            logits = model(context)[0, -1] / max(temperature, 0.05)
            probabilities = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probabilities, 1).item())
            generated.append(next_id)
            token_ids.append(next_id)
            if next_id == 2:
                break
    return tokenizer.decode(generated).strip()


def main():
    model, tokenizer = load_domain_model()
    print("ECHO DOMAIN MODEL")
    print(f"  parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"  tokenizer:  {tokenizer.vocab_size:,} subword pieces")
    print(f"  device:     {model.device}")
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
        answer = generate(model, tokenizer, f"user: {text}\necho:")
        print(f"echo> {answer or '...'}")


if __name__ == "__main__":
    main()
