#!/usr/bin/env python3
"""Pretrain Echo's 0.5B decoder from scratch on processed domain data.

Example:
    python3 train_domain.py --steps 10000 --seq-len 256
"""

import argparse
import glob
import json
import os
import random
import signal
import sys
from array import array

import numpy as np
import torch

from echo_model_config import get_transformer_profile
from echo_tokenizer import EchoTokenizer, train_tokenizer
from echo_transformer import QuantumTransformerLM


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(ROOT, "pipeline", "input", "processed")
DEFAULT_OUTPUT = os.path.join(ROOT, "domain_brain")


def collect_files(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    if not files:
        raise SystemExit(f"No .txt files found in {input_dir}")
    return files


def encode_corpus(files, tokenizer, token_path):
    """Encode one file at a time so the full corpus is never held in RAM."""
    token_count = 0
    with open(token_path, "wb") as handle:
        for path in files:
            with open(path, "r", encoding="utf-8", errors="ignore") as source:
                for line in source:
                    tokens = tokenizer.encode(line)
                    values = array("I", tokens)
                    values.tofile(handle)
                    token_count += len(tokens)
    return token_count


def sample_batch(tokens, count, seq_len, rng):
    starts = rng.integers(0, len(tokens) - seq_len - 1, size=count)
    inputs = [tokens[start:start + seq_len].tolist() for start in starts]
    targets = [tokens[start + 1:start + seq_len + 1].tolist() for start in starts]
    return inputs, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--profile", default="from-scratch-0.5b",
                        help="model profile: coherent-150m, from-scratch-0.5b, etc")
    parser.add_argument("--resume", action="store_true",
                        help="resume domain_brain/model.pt instead of starting over")
    parser.add_argument("--rebuild-tokens", action="store_true",
                        help="re-encode all input files after adding data")
    parser.add_argument("--lr", type=float, default=None,
                        help="override learning rate on resume (e.g. 0.001)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1000,
                        help="save a checkpoint every N steps (0 disables periodic saving)")
    args = parser.parse_args()

    files = collect_files(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, "echo_domain")
    tokenizer_path = prefix + ".model"
    if os.path.exists(tokenizer_path):
        tokenizer = EchoTokenizer(tokenizer_path)
        print(f"Using tokenizer: {tokenizer.vocab_size:,} pieces")
    else:
        print(f"Training tokenizer from {len(files)} files...")
        tokenizer = train_tokenizer(files, prefix, args.vocab_size)
        tokenizer.save_metadata(prefix + ".json")
        print(f"Tokenizer vocabulary: {tokenizer.vocab_size:,} pieces")

    token_path = os.path.join(args.output_dir, "tokens.u32")
    if args.rebuild_tokens and os.path.exists(token_path):
        os.remove(token_path)
    if not os.path.exists(token_path):
        print("Encoding corpus to streaming token storage...")
        token_count = encode_corpus(files, tokenizer, token_path)
    else:
        token_count = os.path.getsize(token_path) // array("I").itemsize
    print(f"Training tokens: {token_count:,}")
    if token_count <= args.seq_len + 1:
        raise SystemExit(
            f"Need more than {args.seq_len + 1} tokens, found {token_count}; add more text or reduce --seq-len"
        )

    tokens = np.memmap(token_path, dtype=np.uint32, mode="r")
    checkpoint_path = os.path.join(args.output_dir, "model.pt")
    checkpoint = None
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        model = QuantumTransformerLM.from_dict({
            **config,
            "state_dict": checkpoint["state_dict"],
        })
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer" in checkpoint:
            model.optimizer.load_state_dict(checkpoint["optimizer"])
        if args.lr is not None:
            for group in model.optimizer.param_groups:
                group["lr"] = args.lr
            model.learning_rate = args.lr
            print(f"Overriding learning rate to {args.lr}")
        print(f"Resuming checkpoint at {model.total_epochs:,} steps")
    else:
        profile = get_transformer_profile(args.profile)
        model = QuantumTransformerLM(
            vocab_size=tokenizer.vocab_size,
            profile_name=args.profile,
            **profile,
            seed=args.seed,
        )
    rng = np.random.default_rng(args.seed)
    print(model.info())
    print(f"Training on {model.device} for {args.steps:,} steps...")

    def save_checkpoint():
        config = {
            "vocab_size": model.vocab_size,
            "d_model": model.d_model,
            "n_layers": model.n_layers,
            "n_heads": model.n_heads,
            "n_kv_heads": model.n_kv_heads,
            "ff_multiplier": model.ff_multiplier,
            "max_context": model.max_context,
            "learning_rate": model.learning_rate,
            "batch_size": model.batch_size,
            "gradient_accumulation_steps": model.gradient_accumulation_steps,
            "lora_rank": model.lora_rank,
            "freeze_base": model.freeze_base,
            "optimizer": model.optimizer_name,
            "gradient_checkpointing": model.gradient_checkpointing,
            "profile_name": model.profile_name,
            "seed": model.seed,
            "total_epochs": model.total_epochs,
            "total_chars_seen": model.total_chars_seen,
            "smooth_loss": model.smooth_loss,
        }
        checkpoint_tmp = checkpoint_path + ".tmp"
        torch.save({
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "config": config,
            "optimizer": model.optimizer.state_dict(),
        }, checkpoint_tmp)
        os.replace(checkpoint_tmp, checkpoint_path)
        with open(os.path.join(args.output_dir, "training.json"), "w", encoding="utf-8") as handle:
            json.dump({"steps": model.total_epochs, "tokens": int(token_count), "seq_len": args.seq_len}, handle, indent=2)

    def handle_signal(signum, _frame):
        print(f"\nReceived signal {signum}, saving checkpoint before exit...", flush=True)
        save_checkpoint()
        print(f"Saved domain checkpoint to {args.output_dir} at step {model.total_epochs:,}", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    warmup_steps = min(500, args.steps // 10)
    base_lr = model.learning_rate
    import math
    for step in range(args.steps):
        if step < warmup_steps:
            current_lr = base_lr * (step + 1) / warmup_steps
        else:
            decay_steps = args.steps - warmup_steps
            progress = (step - warmup_steps) / max(decay_steps, 1)
            current_lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in model.optimizer.param_groups:
            group["lr"] = current_lr
        inputs, targets = sample_batch(
            tokens,
            model.batch_size * model.gradient_accumulation_steps,
            args.seq_len,
            rng,
        )
        loss, _ = model.train_step_batch(inputs, targets)
        model.total_epochs += 1
        if step % 50 == 0 or step == args.steps - 1:
            current_lr = model.optimizer.param_groups[0]["lr"]
            print(f"step {step + 1}/{args.steps} loss={loss:.4f} smooth={model.smooth_loss:.4f} lr={current_lr:.6f}", flush=True)
        if args.save_every and (step + 1) % args.save_every == 0:
            save_checkpoint()
            print(f"Checkpoint saved at step {step + 1:,}", flush=True)

    save_checkpoint()
    print(f"Saved domain checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
