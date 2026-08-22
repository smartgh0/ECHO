#!/usr/bin/env python3
"""Pretrain Echo's decoder on processed domain data.

Supports uniform stream sampling (legacy) or weighted, turn-boundary sampling
with optional user-token loss masking for coding/tool-focused training.

Example:
    python3 train_domain.py --steps 10000 --seq-len 256
    python3 train_domain.py --resume --weighted-sampling --mask-user-tokens --seq-len 512
"""

import argparse
import glob
import json
import os
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

# Coding + tools first, some general, wired identity boost.
CATEGORY_WEIGHTS = {
    "code": 0.50,
    "tools": 0.25,
    "general": 0.10,
    "identity": 0.05,
    "identity_boost": 0.10,
}
SFT_CATEGORY_WEIGHTS = {
    "code": 0.45,
    "tools": 0.35,
    "general": 0.05,
    "identity": 0.05,
    "identity_boost": 0.10,
}
# Mastery: keep coding/agent dominant, but allow heavy wiki/general boost.
MASTERY_CATEGORY_WEIGHTS = {
    "code": 0.38,
    "tools": 0.37,
    "general": 0.20,
    "identity": 0.025,
    "identity_boost": 0.025,
}
CATEGORY_IDS = {name: index for index, name in enumerate(CATEGORY_WEIGHTS)}
ID_TO_CATEGORY = {index: name for name, index in CATEGORY_IDS.items()}
# Quantum branch A (1) = coding/tools; branch B (0) = general/identity.
BRANCH_TARGET = {
    "code": 1.0,
    "tools": 1.0,
    "general": 0.0,
    "identity": 0.0,
    "identity_boost": 0.0,
}


def collect_files(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    if not files:
        raise SystemExit(f"No .txt files found in {input_dir}")
    return files


def categorize_file(path):
    name = os.path.basename(path).lower()
    if "identity_boost" in name:
        return "identity_boost"
    if "identity" in name:
        return "identity"
    if name.startswith("agent_") or name.startswith("tool_"):
        return "tools"
    code_markers = (
        "code", "stackoverflow", "gpt4_coding", "alpaca",
        "converted_datasets", "reasoning",
    )
    if any(marker in name for marker in code_markers):
        return "code"
    return "general"


def encode_corpus(files, tokenizer, token_path, mask_path, segment_path):
    """Encode corpus; write tokens, per-token train masks, and turn segments."""
    token_count = 0
    segments = []  # (start, length, category_id)
    seg_start = None
    seg_len = 0
    seg_cat = 0

    def flush_segment():
        nonlocal seg_start, seg_len
        if seg_start is not None and seg_len > 0:
            segments.append((seg_start, seg_len, seg_cat))
        seg_start = None
        seg_len = 0

    def append_eos():
        """Teach the model to end a turn instead of inventing user:/echo:."""
        nonlocal token_count, seg_len
        if seg_start is None or seg_len <= 0:
            return
        eos = getattr(tokenizer, "eos_id", 2)
        array("I", [eos]).tofile(token_handle)
        mask_handle.write(bytes([1]))
        token_count += 1
        seg_len += 1

    with open(token_path, "wb") as token_handle, open(mask_path, "wb") as mask_handle:
        for path in files:
            category = categorize_file(path)
            category_id = CATEGORY_IDS[category]
            with open(path, "r", encoding="utf-8", errors="ignore") as source:
                for line in source:
                    tokens = tokenizer.encode(line)
                    if not tokens:
                        continue
                    stripped = line.lstrip()
                    is_user = stripped.startswith("user:")
                    # Train on assistant/tool/think lines; skip pure user prompt tokens.
                    train_mask = 0 if is_user else 1
                    if is_user:
                        append_eos()
                        flush_segment()
                        seg_start = token_count
                        seg_len = 0
                        seg_cat = category_id
                    elif seg_start is None:
                        # Orphan non-user text (file headers etc.) — own segment.
                        seg_start = token_count
                        seg_len = 0
                        seg_cat = category_id

                    values = array("I", tokens)
                    values.tofile(token_handle)
                    mask_handle.write(bytes([train_mask]) * len(tokens))
                    token_count += len(tokens)
                    seg_len += len(tokens)
            append_eos()
            flush_segment()

    starts = np.array([segment[0] for segment in segments], dtype=np.uint64)
    lengths = np.array([segment[1] for segment in segments], dtype=np.uint32)
    cats = np.array([segment[2] for segment in segments], dtype=np.uint8)
    np.savez_compressed(segment_path, starts=starts, lengths=lengths, categories=cats)
    print(
        f"Encoded {token_count:,} tokens across {len(segments):,} turn segments "
        f"({', '.join(f'{ID_TO_CATEGORY[cid]}={int((cats == cid).sum())}' for cid in sorted(ID_TO_CATEGORY))})",
        flush=True,
    )
    return token_count


def sample_batch(tokens, count, seq_len, rng):
    """Legacy uniform random windows over the flat token stream."""
    starts = rng.integers(0, len(tokens) - seq_len - 1, size=count)
    inputs = [tokens[start:start + seq_len].tolist() for start in starts]
    targets = [tokens[start + 1:start + seq_len + 1].tolist() for start in starts]
    return inputs, targets, None


def build_category_segments(segment_path, seq_len):
    data = np.load(segment_path)
    starts = data["starts"]
    lengths = data["lengths"]
    categories = data["categories"]
    by_cat = {name: [] for name in CATEGORY_WEIGHTS}
    # Keep short turns too (identity/tools); sampler will pad up to seq_len.
    min_len = 8
    for start, length, category_id in zip(starts, lengths, categories):
        if int(length) > min_len:
            by_cat[ID_TO_CATEGORY[int(category_id)]].append((int(start), int(length)))
    return by_cat


def _pad_window(values, seq_len, pad_value):
    if len(values) >= seq_len:
        return list(values[:seq_len])
    return list(values) + [pad_value] * (seq_len - len(values))


def sample_batch_weighted(tokens, masks, by_cat, count, seq_len, rng, boundary_prob=0.7):
    """Sample windows from turn segments with category mixture weights."""
    available = [name for name, segs in by_cat.items() if segs]
    if not available:
        raise SystemExit("No segments available; rebuild tokens or check input data")
    weights = np.array([CATEGORY_WEIGHTS[name] for name in available], dtype=np.float64)
    weights /= weights.sum()

    inputs, targets, target_masks, branch_targets = [], [], [], []
    for _ in range(count):
        category = available[int(rng.choice(len(available), p=weights))]
        start, length = by_cat[category][int(rng.integers(0, len(by_cat[category])))]
        need = seq_len + 1
        if length > need:
            max_offset = length - need
            if rng.random() < boundary_prob:
                offset = 0
            else:
                offset = int(rng.integers(0, max_offset + 1))
            window = start + offset
            raw_in = tokens[window:window + seq_len]
            raw_tgt = tokens[window + 1:window + seq_len + 1]
            raw_mask = (
                masks[window + 1:window + seq_len + 1]
                if masks is not None else np.ones(seq_len, dtype=np.uint8)
            )
        else:
            # Short turn: take full segment and pad (pads get loss mask 0).
            window = start
            raw_in = tokens[window:window + length - 1]
            raw_tgt = tokens[window + 1:window + length]
            raw_mask = (
                masks[window + 1:window + length]
                if masks is not None else np.ones(length - 1, dtype=np.uint8)
            )
            raw_in = _pad_window(raw_in, seq_len, 0)
            raw_tgt = _pad_window(raw_tgt, seq_len, 0)
            raw_mask = _pad_window(np.asarray(raw_mask, dtype=np.float32), seq_len, 0.0)

        inputs.append(np.asarray(raw_in, dtype=np.uint32).tolist())
        targets.append(np.asarray(raw_tgt, dtype=np.uint32).tolist())
        branch_targets.append(BRANCH_TARGET[category])
        if masks is not None:
            target_masks.append(np.asarray(raw_mask, dtype=np.float32).tolist())
    return inputs, targets, (target_masks if masks is not None else None), branch_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--total-steps", type=int, default=None,
                        help="absolute step count the LR schedule should decay over; "
                             "defaults to --steps for a fresh run. Pass the true overall "
                             "target (e.g. 150000) on --resume so warmup/cosine decay "
                             "don't reset at every resume.")
    parser.add_argument("--profile", default="from-scratch-0.5b",
                        help="model profile: coherent-150m, from-scratch-0.5b, etc")
    parser.add_argument("--resume", action="store_true",
                        help="resume domain_brain/model.pt instead of starting over")
    parser.add_argument("--rebuild-tokens", action="store_true",
                        help="re-encode all input files after adding data")
    parser.add_argument("--weighted-sampling", action="store_true",
                        help="sample from turn boundaries with code/tools/general/identity weights")
    parser.add_argument("--mask-user-tokens", action="store_true",
                        help="ignore CE loss on user: prompt tokens (requires masks from encode)")
    parser.add_argument("--boundary-prob", type=float, default=0.7,
                        help="probability of starting a window at the user: boundary")
    parser.add_argument("--lr", type=float, default=None,
                        help="override learning rate on resume (e.g. 0.001)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override micro-batch size (useful when raising --seq-len)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1000,
                        help="save a checkpoint every N steps (0 disables periodic saving)")
    parser.add_argument("--snapshot-every", type=int, default=5000,
                        help="save a compact chat snapshot every N steps (0 disables snapshots)")
    parser.add_argument("--sft-mix", action="store_true",
                        help="use SFT category weights (more tools, less general dump)")
    parser.add_argument("--sft-mastery-mix", action="store_true",
                        help="SFT weights with heavier general/wiki while keeping code+tools dominant")
    parser.add_argument("--reset-lr-schedule", action="store_true",
                        help="decay LR over this run only, ignoring resumed absolute step")
    args = parser.parse_args()
    if args.sft_mastery_mix:
        CATEGORY_WEIGHTS.clear()
        CATEGORY_WEIGHTS.update(MASTERY_CATEGORY_WEIGHTS)
        print("Using mastery SFT category weights:", dict(CATEGORY_WEIGHTS))
    elif args.sft_mix:
        CATEGORY_WEIGHTS.clear()
        CATEGORY_WEIGHTS.update(SFT_CATEGORY_WEIGHTS)
        print("Using SFT category weights:", dict(CATEGORY_WEIGHTS))

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
    mask_path = os.path.join(args.output_dir, "masks.u8")
    segment_path = os.path.join(args.output_dir, "segments.npz")
    if args.rebuild_tokens:
        for path in (token_path, mask_path, segment_path):
            if os.path.exists(path):
                os.remove(path)
    if not os.path.exists(token_path) or (
        (args.weighted_sampling or args.mask_user_tokens)
        and (not os.path.exists(mask_path) or not os.path.exists(segment_path))
    ):
        print("Encoding corpus to streaming token storage...")
        token_count = encode_corpus(files, tokenizer, token_path, mask_path, segment_path)
    else:
        token_count = os.path.getsize(token_path) // array("I").itemsize
    print(f"Training tokens: {token_count:,}")
    if token_count <= args.seq_len + 1:
        raise SystemExit(
            f"Need more than {args.seq_len + 1} tokens, found {token_count}; add more text or reduce --seq-len"
        )

    tokens = np.memmap(token_path, dtype=np.uint32, mode="r")
    masks = None
    by_cat = None
    if args.mask_user_tokens or args.weighted_sampling:
        if not os.path.exists(mask_path) or not os.path.exists(segment_path):
            raise SystemExit("masks/segments missing; re-run with --rebuild-tokens")
        if args.mask_user_tokens:
            masks = np.memmap(mask_path, dtype=np.uint8, mode="r")
            if len(masks) != len(tokens):
                raise SystemExit("masks.u8 length does not match tokens.u32; rebuild tokens")
        if args.weighted_sampling:
            by_cat = build_category_segments(segment_path, args.seq_len)
            print(
                "Weighted sampling: "
                + ", ".join(f"{name}={len(by_cat[name]):,} segs (w={CATEGORY_WEIGHTS[name]})"
                            for name in CATEGORY_WEIGHTS),
                flush=True,
            )

    checkpoint_path = os.path.join(args.output_dir, "model.pt")
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        model = QuantumTransformerLM.from_dict({
            **config,
            "state_dict": checkpoint["state_dict"],
        })
        has_gate_proj = any("gate_proj" in key for key in checkpoint["state_dict"])
        if has_gate_proj and "optimizer" in checkpoint:
            try:
                model.optimizer.load_state_dict(checkpoint["optimizer"])
            except (ValueError, RuntimeError) as exc:
                print(f"Optimizer state incompatible ({exc}); rebuilding optimizer")
                model._build_optimizer()
        else:
            if not has_gate_proj:
                print("Checkpoint pre-dates conditional quantum gates; routers zero-init, fresh optimizer")
            model._build_optimizer()
        if args.lr is not None:
            model.learning_rate = args.lr
            for group in model.optimizer.param_groups:
                group["lr"] = args.lr * group.get("lr_scale", 1.0)
            print(f"Overriding learning rate to {args.lr} (gates x{model.quantum_gate_lr_scale})")
        if args.batch_size is not None:
            model.batch_size = args.batch_size
            print(f"Overriding batch size to {args.batch_size}")
        print(f"Resuming checkpoint at {model.total_epochs:,} steps")
    else:
        profile = get_transformer_profile(args.profile)
        if args.batch_size is not None:
            profile["batch_size"] = args.batch_size
        model = QuantumTransformerLM(
            vocab_size=tokenizer.vocab_size,
            profile_name=args.profile,
            **profile,
            seed=args.seed,
        )
    rng = np.random.default_rng(args.seed)
    print(model.info())
    mode = "weighted+boundary" if args.weighted_sampling else "uniform-stream"
    mask_mode = "mask-user" if args.mask_user_tokens else "all-tokens"
    print(f"Training on {model.device} for {args.steps:,} steps "
          f"(sample={mode}, loss={mask_mode}, seq_len={args.seq_len})...")

    def save_checkpoint(snapshot_step=None, write_latest=True):
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
        if write_latest:
            checkpoint_tmp = checkpoint_path + ".tmp"
            torch.save({
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "config": config,
                "optimizer": model.optimizer.state_dict(),
            }, checkpoint_tmp)
            os.replace(checkpoint_tmp, checkpoint_path)
            with open(os.path.join(args.output_dir, "training.json"), "w", encoding="utf-8") as handle:
                json.dump({
                    "steps": model.total_epochs,
                    "tokens": int(token_count),
                    "seq_len": args.seq_len,
                    "weighted_sampling": args.weighted_sampling,
                    "mask_user_tokens": args.mask_user_tokens,
                }, handle, indent=2)
        if snapshot_step is not None:
            snapshot_dir = os.path.join(args.output_dir, "snapshots")
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_path = os.path.join(snapshot_dir, f"step-{snapshot_step:06d}.pt")
            snapshot_state = {
                key: value.detach().cpu().to(torch.bfloat16)
                for key, value in model.state_dict().items()
            }
            torch.save({"state_dict": snapshot_state, "config": config}, snapshot_path)
            print(f"Chat snapshot saved at step {snapshot_step:,}", flush=True)

    def handle_signal(signum, _frame):
        print(f"\nReceived signal {signum}, saving checkpoint before exit...", flush=True)
        save_checkpoint()
        print(f"Saved domain checkpoint to {args.output_dir} at step {model.total_epochs:,}", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    resume_offset = model.total_epochs if args.resume else 0
    total_steps = args.total_steps if args.total_steps is not None else args.steps
    warmup_steps = min(500, (args.steps if args.reset_lr_schedule else total_steps) // 10)
    base_lr = model.learning_rate
    import math
    if args.reset_lr_schedule:
        print(f"LR schedule reset over this run ({args.steps} steps), resume_offset={resume_offset}")
    for step in range(args.steps):
        absolute_step = resume_offset + step
        schedule_step = step if args.reset_lr_schedule else absolute_step
        schedule_total = args.steps if args.reset_lr_schedule else total_steps
        if schedule_step < warmup_steps:
            current_lr = base_lr * (schedule_step + 1) / warmup_steps
        else:
            decay_steps = schedule_total - warmup_steps
            progress = (schedule_step - warmup_steps) / max(decay_steps, 1)
            current_lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        for group in model.optimizer.param_groups:
            group["lr"] = current_lr * group.get("lr_scale", 1.0)
        batch_count = model.batch_size * model.gradient_accumulation_steps
        if args.weighted_sampling:
            inputs, targets, target_masks, branch_targets = sample_batch_weighted(
                tokens, masks, by_cat, batch_count, args.seq_len, rng, args.boundary_prob,
            )
            loss, _ = model.train_step_batch(
                inputs, targets, mask_batch=target_masks, branch_targets=branch_targets,
            )
        elif args.mask_user_tokens:
            starts = rng.integers(0, len(tokens) - args.seq_len - 1, size=batch_count)
            inputs = [tokens[start:start + args.seq_len].tolist() for start in starts]
            targets = [tokens[start + 1:start + args.seq_len + 1].tolist() for start in starts]
            target_masks = [
                masks[start + 1:start + args.seq_len + 1].astype(np.float32).tolist()
                for start in starts
            ]
            loss, _ = model.train_step_batch(inputs, targets, mask_batch=target_masks)
        else:
            inputs, targets, _ = sample_batch(tokens, batch_count, args.seq_len, rng)
            loss, _ = model.train_step_batch(inputs, targets)
        model.total_epochs += 1
        if step % 50 == 0 or step == args.steps - 1:
            current_lr = model.optimizer.param_groups[0]["lr"]
            qs = model.quantum_stats()
            print(
                f"step {step + 1}/{args.steps} loss={loss:.4f} smooth={model.smooth_loss:.4f} "
                f"lr={current_lr:.6f} qaux={model.last_quantum_aux:.4f} "
                f"qent={qs['entropy_ratio']:.3f} qcommit={qs['committed_gates']}/{qs['total_gates']}",
                flush=True,
            )
        current_step = model.total_epochs
        if args.save_every and current_step % args.save_every == 0:
            save_checkpoint()
            print(f"Checkpoint saved at step {current_step:,}", flush=True)
        if args.snapshot_every and current_step % args.snapshot_every == 0:
            save_checkpoint(snapshot_step=current_step, write_latest=False)

    save_checkpoint()
    print(f"Saved domain checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
