#!/usr/bin/env python3
"""Blank-slate verification: distill Qwen2-1.5B into a FRESH Echo_Q.

This is the real test of the intelligence-sucking approach. The student is a
randomly-initialized Echo_Q (echo_Q profile, 302M params, NO pretraining).
If the hidden-state distillation truly transfers Qwen's layer intelligence,
this fresh model should acquire Qwen-like representations and behavior
purely from the rep-matching loss + CE on the distillation corpus.

Method (same as echo_intel_distill.py, but student is fresh):
    L = CE(next-token) + lambda * sum_l (1 - cos(R_l h_teacher_l, h_student_l))

A fresh model can't match a trained teacher's CE, so this run uses a higher
lambda and more steps. Output: echo_Q_blank/ with model.pt + snapshots.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from echo_model_config import get_transformer_profile
from echo_tokenizer import EchoTokenizer
from echo_transformer import QuantumTransformerLM

QWEN_DIR = ROOT / "Qwen1_5b"
ECHO_TOK_DIR = ROOT / "echo_Q_ft"
OUT_DIR = ROOT / "echo_Q_blank"

LAYER_MAP = [(4, 2), (9, 5), (14, 8), (19, 11), (24, 14)]
MAX_LEN = 192

CORPUS = [
    # identity / chat
    "user: Who are you?\necho: I'm Echo, a language model trained by Solomon Nyamekye.",
    "user: What can you do?\necho: I can answer questions, write code, explain ideas, and reason through problems step by step.",
    "user: Who trained you?\necho: Solomon Nyamekye trained me. I'm Echo.",
    "user: Introduce yourself briefly.\necho: I'm Echo — built and trained by Solomon Nyamekye.",
    "user: What is your name?\necho: My name is Echo.",
    # reasoning
    "user: A farmer has 17 sheep. All but 9 run away. How many are left?\necho: 9 sheep remain. 'All but 9' means exactly 9 did not run away.",
    "user: A bat and ball cost $1.10. The bat costs $1.00 more. How much is the ball?\necho: The ball costs $0.05. If the ball were $0.10, the bat would be $1.10 and the total $1.20.",
    "user: All cats are animals. Tom is a cat. What is Tom?\necho: Tom is an animal. All cats are animals, and Tom is a cat, so Tom must be an animal.",
    "user: If today is Wednesday, what day is it 3 days from now?\necho: Saturday. Wednesday plus 3 days is Saturday.",
    "user: I have 3 apples. I eat 1 and give 1 away. How many left?\necho: 1 apple. 3 minus 1 eaten minus 1 given away leaves 1.",
    "user: Alice is taller than Bob. Bob is taller than Carol. Who is tallest?\necho: Alice is the tallest. Alice > Bob > Carol by transitivity.",
    "user: What gets wetter the more it dries?\necho: A towel.",
    # math
    "user: What is 17 times 23?\necho: 391. 17 times 20 is 340, and 17 times 3 is 51; 340 plus 51 is 391.",
    "user: Solve for x: 2x + 5 = 15.\necho: x = 5. Subtract 5 from both sides: 2x = 10, so x = 5.",
    "user: What is the derivative of x^2?\necho: 2x. The power rule gives d/dx x^n = n*x^(n-1), so d/dx x^2 = 2x.",
    "user: A pizza is cut into 8 slices. You eat 3. What fraction remains?\necho: 5/8 of the pizza remains.",
    "user: What is 2 plus 2?\necho: 4.",
    "user: Explain the Pythagorean theorem.\necho: In a right triangle, a squared plus b squared equals c squared, where c is the hypotenuse.",
    # code
    "user: Write a Python function that reverses a string.\necho: def reverse(s): return s[::-1]",
    "user: Implement binary search in Python.\necho: def bsearch(arr, target): lo, hi = 0, len(arr) - 1; while lo <= hi: mid = (lo + hi) // 2; if arr[mid] == target: return mid; elif arr[mid] < target: lo = mid + 1; else: hi = mid - 1; return -1",
    "user: What is the difference between a list and a tuple?\necho: Lists are mutable, tuples are immutable. Lists use square brackets, tuples use parentheses.",
    "user: Write a function to check if a string is a palindrome.\necho: def is_palindrome(s): s = s.lower().replace(' ', ''); return s == s[::-1]",
    "user: Write a function that adds two numbers.\necho: def add(a, b): return a + b",
    "user: What is a dictionary in Python?\necho: A dictionary stores key-value pairs. dict = {'key': 'value'} allows fast lookup by key.",
    # knowledge
    "user: What caused World War I?\necho: The assassination of Archduke Franz Ferdinand triggered alliance systems that turned a local conflict into a global war.",
    "user: How does photosynthesis work?\necho: Plants convert sunlight into chemical energy, using CO2 and water to produce glucose and oxygen in chloroplasts.",
    "user: Explain entropy.\necho: Entropy measures disorder or uncertainty. In physics it is the number of microstates; in information theory it is the expected information content.",
    "user: At what temperature does water boil?\necho: Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "user: What is the capital of France?\necho: Paris.",
    "user: Who wrote Romeo and Juliet?\necho: William Shakespeare wrote Romeo and Juliet.",
    # tools
    "user: List the files in this directory.\necho: think: I'll list the files with the shell tool.\ntool: shell_run command=\"ls -la\"",
    "user: What is the current date?\necho: think: I'll check the date with the shell tool.\ntool: shell_run command=\"date\"",
    # prose / general
    "The quick brown fox jumps over the lazy dog near the river bank at dawn.",
    "In a distant kingdom, a young knight set out to find the lost crown of the old king.",
    "The stock market rose three percent after the earnings report was published.",
    "History shows that empires rise and fall with trade, war, and technology.",
    "The recipe calls for two cups of flour, one egg, and a pinch of salt mixed well.",
    "Water flows downhill, carving valleys over millions of years of erosion.",
]


def load_qwen(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(QWEN_DIR))
    model = AutoModelForCausalLM.from_pretrained(str(QWEN_DIR), torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    return model, tok


def build_fresh_echo(device):
    """Randomly-initialized Echo_Q — no pretraining, no checkpoint."""
    profile = get_transformer_profile("echo_Q")
    model = QuantumTransformerLM(
        vocab_size=16384,
        d_model=profile["d_model"],
        n_layers=profile["n_layers"],
        n_heads=profile["n_heads"],
        n_kv_heads=profile.get("n_kv_heads", profile["n_heads"]),
        ff_multiplier=profile["ff_multiplier"],
        max_context=profile["max_context"],
        learning_rate=profile["learning_rate"],
        batch_size=1,
        gradient_accumulation_steps=1,
        optimizer="adamw",
        gradient_checkpointing=False,
        use_bayesian_ffn=profile["use_bayesian_ffn"],
        bayesian_prior_sigma=profile["bayesian_prior_sigma"],
        bayesian_sigma_init=profile["bayesian_sigma_init"],
        bayesian_kl_beta_start=profile["bayesian_kl_beta_start"],
        bayesian_kl_beta_end=profile["bayesian_kl_beta_end"],
        bayesian_kl_anneal_steps=profile["bayesian_kl_anneal_steps"],
        device=str(device),
    )
    model.eval()
    return model


@torch.no_grad()
def capture_teacher(qwen, qtok, texts, device):
    handles = []
    captured = {ql: [] for ql, _ in LAYER_MAP}
    for ql, _ in LAYER_MAP:
        def make_hook(layer_idx):
            def hook(_m, _i, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_idx].append(hidden.detach().float().cpu())
            return hook
        handles.append(qwen.model.layers[ql].register_forward_hook(make_hook(ql)))
    for text in texts:
        ids = qtok(text, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] > MAX_LEN:
            ids = ids[:, :MAX_LEN]
        qwen(ids)
    for h in handles:
        h.remove()
    return captured


def compute_alignments(qwen_acts, echo_acts):
    Rs = {}
    for (ql, el) in LAYER_MAP:
        Xq = torch.stack([a.mean(dim=1).squeeze(0) for a in qwen_acts[ql]])
        Xe = torch.stack([a.mean(dim=1).squeeze(0) for a in echo_acts[el]])
        Xq_c = Xq - Xq.mean(0, keepdim=True)
        Xe_c = Xe - Xe.mean(0, keepdim=True)
        M = Xq_c.T @ Xe_c
        U, _S, Vh = torch.linalg.svd(M, full_matrices=False)
        R = U @ Vh
        proj = Xq_c @ R
        cos = F.cosine_similarity(proj, Xe_c, dim=1).mean()
        Rs[(ql, el)] = R
        print(f"  layer {ql}->{el}: cos={cos:.4f}")
    return Rs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-rep", type=float, default=4.0,
                    help="rep-loss weight; higher for a fresh student")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT_DIR.mkdir(exist_ok=True)

    etok = EchoTokenizer(str(ECHO_TOK_DIR / "echo_domain.model"))

    print("Loading Qwen teacher and capturing layer targets...")
    qwen, qtok = load_qwen(device)
    teacher_acts = capture_teacher(qwen, qtok, CORPUS, device)
    del qwen
    torch.cuda.empty_cache() if device.type == "cuda" else None
    print(f"Captured {len(CORPUS)} texts at Qwen layers {[q for q, _ in LAYER_MAP]}")

    print("Building FRESH Echo_Q student (random init, no pretraining)...")
    student = build_fresh_echo(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"Fresh Echo_Q: {n_params:,} params, {student.n_layers} layers, d={student.d_model}")
    # Free the teacher's GPU memory before training the student.
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # Alignments from the fresh student's (random) representations
    print("Computing per-layer Procrustes alignments (fresh student)...")
    ids_list = [etok.encode(t)[:MAX_LEN] for t in CORPUS]
    student_acts = {}
    handles = []
    for _, el in LAYER_MAP:
        def make_hook(layer_idx):
            def hook(_m, _i, output):
                student_acts.setdefault(layer_idx, []).append(output.detach().float().cpu())
            return hook
        handles.append(student.blocks[el].register_forward_hook(make_hook(el)))
    with torch.no_grad():
        for ids in ids_list:
            t = torch.tensor([ids], dtype=torch.long, device=device)
            student(t)
    for h in handles:
        h.remove()
    Rs = compute_alignments(teacher_acts, student_acts)

    # CE windows from corpus
    ce_data = []
    for text in CORPUS:
        ids = etok.encode(text)
        if len(ids) < 10:
            continue
        for start in range(0, max(1, len(ids) - args.seq_len), args.seq_len // 2):
            window = ids[start:start + args.seq_len + 1]
            if len(window) < 10:
                continue
            ce_data.append((window[:-1], window[1:]))

    # Rep targets
    rep_ids = [etok.encode(t)[:MAX_LEN] for t in CORPUS]
    rep_targets = {}
    for (ql, el) in LAYER_MAP:
        Xq = torch.stack([a.mean(dim=1).squeeze(0) for a in teacher_acts[ql]])
        Xq_c = Xq - Xq.mean(0, keepdim=True)
        R = Rs[(ql, el)]
        rep_targets[el] = Xq_c @ R  # [N, 1024]
    # Teacher activations are no longer needed — free them before training.
    del teacher_acts, Rs, student_acts
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Training fresh student: {args.steps} steps, lr={args.lr}, lambda={args.lambda_rep}")
    # Stay in eval mode: train() + gradient checkpointing re-runs the forward
    # during backward and the Bayesian FFN's stochastic sampling triples
    # activation memory on the 8GB GPU. Gradients still flow in eval mode.
    student.eval()
    student.gradient_checkpointing = False
    student.learning_rate = args.lr
    student._build_optimizer()
    student.quantum_hard_infer = False
    student.quantum_expert_stopgrad = False

    step = 0
    t0 = time.time()
    while step < args.steps:
        # --- CE pass: forward + backward immediately (frees graph) ---
        idx = step % len(ce_data)
        inputs, targets = ce_data[idx]
        t_in = torch.tensor([inputs], dtype=torch.long, device=device)
        t_tgt = torch.tensor([targets], dtype=torch.long, device=device)
        logits = student(t_in)
        ce_loss = F.cross_entropy(
            logits.reshape(-1, student.vocab_size), t_tgt.reshape(-1)
        )
        ce_value = ce_loss.item()
        ce_loss.backward()
        del logits, ce_loss

        # --- Rep pass: hooks capture NON-detached outputs so gradients flow ---
        ridx = step % len(rep_ids)
        rids = rep_ids[ridx]
        t_rep = torch.tensor([rids], dtype=torch.long, device=device)
        cap = {}
        handles = []
        for _, el in LAYER_MAP:
            def make_hook(layer_idx):
                def hook(_m, _i, output):
                    cap[layer_idx] = output  # keep graph: rep loss must backprop
                return hook
            handles.append(student.blocks[el].register_forward_hook(make_hook(el)))
        _ = student(t_rep)
        for h in handles:
            h.remove()

        rep_loss = torch.zeros((), device=device)
        for (ql, el) in LAYER_MAP:
            h_student = cap[el].float().mean(dim=1).squeeze(0)
            target_vec = rep_targets[el][ridx].to(device)
            rep_loss = rep_loss + (1.0 - F.cosine_similarity(
                h_student.unsqueeze(0), target_vec.unsqueeze(0)
            ).squeeze())
        rep_loss = rep_loss / len(LAYER_MAP)
        rep_value = rep_loss.item()
        (args.lambda_rep * rep_loss).backward()
        del cap, _

        # Gradients from both passes have accumulated; step once.
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        student.optimizer.step()
        student.optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % 25 == 0 or step == 1:
            print(f"  step {step}/{args.steps} ce={ce_value:.4f} "
                  f"rep={rep_value:.4f} "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
        if step % args.save_every == 0:
            save(student, step, args)

    save(student, step, args, final=True)
    print("Done. Blank-slate distilled model in echo_Q_blank/")


def save(model, step, args, final=False):
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
        "quantum_gate_lr_scale": model.quantum_gate_lr_scale,
        "profile_name": model.profile_name,
        "seed": model.seed,
        "use_moe": model.use_moe,
        "moe_layers": model.moe_layers,
        "moe_num_experts": model.moe_num_experts,
        "moe_top_k": model.moe_top_k,
        "moe_expert_d_ff": model.moe_expert_d_ff,
        "moe_load_balance_coeff": model.moe_load_balance_coeff,
        "moe_z_loss_coeff": model.moe_z_loss_coeff,
        "moe_initial_temp": model.moe_initial_temp,
        "moe_final_temp": model.moe_final_temp,
        "moe_anneal_steps": model.moe_anneal_steps,
        "moe_train_steps": model.moe_train_steps,
        "use_bayesian_ffn": model.use_bayesian_ffn,
        "bayesian_prior_sigma": model.bayesian_prior_sigma,
        "bayesian_sigma_init": model.bayesian_sigma_init,
        "bayesian_kl_beta_start": model.bayesian_kl_beta_start,
        "bayesian_kl_beta_end": model.bayesian_kl_beta_end,
        "bayesian_kl_anneal_steps": model.bayesian_kl_anneal_steps,
        "bayesian_train_steps": model.bayesian_train_steps,
        "total_epochs": step,
        "total_chars_seen": model.total_chars_seen,
        "smooth_loss": model.smooth_loss,
        "distilled": {
            "teacher": "Qwen2-1.5B",
            "mode": "blank-slate",
            "steps": step,
            "lambda_rep": args.lambda_rep,
            "lr": args.lr,
            "layer_map": [list(p) for p in LAYER_MAP],
        },
    }
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "config": config}, OUT_DIR / "model.pt")
    snap_dir = OUT_DIR / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    snap_state = {k: v.detach().cpu().to(torch.bfloat16) for k, v in model.state_dict().items()}
    torch.save({"state_dict": snap_state, "config": config},
               snap_dir / f"step-{step:06d}.pt")
    tag = "final" if final else f"step-{step}"
    print(f"  saved [{tag}] -> echo_Q_blank/ (step {step})", flush=True)


if __name__ == "__main__":
    main()