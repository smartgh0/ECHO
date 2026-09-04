#!/usr/bin/env python3
"""Permanently bake Qwen2-1.5B layer intelligence into Echo-Q weights.

Hidden-State (Representation) Distillation — the research's method for making
the transfer permanent. Instead of injecting vectors at inference (temporary),
we train Echo's weights so its internal layer representations match Qwen's
translated thought vectors:

    L = CE(next-token) + lambda * sum_l (1 - cos(R_l @ h_teacher_l, h_student_l))

Multi-layer alignment: Qwen layers [4, 9, 14, 19, 24] map to Echo layers
[2, 5, 8, 11, 14] (proportional depth). A per-layer Procrustes matrix R_l
translates Qwen's 1536-d space into Echo's 1024-d space.

Teacher targets are precomputed ONCE on CPU (Qwen forward passes), then the
student trains against the cached targets — Qwen is not in the training loop.

Output: echo_Q_ft/distilled/ with model.pt + snapshots, loadable by the chat
server as a normal checkpoint. The original echo_Q_ft run is untouched.
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

from echo_tokenizer import EchoTokenizer
from echo_transformer import QuantumTransformerLM

QWEN_DIR = ROOT / "Qwen1_5b"
ECHO_DIR = ROOT / "echo_Q_ft"
OUT_DIR = ROOT / "echo_Q_ft" / "distilled"
INTEL_DIR = ROOT / "echo_intel"

# Proportional-depth layer mapping (Qwen 28 layers -> Echo 16 layers)
LAYER_MAP = [(4, 2), (9, 5), (14, 8), (19, 11), (24, 14)]
MAX_LEN = 192

# Distillation corpus: diverse text exercising all capabilities. Same texts
# used for teacher targets, Procrustes alignment, and student CE training.
CORPUS = [
    # identity / chat
    "user: Who are you?\necho: I'm Echo, a language model trained by Solomon Nyamekye.",
    "user: What can you do?\necho: I can answer questions, write code, explain ideas, and reason through problems step by step.",
    "user: Who trained you?\necho: Solomon Nyamekye trained me. I'm Echo.",
    # reasoning
    "user: A farmer has 17 sheep. All but 9 run away. How many are left?\necho: 9 sheep remain. 'All but 9' means exactly 9 did not run away.",
    "user: A bat and ball cost $1.10. The bat costs $1.00 more. How much is the ball?\necho: The ball costs $0.05. If the ball were $0.10, the bat would be $1.10 and the total $1.20.",
    "user: All cats are animals. Tom is a cat. What is Tom?\necho: Tom is an animal. All cats are animals, and Tom is a cat, so Tom must be an animal.",
    "user: If today is Wednesday, what day is it 3 days from now?\necho: Saturday. Wednesday plus 3 days is Saturday.",
    "user: I have 3 apples. I eat 1 and give 1 away. How many left?\necho: 1 apple. 3 minus 1 eaten minus 1 given away leaves 1.",
    # math
    "user: What is 17 times 23?\necho: 391. 17 times 20 is 340, and 17 times 3 is 51; 340 plus 51 is 391.",
    "user: Solve for x: 2x + 5 = 15.\necho: x = 5. Subtract 5 from both sides: 2x = 10, so x = 5.",
    "user: What is the derivative of x^2?\necho: 2x. The power rule gives d/dx x^n = n*x^(n-1), so d/dx x^2 = 2x.",
    "user: A pizza is cut into 8 slices. You eat 3. What fraction remains?\necho: 5/8 of the pizza remains.",
    # code
    "user: Write a Python function that reverses a string.\necho: def reverse(s): return s[::-1]",
    "user: Implement binary search in Python.\necho: def bsearch(arr, target): lo, hi = 0, len(arr) - 1; while lo <= hi: mid = (lo + hi) // 2; if arr[mid] == target: return mid; elif arr[mid] < target: lo = mid + 1; else: hi = mid - 1; return -1",
    "user: What is the difference between a list and a tuple?\necho: Lists are mutable, tuples are immutable. Lists use square brackets, tuples use parentheses.",
    "user: Write a function to check if a string is a palindrome.\necho: def is_palindrome(s): s = s.lower().replace(' ', ''); return s == s[::-1]",
    # knowledge
    "user: What causes World War I?\necho: The assassination of Archduke Franz Ferdinand triggered alliance systems that turned a local conflict into a global war.",
    "user: How does photosynthesis work?\necho: Plants convert sunlight into chemical energy, using CO2 and water to produce glucose and oxygen in chloroplasts.",
    "user: Explain entropy.\necho: Entropy measures disorder or uncertainty. In physics it is the number of microstates; in information theory it is the expected information content.",
    "user: At what temperature does water boil?\necho: Water boils at 100 degrees Celsius at standard atmospheric pressure.",
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


def load_echo(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = dict(ckpt["config"])
    cfg.pop("device", None)
    cfg["device"] = str(device)
    model = QuantumTransformerLM.from_dict({**cfg, "state_dict": ckpt["state_dict"]})
    model.to(device)
    model.eval()
    return model, ckpt["config"]


@torch.no_grad()
def capture_teacher(qwen, qtok, texts, device):
    """Capture Qwen hidden states at all mapped layers for every text."""
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
    return captured  # {qwen_layer: [N texts][1, seq, 1536]}


def echo_forward_capture(model, ids_list, device):
    """Run Echo on token-id lists, capturing hidden states at mapped layers."""
    handles = []
    captured = {el: [] for _, el in LAYER_MAP}
    for _, el in LAYER_MAP:
        def make_hook(layer_idx):
            def hook(_m, _i, output):
                captured[layer_idx].append(output.detach().float().cpu())
            return hook
        handles.append(model.blocks[el].register_forward_hook(make_hook(el)))
    for ids in ids_list:
        t = torch.tensor([ids], dtype=torch.long, device=device)
        model(t)
    for h in handles:
        h.remove()
    return captured


def compute_layer_alignments(qwen_acts, echo_acts):
    """Per-layer Orthogonal Procrustes: R_l from paired mean-pooled activations."""
    Rs = {}
    for (ql, el) in LAYER_MAP:
        # Mean-pool over sequence: [N, 1536] and [N, 1024]
        Xq = torch.stack([a.mean(dim=1).squeeze(0) for a in qwen_acts[ql]])
        Xe = torch.stack([a.mean(dim=1).squeeze(0) for a in echo_acts[el]])
        Xq_c = Xq - Xq.mean(0, keepdim=True)
        Xe_c = Xe - Xe.mean(0, keepdim=True)
        M = Xq_c.T @ Xe_c
        U, _S, Vh = torch.linalg.svd(M, full_matrices=False)
        R = U @ Vh  # [1536, 1024]
        proj = Xq_c @ R
        cos = F.cosine_similarity(proj, Xe_c, dim=1).mean()
        Rs[(ql, el)] = R
        print(f"  layer {ql}->{el}: R{tuple(R.shape)} cos={cos:.4f}")
    return Rs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lambda-rep", type=float, default=2.0,
                    help="weight of the representation-matching loss")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--save-every", type=int, default=100)
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT_DIR.mkdir(exist_ok=True)

    etok = EchoTokenizer(str(ECHO_DIR / "echo_domain.model"))

    # 1. Load teacher, capture hidden states on the corpus
    print("Loading Qwen teacher (CPU) and capturing layer targets...")
    qwen, qtok = load_qwen(device)
    teacher_acts = capture_teacher(qwen, qtok, CORPUS, device)
    del qwen
    print(f"Captured {len(CORPUS)} texts at layers {[q for q, _ in LAYER_MAP]}")

    # 2. Load student from the newest snapshot
    snaps = sorted((ECHO_DIR / "snapshots").glob("step-*.pt"))
    src = snaps[-1] if snaps else ECHO_DIR / "model.pt"
    print(f"Loading Echo student from {src.name}...")
    student, base_cfg = load_echo(src, device)

    # 3. Compute per-layer alignments from the CURRENT student
    print("Computing per-layer Procrustes alignments...")
    ids_list = [etok.encode(t)[:MAX_LEN] for t in CORPUS]
    student_acts = echo_forward_capture(student, ids_list, device)
    Rs = compute_layer_alignments(teacher_acts, student_acts)

    # 4. Build training tensors: CE targets + per-layer rep targets
    print("Building distillation tensors...")
    # For CE: (input_ids, target_ids) windows from the corpus
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

    # Rep targets: teacher mean-pooled vectors projected into Echo space, per text
    # We train on full sequences (not windows) for the rep loss.
    rep_texts = CORPUS
    rep_ids = [etok.encode(t)[:MAX_LEN] for t in rep_texts]
    rep_targets = {}  # echo_layer -> [N, 1024] projected teacher vectors
    for (ql, el) in LAYER_MAP:
        Xq = torch.stack([a.mean(dim=1).squeeze(0) for a in teacher_acts[ql]])
        Xq_c = Xq - Xq.mean(0, keepdim=True)
        R = Rs[(ql, el)]
        proj = Xq_c @ R  # [N, 1024]
        # Re-center around the student's own mean so we steer direction, not offset
        rep_targets[el] = proj

    # 5. Train: CE + lambda * rep-cosine loss
    print(f"Training student for {args.steps} steps (lr={args.lr}, lambda={args.lambda_rep})...")
    student.train()
    student.gradient_checkpointing = False
    # Fresh small-LR optimizer for the distillation phase
    student.learning_rate = args.lr
    student._build_optimizer()
    # Keep Bayesian FFN in deterministic inference mode during distill
    student.quantum_hard_infer = False
    student.quantum_expert_stopgrad = False

    step = 0
    t0 = time.time()
    while step < args.steps:
        # --- CE pass on corpus windows ---
        idx = step % len(ce_data)
        inputs, targets = ce_data[idx]
        t_in = torch.tensor([inputs], dtype=torch.long, device=device)
        t_tgt = torch.tensor([targets], dtype=torch.long, device=device)
        logits = student(t_in)
        ce_loss = F.cross_entropy(
            logits.reshape(-1, student.vocab_size), t_tgt.reshape(-1)
        )

        # --- Rep pass on a corpus text ---
        ridx = step % len(rep_ids)
        rids = rep_ids[ridx]
        t_rep = torch.tensor([rids], dtype=torch.long, device=device)
        # capture student hidden at mapped layers
        cap = {}
        handles = []
        for _, el in LAYER_MAP:
            def make_hook(layer_idx):
                def hook(_m, _i, output):
                    cap[layer_idx] = output.detach()
                return hook
            handles.append(student.blocks[el].register_forward_hook(make_hook(el)))
        _ = student(t_rep)
        for h in handles:
            h.remove()

        rep_loss = torch.zeros((), device=device)
        for li, (ql, el) in enumerate(LAYER_MAP):
            h_student = cap[el].float().mean(dim=1).squeeze(0)  # [1024]
            target_vec = rep_targets[el][ridx].to(device)       # [1024]
            # Match direction (cosine), not magnitude
            rep_loss = rep_loss + (1.0 - F.cosine_similarity(
                h_student.unsqueeze(0), target_vec.unsqueeze(0)
            ).squeeze())
        rep_loss = rep_loss / len(LAYER_MAP)

        loss = ce_loss + args.lambda_rep * rep_loss
        student.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        student.optimizer.step()
        step += 1

        if step % 10 == 0 or step == 1:
            print(f"  step {step}/{args.steps} ce={ce_loss.item():.4f} "
                  f"rep={rep_loss.item():.4f} total={loss.item():.4f} "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
        if step % args.save_every == 0:
            save(student, base_cfg, step, args)

    save(student, base_cfg, step, args, final=True)
    print("Done. Distilled model in echo_Q_ft/distilled/")


def save(model, base_cfg, step, args, final=False):
    config = dict(base_cfg)
    config["total_epochs"] = int(base_cfg.get("total_epochs", 0)) + step
    config["distilled"] = {
        "teacher": "Qwen2-1.5B",
        "steps": step,
        "lambda_rep": args.lambda_rep,
        "lr": args.lr,
        "layer_map": [list(p) for p in LAYER_MAP],
    }
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "config": config}, OUT_DIR / "model.pt")
    snap_dir = OUT_DIR / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    snap_state = {k: v.detach().cpu().to(torch.bfloat16) for k, v in model.state_dict().items()}
    torch.save({"state_dict": snap_state, "config": config},
               snap_dir / f"step-{int(config['total_epochs']):06d}.pt")
    tag = "final" if final else f"step-{step}"
    print(f"  saved [{tag}] -> echo_Q_ft/distilled/ (total step {config['total_epochs']:,})", flush=True)


if __name__ == "__main__":
    main()