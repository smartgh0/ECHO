#!/usr/bin/env python3
"""Suck layer intelligence out of Qwen2-1.5B into Echo-Q's residual stream.

Implements the RepE-style transfer from the research notes:

  1. CONCEPT VECTORS (difference-in-means): run concept vs neutral prompts
     through Qwen, capture mid-layer activations, v = E[x_concept] - E[x_neutral].
  2. ALIGNMENT MATRIX (Orthogonal Procrustes): run the SAME text through both
     models, capture paired activations, M = Xq^T @ Xe, SVD -> R = U @ Vh.
     R translates Qwen's 1536-d space into Echo's 1024-d space.
  3. INJECTION: v_echo = R @ v_qwen, added to Echo's residual stream at the
     matching layer via forward hooks during generation. Echo weights frozen.

Everything runs on CPU (the GPU is owned by the local trainer).
Artifacts land in echo_intel/:
  concepts.npz   {name: (layer, vector_qwen[1536])}
  alignment.npz   R [1536, 1024], qwen_layer, echo_layer
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
QWEN_DIR = ROOT / "Qwen1_5b"
ECHO_CKPT = ROOT / "echo_Q_ft" / "snapshots"
OUT_DIR = ROOT / "echo_intel"

# Mid-layers hold the richest semantic representations (per the research:
# bottom = syntax, mid = semantics/reasoning, top = task-specific output).
QWEN_LAYER = 14          # of 28
ECHO_LAYER = 8           # of 16
MAX_LEN = 128

CONCEPT_PROMPTS = {
    "reasoning": {
        "concept": [
            "Solve step by step: If a train travels 60 mph for 2.5 hours, how far does it go?",
            "All cats are animals. Tom is a cat. What is Tom? Explain the logic.",
            "A bat and ball cost $1.10. The bat costs $1.00 more. How much is the ball? Show reasoning.",
            "If today is Wednesday, what day is it 3 days from now? Explain.",
            "Three people check into a hotel for $30... explain the missing dollar step by step.",
            "You have a 5L jug and a 3L jug. How do you measure exactly 4 liters? Walk through it.",
            "If it rains, the ground gets wet. The ground is NOT wet. Did it rain? Explain the logic.",
            "Sort Apple, Carrot, Banana, Broccoli into groups and explain the reasoning.",
        ],
        "neutral": [
            "The sky is blue today.",
            "I like pizza with extra cheese.",
            "My dog is cute and sleepy.",
            "The chair is in the corner of the room.",
            "She walked to the store this morning.",
            "The window was open all night.",
            "He hummed a tune while cooking.",
            "The leaves fell quietly in autumn.",
        ],
    },
    "code": {
        "concept": [
            "Write a Python function that reverses a string.",
            "Implement a binary search algorithm in Python with tests.",
            "Fix this bug: def fib(n): if n = 0: return 0",
            "Write a REST API endpoint in FastAPI for a TODO app.",
            "Explain the difference between a list and a tuple in Python.",
            "Write a thread-safe producer-consumer queue in Python.",
            "Refactor this loop into a list comprehension.",
            "What is the time complexity of quicksort? Explain.",
        ],
        "neutral": [
            "The ocean waves crashed on the shore.",
            "Breakfast was oatmeal and honey.",
            "The train arrived late again.",
            "A bird sang outside my window.",
            "The coffee shop was crowded at noon.",
            "Rain drummed on the tin roof.",
            "The old library smelled of paper.",
            "Children played in the park nearby.",
        ],
    },
    "math": {
        "concept": [
            "What is the derivative of x^2? Show the steps.",
            "Solve for x: 2x + 5 = 15.",
            "Explain the Pythagorean theorem with an example.",
            "What is 17 times 23? Show your work.",
            "Explain Bayes' theorem and when to use it.",
            "Calculate the compound interest on $10,000 at 5% for 10 years.",
            "What is the integral of 1/x? Explain.",
            "A pizza is cut into 8 slices. You eat 3. What fraction remains?",
        ],
        "neutral": [
            "The garden has many flowers in spring.",
            "He wore a grey coat to work.",
            "The movie last night was long.",
            "Snow covered the mountain tops.",
            "The kettle whistled softly.",
            "They danced until midnight.",
            "The book lay open on the desk.",
            "Traffic was light this evening.",
        ],
    },
    "knowledge": {
        "concept": [
            "Explain photosynthesis at a molecular level.",
            "What caused World War I? Summarize the alliance systems.",
            "Explain the difference between copyright, trademark, and patent.",
            "How does a car engine work? Explain the four strokes.",
            "What is the current status of the James Webb Space Telescope?",
            "Explain entropy in physics and information theory.",
            "Compare the Roman Empire and the Han Dynasty.",
            "Explain the process of vaccination and immunity.",
        ],
        "neutral": [
            "The cat slept on the warm laptop.",
            "He stirred sugar into his tea.",
            "The curtain fluttered in the breeze.",
            "A red ball rolled under the table.",
            "The clock ticked in the quiet room.",
            "She folded the laundry carefully.",
            "The candle flickered and dimmed.",
            "Footsteps echoed in the hallway.",
        ],
    },
}

# Shared text for the Procrustes pairing — same content through both models.
ALIGNMENT_TEXTS = [
    "user: Who are you? echo: I'm Echo, a language model.",
    "The quick brown fox jumps over the lazy dog near the river bank at dawn.",
    "Python is a programming language used for data science, web servers, and automation scripts.",
    "def add(a, b): return a + b",
    "If it rains tomorrow, the picnic will be moved inside the community hall.",
    "The capital of France is Paris, a city known for art, food, and the Eiffel Tower.",
    "2 plus 2 equals 4, and 10 divided by 2 equals 5 in basic arithmetic.",
    "Photosynthesis converts sunlight into chemical energy inside plant chloroplasts.",
    "user: What can you do? echo: I can answer questions, write code, and explain ideas.",
    "A farmer has 17 sheep. All but 9 run away, so 9 sheep remain with the farmer.",
    "The stock market rose three percent after the earnings report was published.",
    "In a distant kingdom, a young knight set out to find the lost crown of the old king.",
    "tool: shell_run command=\"ls -la\" output: file1.txt file2.txt",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure at sea level.",
    "The recipe calls for two cups of flour, one egg, and a pinch of salt mixed well.",
    "History shows that empires rise and fall with trade, war, and technology.",
]


def load_qwen(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(QWEN_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(QWEN_DIR), torch_dtype=torch.float32
    )
    model.to(device)
    model.eval()
    return model, tok


def load_echo(device):
    import sys
    sys.path.insert(0, str(ROOT))
    from echo_tokenizer import EchoTokenizer
    from echo_transformer import QuantumTransformerLM

    snaps = sorted(ECHO_CKPT.glob("step-*.pt"))
    path = snaps[-1] if snaps else ROOT / "echo_Q_ft" / "model.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = dict(ckpt["config"])
    cfg.pop("device", None)
    cfg["device"] = str(device)
    model = QuantumTransformerLM.from_dict({**cfg, "state_dict": ckpt["state_dict"]})
    model.to(device)
    model.eval()
    tok = EchoTokenizer(str(ROOT / "echo_Q_ft" / "echo_domain.model"))
    print(f"Echo loaded: {path.name} (step={model.total_epochs:,})")
    return model, tok


def qwen_layer_hook(model, layer_idx):
    """Capture residual-stream activations at the given Qwen layer."""
    captured = {}
    layer = model.model.layers[layer_idx]

    def hook(_module, _inputs, output):
        # Qwen2 decoder layer returns (hidden, attn_weights, past_kv)
        hidden = output[0] if isinstance(output, tuple) else output
        captured["h"] = hidden.detach()

    handle = layer.register_forward_hook(hook)
    return captured, handle


def echo_layer_hook(model, layer_idx):
    """Capture Echo residual-stream activations after the given block."""
    captured = {}
    block = model.blocks[layer_idx]

    def hook(_module, _inputs, output):
        captured["h"] = output.detach()

    handle = block.register_forward_hook(hook)
    return captured, handle


@torch.no_grad()
def qwen_mean_activation(model, tok, texts, layer_idx, device):
    captured, handle = qwen_layer_hook(model, layer_idx)
    acts = []
    for text in texts:
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] > MAX_LEN:
            ids = ids[:, :MAX_LEN]
        model(ids)
        acts.append(captured["h"].float().mean(dim=1).squeeze(0).cpu())
    handle.remove()
    return torch.stack(acts)  # [N, 1536]


@torch.no_grad()
def echo_mean_activation(model, tok, texts, layer_idx, device):
    captured, handle = echo_layer_hook(model, layer_idx)
    acts = []
    for text in texts:
        ids = tok.encode(text)
        if len(ids) > MAX_LEN:
            ids = ids[:MAX_LEN]
        t = torch.tensor([ids], dtype=torch.long, device=device)
        model(t)
        acts.append(captured["h"].float().mean(dim=1).squeeze(0).cpu())
    handle.remove()
    return torch.stack(acts)  # [N, 1024]


def extract_concepts(qwen, qtok, device):
    """Difference-in-means concept vectors from Qwen's mid layer."""
    concepts = {}
    for name, spec in CONCEPT_PROMPTS.items():
        concept_acts = qwen_mean_activation(qwen, qtok, spec["concept"], QWEN_LAYER, device)
        neutral_acts = qwen_mean_activation(qwen, qtok, spec["neutral"], QWEN_LAYER, device)
        v = concept_acts.mean(0) - neutral_acts.mean(0)
        v = v / v.norm().clamp_min(1e-8)
        concepts[name] = v.numpy()
        print(f"  concept {name}: |v|={float(v.norm()):.4f} dim={v.shape[0]}")
    return concepts


def compute_alignment(qwen, qtok, echo, etok, device):
    """Orthogonal Procrustes: R = U @ Vh of SVD(Xq^T @ Xe)."""
    Xq = qwen_mean_activation(qwen, qtok, ALIGNMENT_TEXTS, QWEN_LAYER, device)
    Xe = echo_mean_activation(echo, etok, ALIGNMENT_TEXTS, ECHO_LAYER, device)
    Xq_c = Xq - Xq.mean(0, keepdim=True)
    Xe_c = Xe - Xe.mean(0, keepdim=True)
    M = Xq_c.T @ Xe_c  # [1536, 1024]
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    R = U @ Vh  # [1536, 1024]
    # Alignment quality: how much variance R explains
    Xq_proj = Xq_c @ R
    cos = F.cosine_similarity(Xq_proj, Xe_c, dim=1)
    print(f"  alignment: R{tuple(R.shape)} mean-cosine={cos.mean():.4f}")
    return R.numpy(), float(cos.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading Qwen2-1.5B (CPU, float32)...")
    qwen, qtok = load_qwen(device)
    print("Loading Echo-Q snapshot...")
    echo, etok = load_echo(device)

    print("Extracting concept vectors (difference-in-means)...")
    concepts = extract_concepts(qwen, qtok, device)
    np.savez(
        OUT_DIR / "concepts.npz",
        qwen_layer=QWEN_LAYER,
        echo_layer=ECHO_LAYER,
        **concepts,
    )

    print("Computing Procrustes alignment matrix...")
    R, quality = compute_alignment(qwen, qtok, echo, etok, device)
    np.savez(
        OUT_DIR / "alignment.npz",
        R=R,
        qwen_layer=QWEN_LAYER,
        echo_layer=ECHO_LAYER,
        quality=quality,
    )

    meta = {
        "qwen": "Qwen1_5b (qwen2, 28 layers, hidden 1536)",
        "echo": f"echo_Q_ft snapshot (16 layers, hidden 1024)",
        "qwen_layer": QWEN_LAYER,
        "echo_layer": ECHO_LAYER,
        "alignment_quality": quality,
        "concepts": list(concepts.keys()),
        "n_alignment_texts": len(ALIGNMENT_TEXTS),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved to {OUT_DIR}/: concepts.npz, alignment.npz, meta.json")
    print(f"Alignment quality (mean cosine): {quality:.4f}")


if __name__ == "__main__":
    main()