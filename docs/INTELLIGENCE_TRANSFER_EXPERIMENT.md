# Intelligence Transfer Experiment: Sucking Qwen2-1.5B's Layer Intelligence into Echo-Q

**Date:** 2026-09-04
**Student model:** Echo-Q (302M params, 16 layers, d_model=1024, Bayesian FFN, custom SentencePiece tokenizer, vocab 16,384)
**Teacher model:** Qwen2-1.5B (28 layers, hidden 1536, vocab 151,936)
**Hardware:** Single 8GB GPU (RTX-class), CPU fallback

---

## 1. Motivation

The research question: can the "intelligence" embedded in a trained model's layers be
extracted and transferred into a different architecture — not via traditional output
distillation, but by directly manipulating internal layer representations?

Three techniques from the representation-engineering literature were implemented and tested:

1. **Concept-vector extraction** (difference-in-means)
2. **Cross-architecture alignment** (Orthogonal Procrustes)
3. **Hidden-state distillation** (representation matching, permanent)

The experiment ran in three phases, each producing a measurable verdict.

---

## 2. Background: The Math

### 2.1 The residual stream as a knowledge bus

A transformer layer writes its contribution additively into a shared residual stream:

$$x_{i+1} = x_i + \text{Layer}_i(x_i)$$

The "intelligence" a layer contributes is isolable: $\Delta x_k = x_{k+1} - x_k$.
Concepts are stored as **linear directions** in this space (the Linear Representation
Hypothesis), which makes them extractable as vectors.

### 2.2 Concept vectors (difference-in-means)

Run concept-activating and neutral prompts through the teacher, capture mid-layer
activations, and take the mean difference:

$$v_{\text{concept}} = \mathbb{E}[x_{\text{concept}}] - \mathbb{E}[x_{\text{neutral}}]$$

### 2.3 Cross-architecture alignment (Orthogonal Procrustes)

Two models represent the same concepts in different coordinate systems. Given paired
activations $X_q$ (teacher, 1536-d) and $X_e$ (student, 1024-d) on the same texts:

$$M = X_q^T X_e, \quad M = U \Sigma V^T, \quad R = U V^T$$

$R$ is the optimal rotation translating the teacher's space into the student's.

### 2.4 Hidden-state distillation (the permanent method)

Train the student so its internal representations match the teacher's translated vectors:

$$L = \text{CE}(\text{next-token}) + \lambda \sum_{\ell} \left(1 - \cos(R_\ell h^{\text{teacher}}_\ell,\ h^{\text{student}}_\ell)\right)$$

Unlike inference-time injection, this **permanently bakes** the teacher's representational
directions into the student's weights.

---

## 3. Phase 1 — Runtime Injection (Temporary Steering)

**Script:** `echo_intel_extract.py`, `echo_intel_inject.py`
**Student:** Echo-Q at training step 230,000 (230k steps of pretraining)

### Procedure

1. Extracted 4 concept vectors (reasoning, code, math, knowledge) from Qwen layer 14
   using 8 concept + 8 neutral prompts each.
2. Computed the Procrustes alignment matrix $R$ from 16 paired texts
   (Qwen layer 14 ↔ Echo layer 8).
3. Injected translated vectors into Echo's residual stream at layer 8 via forward hooks
   during generation. Echo's weights untouched.

### Results

| Metric | Value |
|---|---|
| Alignment quality (mean cosine) | **0.806** |
| A/B generation shift | Output measurably changes with injection on/off |
| Scale response (α = 1.0 vs 2.0) | Stronger injection → stronger steering |

Example: with `reasoning=2.0` injected, Echo pulled Pythagorean-theorem vocabulary
into an unrelated sheep-counting problem — the reasoning direction genuinely steered
the output distribution.

**Verdict:** The mechanism works. Injection is a real-time steering knob, reversible,
zero training cost. Exposed as `:inject reasoning=1.5` in the chat server.

---

## 4. Phase 2 — Permanent Distillation into a Pretrained Student

**Script:** `echo_intel_distill.py`
**Student:** Echo-Q step 230,000 (same pretrained checkpoint)

### Procedure

1. Captured Qwen hidden states at 5 layers [4, 9, 14, 19, 24] on a 28-text corpus
   (identity, reasoning, math, code, knowledge, tools, prose).
2. Mapped them proportionally to Echo layers [2, 5, 8, 11, 14] with per-layer Procrustes
   matrices.
3. Trained 300 steps (lr 3e-5, λ=2.0) with CE + rep-cosine loss. Teacher targets
   precomputed once; Qwen not in the training loop.

### Per-layer alignment quality (before training)

| Qwen → Echo | cosine |
|---|---|
| 4 → 2 | 0.652 |
| 9 → 5 | 0.725 |
| 14 → 8 | 0.755 |
| 19 → 11 | 0.856 |
| 24 → 14 | **0.926** |

### Results

**Validation loss (no regression):** 3.8976 → **3.8966**

**Generation A/B (original → distilled):**

| Prompt | Original | Distilled |
|---|---|---|
| "Who are you?" | `echo echo echo 1991, 20191, 20199...` | **"I'm Echo — built and a language model"** |
| "Reverse a string" | Correct code, no explanation | Correct code **+ correct explanation** |
| Math | Numeric gibberish | Actual arithmetic vocabulary (still wrong numerically) |

**Verdict:** Real, bounded improvement. The pretrained substrate gained Qwen-aligned
directions without losing anything (validation loss flat). Registered as
`echo-q-distilled` in the chat server.

---

## 5. Phase 3 — The Blank-Slate Experiment (The Real Test)

**Script:** `echo_intel_blank.py`
**Student:** Echo-Q built from **random initialization — zero pretraining**

This is the decisive test of the "sucking intelligence" hypothesis. If layer
intelligence transfer is a knowledge transplant, a fresh model should acquire
Qwen's capabilities from the distillation alone. If it's only a steering mechanism,
the fresh model should align internally but fail to generalize.

### Procedure

Identical to Phase 2, but the student starts from random weights. 2000 steps
(lr 1e-4, λ=4.0 — higher, since a fresh model can't match a trained teacher's CE).

### Training metrics

- CE loss: **9.94 → ~0.1** (memorized the 38-text corpus)
- Rep loss: **0.31 → ~0.01**
- Post-training layer alignment: **cos = 0.995 at every mapped layer**

### Generation results

| Test | Output | Verdict |
|---|---|---|
| **In-corpus** "Who are you?" | `"I'm Echo, a language model trained by Solomon Nyamekye..."` | ✅ Perfect |
| **In-corpus** sheep problem | `"9 sheep remain. 'All but 9' means exactly 9 did not run away."` | ✅ Correct answer + reasoning |
| **Out-of-corpus** capital of Germany | `"Paris..... is an animal. tool: shell plus..."` | ❌ Gibberish |
| **Out-of-corpus** hello world | `"def reverse(s): return s[::-1..."` | ❌ Code syntax, wrong content |

### Objective eval

| Model | Validation loss |
|---|---|
| Original Echo-Q (230k steps pretraining) | 3.8976 |
| Random init (theoretical: ln(16384)) | 9.70 |
| **Blank-slate distilled** | **11.02** |

The blank-slate model scores *worse than random init* on held-out text — it
overfit the tiny corpus while its internal representations match Qwen's almost
perfectly.

---

## 6. Conclusions

### What was proven

1. **Representation transfer works mechanically.** Post-training cosine alignment of
   0.995 across all five mapped layers is near-perfect. The Procrustes math genuinely
   moves a student's internal vectors into the teacher's configuration.

2. **But representations alone are not intelligence.** A model with Qwen-aligned
   internals and no language pretraining cannot generalize. Out-of-corpus generation
   collapses into memorized corpus fragments.

3. **Layer intelligence transfer is a steering mechanism, not a knowledge transplant.**
   It can redirect an existing language model's behavior (Phase 2's identity fix,
   Phase 1's reasoning steering) but cannot bootstrap one from random weights.

### The core insight

The research's "snapshot beamed into the student's brain" analogy is accurate in an
unexpected way: the snapshot contains the *direction* of thought, not the *content*.
A student that already speaks the language (pretrained Echo-Q) can be steered along
those directions productively. A student that doesn't (blank slate) aligns its
internals perfectly and still produces gibberish — because the residual stream's
coordinate alignment is necessary but not sufficient. The knowledge lives in the
*interaction* of aligned directions with a pretrained decoding stack.

### Practical guidance

- **To improve an existing model:** hidden-state distillation on a pretrained
  substrate (Phase 2) — real gains, no regression risk at low LR.
- **To steer at runtime:** concept-vector injection (Phase 1) — instant, reversible,
  per-request control.
- **To create capability from nothing:** doesn't work. Pretraining on real data
  remains irreplaceable; distillation directs existing capability, it doesn't create it.

---

## 7. Artifacts

| File | Purpose |
|---|---|
| `echo_intel_extract.py` | Phase 1: concept vectors + Procrustes alignment |
| `echo_intel_inject.py` | Phase 1: runtime hook-based injection |
| `echo_intel_distill.py` | Phase 2: permanent distillation into pretrained Echo |
| `echo_intel_blank.py` | Phase 3: blank-slate experiment |
| `echo_intel/concepts.npz` | Extracted concept vectors (Qwen space) |
| `echo_intel/alignment.npz` | Procrustes matrix R (1536→1024) |
| `echo_Q_ft/distilled/` | Phase 2 output (registered as `echo-q-distilled`) |
| `echo_Q_blank/` | Phase 3 output (research artifact only) |

Chat-server commands exposed by this work:

```
:inject                      list concepts + status
:inject reasoning=1.5        activate steering at scale 1.5
:inject code=0.5 math=0.5    combine concepts
:inject off                  disable
```

---

## 8. Reproduction

```bash
# Phase 1 — extraction + runtime injection
python3 echo_intel_extract.py --device cuda
# then in Echo Chat: :inject reasoning=1.5

# Phase 2 — permanent distillation (pretrained student)
python3 echo_intel_distill.py --device cuda --steps 300

# Phase 3 — blank-slate experiment
python3 echo_intel_blank.py --device cuda --steps 2000
```

**Hardware note:** all three phases fit on a single 8GB GPU. The 302M-param student
in fp32 with AdamW consumes 4.84GB (weights 1.21 + optimizer states 2.42 + grads
1.21), leaving ~2.8GB for activations. The blank-slate run must stay in `eval()`
mode: `train()` mode + gradient checkpointing re-runs the forward during backward
and the Bayesian FFN's stochastic sampling triples activation memory, OOMing an 8GB card.