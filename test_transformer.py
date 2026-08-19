#!/usr/bin/env python3
"""Regression tests for Echo's quantum transformer mode."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from echo_brain import EchoBrain
from echo_transformer import QuantumTransformerLM

PASS = 0
FAIL = 0


def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  PASS {name} {detail}")
        PASS += 1
    else:
        print(f"  FAIL {name} {detail}")
        FAIL += 1


print("ECHO QUANTUM TRANSFORMER TEST")

brain = EchoBrain(mode="quantum_transformer")
brain.training_corpus = "hello echo " * 30
brain.build_vocab()
test("model type", isinstance(brain.model, QuantumTransformerLM))
test("model dimensions", brain.model.d_model == 128 and brain.model.n_layers == 4)

epoch_losses = []
for _ in range(25):
    loss, _ = brain.model.train_step([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
    epoch_losses.append(loss)
test("loss decreases", epoch_losses[-1] < epoch_losses[0],
     f"{epoch_losses[0]:.4f} -> {epoch_losses[-1]:.4f}")

test("sample length", len(brain.model.sample(brain.vocab.encode_one_hot("hello"), length=12)) == 12)
qs = brain.model.quantum_stats()
test("quantum statistics", 0.0 < qs["avg_entropy"] <= qs["max_entropy"])

with tempfile.TemporaryDirectory() as directory:
    brain.save(directory)
    restored = EchoBrain.load(directory)
    test("save/load mode", restored.mode == "quantum_transformer")
    test("save/load model", isinstance(restored.model, QuantumTransformerLM))
    test("save/load loss", restored.model.smooth_loss == brain.model.smooth_loss)

print(f"RESULTS: {PASS} passed, {FAIL} failed")
if FAIL:
    raise SystemExit(1)
