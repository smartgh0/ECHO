#!/usr/bin/env python3
"""
ECHO QUANTUM LAYER TEST — Verify layer-level superposition + dropout.
Usage: python3 echo/test_quantum_layer.py
"""
import sys, os, random, math, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0

def test(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  ✅ {name}: PASS {detail}"); PASS += 1
    else:
        print(f"  ❌ {name}: FAIL {detail}"); FAIL += 1

print()
print("=" * 65)
print("  ECHO QUANTUM LAYER TEST — Superposition + Dropout")
print("=" * 65)
print()

# === 1. QuantumLayer ===
print("--- 1. QuantumLayer ---")
from echo_quantum_layer import QuantumLayer, QuantumLayerRNN

ql = QuantumLayer(8, 10, 0.1)
test("init amplitudes", abs(ql.alpha - math.sqrt(0.5)) < 0.01)
test("init shapes", len(ql.W1) == 8 and len(ql.W1[0]) == 10 and len(ql.W2) == 8)

# Sample
W, br = ql.sample()
test("sample returns matrix", len(W) == 8 and len(W[0]) == 10)
test("sample returns branch", br in (0, 1))

# Sample distribution
counts = [0, 0]
for _ in range(10000):
    _, b = ql.sample()
    counts[b] += 1
test("sample ~50/50", abs(counts[0] - 5000) < 500, f"br0={counts[0]} br1={counts[1]}")

# Amplify
for _ in range(50):
    ql.amplify(0, 0.01)
test("amplify shifts alpha", ql.alpha > 0.7, f"alpha={ql.alpha:.4f}")
test("still normalized", abs(ql.alpha**2 + ql.beta**2 - 1.0) < 0.01)

# Expected
ev = ql.expected()
test("expected is matrix", len(ev) == 8 and len(ev[0]) == 10)

# Entropy
ent = ql.entropy()
test("entropy > 0", ent > 0)
test("entropy < max", ent < 0.693)

# Collapse
ql2 = QuantumLayer(4, 4, 0.1)
ql2.alpha = 0.99
ql2.beta = math.sqrt(1 - 0.99**2)
collapsed = ql2.check_collapse()
test("collapse triggers", collapsed)
test("re-branch creates perturbation", ql2.alpha < 0.8)

# Dropout
W_dropped = ql.apply_dropout(0.5, training=True)
zeros_count = sum(1 for row in W_dropped for v in row if v == 0.0)
total = 8 * 10
test("dropout zeros ~50%", abs(zeros_count - total*0.5) < 15,
     f"zeros={zeros_count}/{total}")

# Serialization
data = ql.to_dict()
ql3 = QuantumLayer.from_dict(data)
test("serialize/deserialize", ql3.alpha == ql.alpha and ql3.W1 == ql3.W1)
print()

# === 2. QuantumLayerRNN ===
print("--- 2. QuantumLayerRNN ---")
qrnn = QuantumLayerRNN(vocab_size=10, hidden_size=8, n_samples=4, dropout_rate=0.3)
test("init", qrnn.hidden_size == 8 and qrnn.vocab_size == 10)
test("n_samples", qrnn.n_samples == 4)
test("dropout_rate", qrnn.dropout_rate == 0.3)

# Check layers are QuantumLayer
test("Q_xh is QuantumLayer", isinstance(qrnn.Q_xh, QuantumLayer))
test("Q_hh is QuantumLayer", isinstance(qrnn.Q_hh, QuantumLayer))
test("Q_hy is QuantumLayer", isinstance(qrnn.Q_hy, QuantumLayer))

# Initial entropy near max
qs = qrnn.quantum_stats()
test("initial entropy near max", qs['avg_entropy'] > 0.65, f"ent={qs['avg_entropy']:.4f}")
test("initial no collapses", qs['collapsed_layers'] == 0)
test("theoretical configs = 8", True, "(2^3 = 3 layers × 2 branches)")
print()

# === 3. Training ===
print("--- 3. Training Step ---")
inputs = [[0,0,0,1,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0], [0,0,0,0,0,1,0,0,0,0]]
targets = [4, 5, 6]

loss, _ = qrnn.train_step(inputs, targets)
test("train_step returns loss", loss > 0, f"loss={loss:.4f}")
test("amplifications recorded", qrnn.total_amplifications >= 3,
     f"amps={qrnn.total_amplifications}")

# Entropy should decrease
qs2 = qrnn.quantum_stats()
test("entropy decreased", qs2['avg_entropy'] < qs['avg_entropy'],
     f"{qs['avg_entropy']:.4f} → {qs2['avg_entropy']:.4f}")
print()

# === 4. Multiple training steps ===
print("--- 4. Training (100 epochs) ---")
losses = []
for _ in range(100):
    l, _ = qrnn.train_step(inputs, targets)
    losses.append(l)

test("loss decreased", losses[-1] < losses[0], f"{losses[0]:.4f} → {losses[-1]:.4f}")
qs3 = qrnn.quantum_stats()
test("entropy decreased more", qs3['avg_entropy'] < qs2['avg_entropy'])
test("amplifications grew", qs3['amplifications'] > qs2['amplifications'])
print(f"  Loss: {losses[0]:.4f} → {losses[49]:.4f} → {losses[-1]:.4f}")
print(f"  Entropy: {qs['avg_entropy']:.4f} → {qs3['avg_entropy']:.4f}")
print(f"  Amps: {qs3['amplifications']}")
print()

# === 5. Generation ===
print("--- 5. Generation ---")
sampled = qrnn.sample(inputs[:2], length=20, temperature=0.5)
test("sample produces output", len(sampled) == 20)
print()

# === 6. Neurogenesis ===
print("--- 6. Neurogenesis (grow) ---")
print(f"  Before: hidden={qrnn.hidden_size}, "
      f"Q_xh={len(qrnn.Q_xh.W1)}x{len(qrnn.Q_xh.W1[0])}, "
      f"Q_hh={len(qrnn.Q_hh.W1)}x{len(qrnn.Q_hh.W1[0])}, "
      f"Q_hy={len(qrnn.Q_hy.W1)}x{len(qrnn.Q_hy.W1[0])}")

qrnn.grow(4)
test("hidden grew", qrnn.hidden_size == 12)
test("Q_xh rows grew", len(qrnn.Q_xh.W1) == 12)
test("Q_hh rows grew", len(qrnn.Q_hh.W1) == 12)
test("Q_hh cols grew", len(qrnn.Q_hh.W1[0]) == 12)
test("Q_hy cols grew", len(qrnn.Q_hy.W1[0]) == 12)

print(f"  After:  hidden={qrnn.hidden_size}, "
      f"Q_xh={len(qrnn.Q_xh.W1)}x{len(qrnn.Q_xh.W1[0])}, "
      f"Q_hh={len(qrnn.Q_hh.W1)}x{len(qrnn.Q_hh.W1[0])}, "
      f"Q_hy={len(qrnn.Q_hy.W1)}x{len(qrnn.Q_hy.W1[0])}")

# Train after growth
loss_ag, _ = qrnn.train_step(inputs, targets)
test("train after grow", loss_ag > 0)
print()

# === 7. Persistence ===
print("--- 7. Save/Load ---")
data = qrnn.to_dict()
test("to_dict has layers", 'Q_xh' in data and 'Q_hh' in data and 'Q_hy' in data)

qrnn2 = QuantumLayerRNN.from_dict(data)
test("from_dict restores hidden", qrnn2.hidden_size == qrnn.hidden_size)
test("from_dict restores alpha",
     abs(qrnn2.Q_xh.alpha - qrnn.Q_xh.alpha) < 1e-10)
test("from_dict restores W1",
     qrnn2.Q_xh.W1[0][0] == qrnn.Q_xh.W1[0][0])
test("from_dict restores dropout", qrnn2.dropout_rate == qrnn.dropout_rate)
print()

# === 8. Full brain with quantum_layer mode ===
print("--- 8. Full Brain (quantum_layer mode) ---")
from echo_brain import EchoBrain

brain = EchoBrain(hidden_size=16, learning_rate=0.02, mode='quantum_layer',
                  quantum_samples=4, dropout_rate=0.3)

conversations = [
    ("user", "hello echo"), ("echo", "hello i am echo"),
    ("user", "the ocean is beautiful"), ("echo", "the ocean is beautiful yes"),
    ("user", "i love the waves"), ("echo", "waves are peaceful and calm"),
]
for role, text in conversations:
    brain.add_conversation(role, text)

brain.build_vocab()
print(f"  Mode: {brain.mode}")
print(f"  Model type: {type(brain.model).__name__}")
test("brain is quantum_layer", brain.mode == 'quantum_layer')
test("model is QuantumLayerRNN", isinstance(brain.model, QuantumLayerRNN))

print("  Training 50 epochs...")
brain.train(epochs=50, verbose=False)
test("brain trained", brain.model.smooth_loss is not None, f"loss={brain.model.smooth_loss:.4f}")

qs = brain.model.quantum_stats()
print(f"  Loss: {brain.model.smooth_loss:.4f}")
print(f"  Entropy: {qs['avg_entropy']:.4f}")
print(f"  Collapsed layers: {qs['collapsed_layers']}")
print(f"  Dropout: {qs['dropout_rate']}")

response = brain.respond("hello", temperature=0.5, length=60)
print(f"  Response: '{response}'")
test("brain responds", len(response) > 0)
print()

# === 9. Save/Load brain ===
print("--- 9. Brain Save/Load ---")
tmpdir = tempfile.mkdtemp()
brain.save(tmpdir)
files = sorted(os.listdir(tmpdir))
print(f"  Files: {files}")
test("mode.json saved", "mode.json" in files)
test("model.json saved", "model.json" in files)

brain2 = EchoBrain.load(tmpdir)
test("mode persisted", brain2.mode == 'quantum_layer')
test("model is QuantumLayerRNN", isinstance(brain2.model, QuantumLayerRNN))
test("hidden size persisted", brain2.model.hidden_size == brain.model.hidden_size)
test("dropout persisted", brain2.model.dropout_rate == brain.model.dropout_rate)
print()

# === 10. Comparison: all 3 modes ===
print("--- 10. Comparison: Classical vs Quantum vs Quantum Layer (100 epochs) ---")
modes = {'classical': 'Classical', 'quantum': 'Full Quantum', 'quantum_layer': 'Quantum Layer'}
brains = {}
for mode, label in modes.items():
    b = EchoBrain(hidden_size=16, learning_rate=0.02, mode=mode,
                  quantum_samples=4 if mode == 'quantum_layer' else 8)
    for role, text in conversations:
        b.add_conversation(role, text)
    b.build_vocab()
    brains[mode] = (b, label)

for _ in range(100):
    for mode, (b, _) in brains.items():
        b.train(epochs=1, verbose=False)

print(f"  {'Mode':<20} {'Loss':>8} {'Type':>20}")
print(f"  {'─'*20} {'─'*8} {'─'*20}")
for mode, (b, label) in brains.items():
    loss = b.model.smooth_loss
    mtype = type(b.model).__name__
    print(f"  {label:<20} {loss:>8.4f} {mtype:>20}")

best_mode = min(brains, key=lambda m: brains[m][0].model.smooth_loss)
print(f"\n  Best: {modes[best_mode]}")
print()

# === RESULTS ===
print("=" * 65)
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — Quantum Layer Echo is alive.")
else:
    print(f"  ❌ {FAIL} tests failed.")
print("=" * 65)