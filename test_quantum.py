#!/usr/bin/env python3
"""
ECHO QUANTUM TEST — Verify quantum superposition neuron works.
Usage: python3 echo/test_quantum.py
"""
import sys, os, random, math, time, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✅ {name}: PASS {detail}")
        PASS += 1
    else:
        print(f"  ❌ {name}: FAIL {detail}")
        FAIL += 1

print()
print("=" * 60)
print("  ECHO QUANTUM TEST — Superposition Neurons")
print("=" * 60)
print()

# === 1. QuantumWeight ===
print("--- 1. QuantumWeight ---")
from echo_quantum import QuantumWeight

qw = QuantumWeight(0.5, -0.3)
test("init amplitudes", abs(qw.alpha - math.sqrt(0.5)) < 0.01 and abs(qw.beta - math.sqrt(0.5)) < 0.01)
test("init values", qw.w1 == 0.5 and qw.w2 == -0.3)

# Sample many times, check distribution
counts = [0, 0]
for _ in range(10000):
    val, br = qw.sample()
    counts[br] += 1
test("sample distribution ~50/50", abs(counts[0] - 5000) < 500, f"br0={counts[0]} br1={counts[1]}")

# Amplify branch 0
for _ in range(100):
    qw.amplify(0, 0.01)
test("amplify shifts alpha up", qw.alpha > 0.8, f"alpha={qw.alpha:.4f}")
test("amplify shifts beta down", qw.beta < 0.6, f"beta={qw.beta:.4f}")

# Normalization
test("still normalized", abs(qw.alpha**2 + qw.beta**2 - 1.0) < 0.01)

# Expected value
ev = qw.expected_value()
expected = qw.alpha**2 * qw.w1 + qw.beta**2 * qw.w2
test("expected_value", abs(ev - expected) < 1e-10, f"ev={ev:.4f}")

# Entropy
ent = qw.entropy()
test("entropy > 0", ent > 0, f"entropy={ent:.4f}")
test("entropy < max", ent < 0.693, f"entropy={ent:.4f} (amplified, should be < 0.693)")

# Collapse check
qw2 = QuantumWeight(0.5, -0.3)
qw2.alpha = 0.99
qw2.beta = math.sqrt(1 - 0.99**2)
collapsed = qw2.check_collapse()
test("collapse triggers", collapsed == True)
test("collapse re-branches", qw2.alpha < 0.8, f"alpha after re-branch={qw2.alpha:.4f}")

# Serialization
qw3 = QuantumWeight(0.7, 1.5, math.sqrt(0.3), -2.0)
data = qw3.to_list()
qw4 = QuantumWeight.from_list(data)
test("serialize/deserialize", qw4.alpha == qw3.alpha and qw4.w1 == qw3.w1 and qw4.w2 == qw3.w2)
print()

# === 2. QuantumRNN ===
print("--- 2. QuantumRNN ---")
from echo_quantum import QuantumRNN

qrnn = QuantumRNN(vocab_size=10, hidden_size=8, n_samples=4)
test("init", qrnn.hidden_size == 8 and qrnn.vocab_size == 10)
test("n_samples", qrnn.n_samples == 4)

# Check all weights are QuantumWeights
test("Q_xh is quantum", isinstance(qrnn.Q_xh[0][0], QuantumWeight))
test("Q_hh is quantum", isinstance(qrnn.Q_hh[0][0], QuantumWeight))
test("Q_hy is quantum", isinstance(qrnn.Q_hy[0][0], QuantumWeight))

# Initial entropy should be near maximum (0.693)
qs = qrnn.quantum_stats()
test("initial entropy near max", qs['avg_entropy'] > 0.65, f"entropy={qs['avg_entropy']:.4f}")
test("initial no collapses", qs['collapsed'] == 0)
print()

# === 3. Training step ===
print("--- 3. Training Step ---")
inputs = [[0,0,0,1,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0], [0,0,0,0,0,1,0,0,0,0]]
targets = [4, 5, 6]

loss, _ = qrnn.train_step(inputs, targets)
test("train_step returns loss", loss > 0, f"loss={loss:.4f}")
test("amplifications recorded", qrnn.total_amplifications > 0, f"amps={qrnn.total_amplifications}")

# After training, entropy should decrease slightly
qs2 = qrnn.quantum_stats()
test("entropy decreased", qs2['avg_entropy'] < qs['avg_entropy'],
     f"{qs['avg_entropy']:.4f} → {qs2['avg_entropy']:.4f}")
print()

# === 4. Multiple training steps ===
print("--- 4. Multiple Training Steps (50 epochs) ---")
losses = []
for i in range(50):
    loss, _ = qrnn.train_step(inputs, targets)
    losses.append(loss)

initial_loss = losses[0]
final_loss = losses[-1]
test("loss decreased over 50 epochs", final_loss < initial_loss,
     f"{initial_loss:.4f} → {final_loss:.4f}")

qs3 = qrnn.quantum_stats()
test("entropy decreased further", qs3['avg_entropy'] < qs2['avg_entropy'],
     f"{qs2['avg_entropy']:.4f} → {qs3['avg_entropy']:.4f}")
test("amplifications grew", qs3['amplifications'] > qs2['amplifications'])
print(f"  Loss curve: {initial_loss:.4f} → {losses[24]:.4f} → {final_loss:.4f}")
print(f"  Entropy:    {qs['avg_entropy']:.4f} → {qs3['avg_entropy']:.4f}")
print(f"  Amps:       {qs3['amplifications']:,}")
print()

# === 5. Generation ===
print("--- 5. Generation ---")
sampled = qrnn.sample(inputs[:2], length=20, temperature=0.5)
test("sample produces output", len(sampled) == 20)
print(f"  Generated: {sampled}")
print()

# === 6. Neurogenesis (grow) ===
print("--- 6. Neurogenesis (grow) ---")
print(f"  Before: hidden={qrnn.hidden_size}, "
      f"Q_xh={len(qrnn.Q_xh)}x{len(qrnn.Q_xh[0])}, "
      f"Q_hh={len(qrnn.Q_hh)}x{len(qrnn.Q_hh[0])}, "
      f"Q_hy={len(qrnn.Q_hy)}x{len(qrnn.Q_hy[0])}")

qrnn.grow(4)
test("hidden grew", qrnn.hidden_size == 12)
test("Q_xh rows grew", len(qrnn.Q_xh) == 12)
test("Q_hh rows grew", len(qrnn.Q_hh) == 12)
test("Q_hh cols grew", len(qrnn.Q_hh[0]) == 12)
test("Q_hy cols grew", len(qrnn.Q_hy[0]) == 12)
test("new weights are quantum", isinstance(qrnn.Q_xh[11][0], QuantumWeight))

print(f"  After:  hidden={qrnn.hidden_size}, "
      f"Q_xh={len(qrnn.Q_xh)}x{len(qrnn.Q_xh[0])}, "
      f"Q_hh={len(qrnn.Q_hh)}x{len(qrnn.Q_hh[0])}, "
      f"Q_hy={len(qrnn.Q_hy)}x{len(qrnn.Q_hy[0])}")

# Train after growth
loss_after_grow, _ = qrnn.train_step(inputs, targets)
test("train after grow", loss_after_grow > 0)
print()

# === 7. Persistence ===
print("--- 7. Save/Load ---")
model_data = qrnn.to_dict()
test("to_dict works", 'Q_xh' in model_data and 'Q_hh' in model_data)

qrnn2 = QuantumRNN.from_dict(model_data)
test("from_dict restores hidden", qrnn2.hidden_size == qrnn.hidden_size)
test("from_dict restores amplitudes",
     abs(qrnn2.Q_xh[0][0].alpha - qrnn.Q_xh[0][0].alpha) < 1e-10)
test("from_dict restores values",
     qrnn2.Q_xh[0][0].w1 == qrnn.Q_xh[0][0].w1)
test("from_dict restores stats", qrnn2.total_amplifications == qrnn.total_amplifications)
print()

# === 8. Full brain with quantum mode ===
print("--- 8. Full Brain (Quantum Mode) ---")
from echo_brain import EchoBrain

brain = EchoBrain(hidden_size=16, learning_rate=0.02, mode='quantum', quantum_samples=4)

conversations = [
    ("user", "hello echo"),
    ("echo", "hello i am echo"),
    ("user", "the ocean is beautiful"),
    ("echo", "the ocean is beautiful yes"),
    ("user", "i love the waves"),
    ("echo", "waves are peaceful and calm"),
]
for role, text in conversations:
    brain.add_conversation(role, text)

brain.build_vocab()
print(f"  Mode: {brain.mode}")
print(f"  Vocab: {brain.vocab.size}")
print(f"  Model type: {type(brain.model).__name__}")
test("brain is quantum", brain.mode == 'quantum')
test("model is QuantumRNN", isinstance(brain.model, QuantumRNN))

# Train
print("  Training 50 epochs...")
brain.train(epochs=50, verbose=False)
test("brain trained", brain.model.smooth_loss is not None, f"loss={brain.model.smooth_loss:.4f}")

qs_final = brain.model.quantum_stats()
print(f"  Loss: {brain.model.smooth_loss:.4f}")
print(f"  Entropy: {qs_final['avg_entropy']:.4f}")
print(f"  Collapsed: {qs_final['collapsed']}")
print(f"  Amplifications: {qs_final['amplifications']:,}")

# Generate
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
test("mode persisted", brain2.mode == 'quantum')
test("model is QuantumRNN after load", isinstance(brain2.model, QuantumRNN))
test("hidden size persisted", brain2.model.hidden_size == brain.model.hidden_size)
test("amplitudes persisted",
     abs(brain2.model.Q_xh[0][0].alpha - brain.model.Q_xh[0][0].alpha) < 1e-10)
print(f"  Loaded: mode={brain2.mode}, neurons={brain2.model.hidden_size}")
print()

# === 10. Quantum vs Classical comparison ===
print("--- 10. Quantum vs Classical (100 epochs) ---")
from echo_rnn import EchoRNN

brain_c = EchoBrain(hidden_size=16, learning_rate=0.02, mode='classical')
for role, text in conversations:
    brain_c.add_conversation(role, text)
brain_c.build_vocab()

brain_q = EchoBrain(hidden_size=16, learning_rate=0.02, mode='quantum', quantum_samples=4)
for role, text in conversations:
    brain_q.add_conversation(role, text)
brain_q.build_vocab()

print(f"  Training both 100 epochs...")
for _ in range(100):
    brain_c.train(epochs=1, verbose=False)
    brain_q.train(epochs=1, verbose=False)

c_loss = brain_c.model.smooth_loss
q_loss = brain_q.model.smooth_loss
print(f"  Classical loss: {c_loss:.4f}")
print(f"  Quantum loss:   {q_loss:.4f}")
print(f"  Difference:     {c_loss - q_loss:+.4f}")

winner = "QUANTUM" if q_loss < c_loss else "CLASSICAL"
print(f"  Winner: {winner}")
test("quantum trains", q_loss < 4.0, f"q_loss={q_loss:.4f}")
test("classical trains", c_loss < 4.0, f"c_loss={c_loss:.4f}")
print()

# === RESULTS ===
print("=" * 60)
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — Quantum Echo is alive.")
else:
    print(f"  ❌ {FAIL} tests failed.")
print("=" * 60)
print()