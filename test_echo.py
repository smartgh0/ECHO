#!/usr/bin/env python3
"""
ECHO LIVE TEST — Run this to verify everything works.
Usage: python3 echo/test_echo.py
"""
import sys, os, tempfile
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
print("=" * 55)
print("  ECHO LIVE TEST — Pure Python, Zero Dependencies")
print("=" * 55)
print()

# === 1. Matrix Math ===
print("--- 1. Matrix Math ---")
from echo_matrix import *

a = [[1, 2], [3, 4]]
b = [[5, 6], [7, 8]]
c = matmul(a, b)
test("matmul", c == [[19, 22], [43, 50]], f"= {c}")

t = transpose(a)
test("transpose", t == [[1, 3], [2, 4]])

test("tanh(0)", abs(tanh(0)) < 1e-10)

s = softmax([1.0, 2.0, 3.0])
test("softmax sums to 1", abs(sum(s) - 1.0) < 1e-10)

test("argmax", argmax([0.1, 0.9, 0.2]) == 1)

r = random_matrix(3, 4, 0.1)
test("random_matrix shape", len(r) == 3 and len(r[0]) == 4)
print()

# === 2. RNN Forward/Backward ===
print("--- 2. RNN Forward/Backward ---")
from echo_rnn import EchoRNN

rnn = EchoRNN(vocab_size=10, hidden_size=8, learning_rate=0.01)
test("RNN init", rnn.hidden_size == 8 and rnn.vocab_size == 10)

inputs = [[0,0,0,1,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0], [0,0,0,0,0,1,0,0,0,0]]
targets = [4, 5, 6]
logits_list, hidden_states, h_final = rnn.forward(inputs)
test("forward output count", len(logits_list) == 3)
test("hidden state count", len(hidden_states) == 4)  # initial + 3 steps
test("logits dimension", len(logits_list[0]) == 10)

loss, h = rnn.train_step(inputs, targets)
test("train_step returns loss", loss > 0, f"loss={loss:.4f}")

sampled = rnn.sample(inputs[:2], length=10, temperature=0.8)
test("sample length", len(sampled) == 10)
print()

# === 3. Neurogenesis (grow) ===
print("--- 3. Neurogenesis (grow) ---")
rnn2 = EchoRNN(vocab_size=10, hidden_size=8)
print(f"  Before: hidden={rnn2.hidden_size}, "
      f"W_xh={len(rnn2.W_xh)}x{len(rnn2.W_xh[0])}, "
      f"W_hh={len(rnn2.W_hh)}x{len(rnn2.W_hh[0])}, "
      f"W_hy={len(rnn2.W_hy)}x{len(rnn2.W_hy[0])}")

rnn2.grow(4)
test("hidden grew", rnn2.hidden_size == 12, f"→ {rnn2.hidden_size}")
test("W_xh rows grew", len(rnn2.W_xh) == 12)
test("W_hh rows grew", len(rnn2.W_hh) == 12)
test("W_hh cols grew", len(rnn2.W_hh[0]) == 12)
test("W_hy cols grew", len(rnn2.W_hy[0]) == 12)
test("b_h grew", len(rnn2.b_h) == 12)

print(f"  After:  hidden={rnn2.hidden_size}, "
      f"W_xh={len(rnn2.W_xh)}x{len(rnn2.W_xh[0])}, "
      f"W_hh={len(rnn2.W_hh)}x{len(rnn2.W_hh[0])}, "
      f"W_hy={len(rnn2.W_hy)}x{len(rnn2.W_hy[0])}")

# Verify forward works after growth
logits, _, _ = rnn2.forward(inputs)
test("forward after grow", len(logits) == 3)
print()

# === 4. Synaptic Pruning ===
print("--- 4. Synaptic Pruning ---")
rnn3 = EchoRNN(vocab_size=10, hidden_size=12)
print(f"  Before: hidden={rnn3.hidden_size}, "
      f"W_xh={len(rnn3.W_xh)}x{len(rnn3.W_xh[0])}, "
      f"W_hh={len(rnn3.W_hh)}x{len(rnn3.W_hh[0])}, "
      f"W_hy={len(rnn3.W_hy)}x{len(rnn3.W_hy[0])}")

rnn3.prune(2)
test("hidden shrank", rnn3.hidden_size == 11, f"→ {rnn3.hidden_size}")
test("W_xh rows pruned", len(rnn3.W_xh) == 11)
test("W_hh rows pruned", len(rnn3.W_hh) == 11)
test("W_hh cols pruned", len(rnn3.W_hh[0]) == 11)
test("W_hy cols pruned", len(rnn3.W_hy[0]) == 11)
test("b_h pruned", len(rnn3.b_h) == 11)

print(f"  After:  hidden={rnn3.hidden_size}, "
      f"W_xh={len(rnn3.W_xh)}x{len(rnn3.W_xh[0])}, "
      f"W_hh={len(rnn3.W_hh)}x{len(rnn3.W_hh[0])}, "
      f"W_hy={len(rnn3.W_hy)}x{len(rnn3.W_hy[0])}")

logits, _, _ = rnn3.forward(inputs)
test("forward after prune", len(logits) == 3)
print()

# === 5. Evolution Tracker ===
print("--- 5. Evolution Tracker ---")
from echo_evolve import EvolutionTracker

evo = EvolutionTracker(initial_hidden=16, min_hidden=8, max_hidden=64, initial_lr=0.01)
test("evo init", evo.hidden_size == 16 and evo.lr == 0.01)

# Simulate plateau
for i in range(200):
    evo.record_loss(3.0)
test("neurogenesis on plateau", evo.total_grows > 0, 
     f"grows={evo.total_grows}")
test("mutations logged", len(evo.mutations) > 0,
     f"{len(evo.mutations)} mutations")

print(f"  Mutation log:")
for m in evo.recent_mutations(5):
    print(f"    {m['type']}: {m['detail']}")
print()

# === 6. Full Brain with Evolution ===
print("--- 6. Full Brain with Evolution ---")
from echo_brain import EchoBrain

brain = EchoBrain(hidden_size=16, learning_rate=0.02)

# Seed conversation
conversations = [
    ("user", "hello echo how are you today"),
    ("echo", "i am well thank you for asking"),
    ("user", "the ocean is beautiful at sunset"),
    ("echo", "the ocean is beautiful at sunset yes"),
    ("user", "i love the sound of waves crashing"),
    ("echo", "waves crashing is peaceful and calm"),
    ("user", "tell me about the sea"),
    ("echo", "the sea is vast and deep and blue"),
    ("user", "what is your favorite color"),
    ("echo", "my favorite color is the blue of the ocean"),
]
for role, text in conversations:
    brain.add_conversation(role, text)

brain.build_vocab()
print(f"  Vocab: {brain.vocab.size} chars")
print(f"  Corpus: {len(brain.training_corpus)} chars")
print(f"  Initial neurons: {brain.model.hidden_size}")

# Train
print("  Training 100 epochs...")
brain.train(epochs=100, verbose=False)

print(f"  After training:")
print(f"    neurons: {brain.model.hidden_size}")
print(f"    loss: {brain.model.smooth_loss:.4f}")
print(f"    grows: {brain.evolution.total_grows}")
print(f"    lr mutations: {brain.evolution.total_lr_mutations}")
print(f"    lr: {brain.evolution.lr:.4f}")

test("brain trained", brain.model.smooth_loss is not None)
test("brain has loss < 4", brain.model.smooth_loss < 4.0, 
     f"loss={brain.model.smooth_loss:.4f}")

# Show mutations
if brain.recent_mutations:
    print(f"  Mutations:")
    for m in brain.recent_mutations[-5:]:
        print(f"    {m['type']}: {m['detail']}")

# Generate response
response = brain.respond("what do you think about the ocean", temperature=0.7, length=100)
print(f"  Response: '{response}'")
test("brain responds", len(response) > 0)
print()

# === 7. Persistence ===
print("--- 7. Save/Load with Evolution ---")
tmpdir = tempfile.mkdtemp()
brain.save(tmpdir)

files = sorted(os.listdir(tmpdir))
print(f"  Saved files: {files}")
test("evolution.json saved", "evolution.json" in files)
test("model.json saved", "model.json" in files)
test("corpus.txt saved", "corpus.txt" in files)
test("vocab.json saved", "vocab.json" in files)
test("convo.json saved", "convo.json" in files)

brain2 = EchoBrain.load(tmpdir)
test("brain loaded", brain2 is not None and brain2.model is not None)
test("evolution loaded", brain2.evolution is not None)
test("hidden size persisted", brain2.model.hidden_size == brain.model.hidden_size)
test("grow count persisted", brain2.evolution.total_grows == brain.evolution.total_grows)

print(f"  Loaded: neurons={brain2.model.hidden_size}, "
      f"grows={brain2.evolution.total_grows}, "
      f"loss={brain2.model.smooth_loss:.4f}")
print()

# === 8. Heavy Training — Watch It Grow ===
print("--- 8. Heavy Training (200 epochs) ---")
print(f"  Starting: neurons={brain.model.hidden_size}, loss={brain.model.smooth_loss:.4f}")

brain.train(epochs=200, verbose=False)

print(f"  After 200 epochs:")
print(f"    neurons: {brain.model.hidden_size}")
print(f"    loss: {brain.model.smooth_loss:.4f}")
print(f"    best loss: {brain.evolution.best_loss:.4f}")
print(f"    grows: {brain.evolution.total_grows}")
print(f"    lr mutations: {brain.evolution.total_lr_mutations}")
print(f"    lr: {brain.evolution.lr:.4f}")
print(f"    total mutations: {len(brain.evolution.mutations)}")

# Show full mutation history
print(f"  Full mutation history:")
for m in brain.evolution.mutations:
    print(f"    {m['type']}: {m['detail']}")

# Final response
response2 = brain.respond("the sea", temperature=0.5, length=80)
print(f"  Response to 'the sea': '{response2}'")
test("heavy training completed", brain.model.smooth_loss < 4.0)
print()

# === RESULTS ===
print("=" * 55)
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — Echo is alive.")
else:
    print(f"  ❌ {FAIL} tests failed.")
print("=" * 55)
print()