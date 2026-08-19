#!/usr/bin/env python3
"""
ECHO PIPELINE TEST — Verify the full distillation pipeline works.
Usage: python3 echo/test_pipeline.py
"""
import sys, os, time, json, tempfile, shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

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
print("  ECHO PIPELINE TEST — Distillation Pipeline")
print("=" * 65)
print()

# === 1. Format detection ===
print("--- 1. Format Detection ---")
from pipeline import detect_format, split_prose_to_conversation, process_file

conv_text = "user: hello\necho: hi there\nuser: how are you"
prose_text = "The ocean is beautiful at sunset. The waves crash on the shore."

test("detect conversation", detect_format(conv_text) == 'conversation')
test("detect prose", detect_format(prose_text) == 'prose')
print()

# === 2. Prose splitting ===
print("--- 2. Prose Splitting ---")
split = split_prose_to_conversation(prose_text, chunk_size=5)
test("split has user/echo", "user:" in split and "echo:" in split)
test("split alternates roles", split.count("user:") >= 1 and split.count("echo:") >= 1)
print(f"  Split result: {split[:100]}...")
print()

# === 3. File processing ===
print("--- 3. File Processing ---")
# Create temp files
tmpdir = tempfile.mkdtemp()

conv_file = os.path.join(tmpdir, "conv.txt")
with open(conv_file, "w") as f:
    f.write(conv_text)
result = process_file(conv_file)
test("process conversation file", "user:" in result and "echo:" in result)

prose_file = os.path.join(tmpdir, "prose.txt")
with open(prose_file, "w") as f:
    f.write(prose_text)
result = process_file(prose_file)
test("process prose file", "user:" in result or len(result) > 0)

# Empty file
empty_file = os.path.join(tmpdir, "empty.txt")
with open(empty_file, "w") as f:
    f.write("")
result = process_file(empty_file)
test("process empty file", result == "")
print()

# === 4. Full pipeline run ===
print("--- 4. Full Pipeline Run ---")
from pipeline import stage_ingest, stage_train, stage_dream, stage_evaluate

# Set up isolated brain dir
test_brain_dir = os.path.join(tmpdir, "brain")
os.makedirs(test_brain_dir, exist_ok=True)

# Create a test brain
from echo_brain import EchoBrain
brain = EchoBrain(hidden_size=16, learning_rate=0.02, mode='quantum_layer',
                  quantum_samples=4, dropout_rate=0.3)

# Add seed conversation
seed = [
    ("user", "hello echo"), ("echo", "hello I am echo"),
    ("user", "the ocean is beautiful"), ("echo", "the ocean is beautiful yes"),
    ("user", "I love the waves"), ("echo", "waves are peaceful and calm"),
]
for role, text in seed:
    brain.add_conversation(role, text)
brain.build_vocab()
brain.save(test_brain_dir)

# Test training stage
print("  Testing stage_train...")
# Monkey-patch BRAIN_DIR
import pipeline as pl
original_brain_dir = pl.BRAIN_DIR
pl.BRAIN_DIR = test_brain_dir

train_data = stage_train(epochs=20)
test("stage_train returns data", train_data is not None)
test("stage_train has loss", 'loss_after' in train_data)
test("stage_train loss < 4", train_data['loss_after'] < 4.0, f"loss={train_data['loss_after']:.4f}")
print()

# Test dream stage
print("  Testing stage_dream...")
dream_data = stage_dream(cycles=20)
test("stage_dream returns data", dream_data is not None)
test("stage_dream has loss", 'loss_after' in dream_data)
print()

# Test evaluate stage
print("  Testing stage_evaluate...")
eval_data = stage_evaluate()
test("stage_evaluate returns data", eval_data is not None)
test("stage_evaluate has samples", 'samples' in eval_data)
test("stage_evaluate has loss", 'loss' in eval_data)
print()

# Restore brain dir
pl.BRAIN_DIR = original_brain_dir

# === 5. Ingest from files ===
print("--- 5. Ingest from Files ---")
# Create test input directory
test_input = os.path.join(tmpdir, "input")
os.makedirs(test_input, exist_ok=True)

# Write test files
with open(os.path.join(test_input, "test1.txt"), "w") as f:
    f.write("user: hello\necho: hi\nuser: bye\necho: goodbye\n")
with open(os.path.join(test_input, "test2.txt"), "w") as f:
    f.write("The sea is deep and blue. The waves are beautiful at sunset.")

# Test ingest
original_input = pl.INPUT_DIR
pl.INPUT_DIR = test_input
pl.BRAIN_DIR = test_brain_dir

ingest_chars = stage_ingest(smart_split=True)
test("stage_ingest returns chars", ingest_chars > 0, f"chars={ingest_chars}")
test("files processed", not os.path.exists(os.path.join(test_input, "test1.txt")))
test("files moved to processed", os.path.exists(
    os.path.join(test_input, "processed", "test1.txt")))
print()

pl.INPUT_DIR = original_input
pl.BRAIN_DIR = original_brain_dir

# === 6. Report generation ===
print("--- 6. Report Generation ---")
from pipeline import stage_report
report = stage_report(
    ingest_chars,
    train_data,
    dream_data,
    eval_data,
    total_time=5.0
)
test("report has timestamp", 'timestamp' in report)
test("report has train data", 'train' in report)
test("report has dream data", 'dream' in report)
test("report has eval data", 'eval' in report)
print()

# === 7. Commands work ===
print("--- 7. Pipeline Commands ---")
# Test status
pl.BRAIN_DIR = test_brain_dir
try:
    cmd_status = None
    import argparse
    from pipeline import cmd_status
    cmd_status(argparse.Namespace())
    test("cmd_status runs", True)
except Exception as e:
    test("cmd_status runs", False, str(e))

# Test sample
try:
    from pipeline import cmd_sample
    cmd_sample(argparse.Namespace())
    test("cmd_sample runs", True)
except Exception as e:
    test("cmd_sample runs", False, str(e))

pl.BRAIN_DIR = original_brain_dir
print()

# === 8. Cleanup ===
shutil.rmtree(tmpdir, ignore_errors=True)

# === RESULTS ===
print()
print("=" * 65)
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✅ ALL TESTS PASSED — Pipeline is ready.")
else:
    print(f"  ❌ {FAIL} tests failed.")
print("=" * 65)