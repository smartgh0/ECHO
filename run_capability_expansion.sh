#!/usr/bin/env bash
# End-to-end pod pipeline for the bounded two-day Echo capability expansion.
set -euo pipefail

ROOT="${ECHO_ROOT:-/workspace/ECHO}"
DOMAIN="${ECHO_DOMAIN:-/workspace/domain_brain_2b}"
CORPUS="${ECHO_CAPABILITY_CORPUS:-$ROOT/pipeline/input/capability_2b}"
BASELINE="${ECHO_BASELINE:-$DOMAIN/snapshots/step-161500.pt}"
TOKENIZER="${ECHO_TOKENIZER:-$DOMAIN/echo_domain.model}"
TARGET_TOKENS="${ECHO_TARGET_TOKENS:-2000000000}"
LOG_DIR="${ECHO_LOG_DIR:-/workspace/echo_capability_logs}"
ALLCOMBINED="${ECHO_ALLCOMBINED:-$ROOT/pipeline/input/distill_only/AllCombined.txt}"

mkdir -p "$LOG_DIR"
exec 9>"$LOG_DIR/pipeline.lock"
if ! flock -n 9; then
    echo "Another capability pipeline owns $LOG_DIR/pipeline.lock" >&2
    exit 1
fi

cd "$ROOT"
python3 -m pip install -q "datasets>=3" huggingface_hub pyarrow

if [[ ! -f "$CORPUS/MANIFEST.json" ]]; then
    rm -rf "$CORPUS"
    mkdir -p "$CORPUS"
    python3 build_capability_corpus.py \
        --output-dir "$CORPUS" \
        --tokenizer "$TOKENIZER" \
        --target-tokens "$TARGET_TOKENS" \
        --allcombined "$ALLCOMBINED" \
        2>&1 | tee "$LOG_DIR/build-corpus.log"
fi

python3 - "$CORPUS/MANIFEST.json" "$TARGET_TOKENS" <<'PY'
import json
import sys

path, target = sys.argv[1], int(sys.argv[2])
manifest = json.load(open(path, encoding="utf-8"))
accepted = int(manifest.get("total_accepted_tokens", 0))
minimum = int(target * 0.92)
if accepted < minimum:
    raise SystemExit(
        f"Corpus validation failed: {accepted:,} tokens; require at least {minimum:,}"
    )
if not manifest.get("sources"):
    raise SystemExit("Corpus manifest has no source/license records")
print(f"Validated corpus: {accepted:,} accepted tokens")
PY

if [[ ! -s "$BASELINE" ]]; then
    echo "Required baseline is missing: $BASELINE" >&2
    exit 1
fi
if [[ ! -s "$TOKENIZER" ]]; then
    echo "Tokenizer is missing: $TOKENIZER" >&2
    exit 1
fi

# Restore model weights atomically while leaving the selected baseline untouched.
cp "$BASELINE" "$DOMAIN/model.pt.restore"
mv "$DOMAIN/model.pt.restore" "$DOMAIN/model.pt"

python3 train_domain.py \
    --input-dir "$CORPUS" \
    --output-dir "$DOMAIN" \
    --profile echo-2b \
    --resume \
    --fresh-optimizer \
    --rebuild-tokens \
    --weighted-sampling \
    --mask-user-tokens \
    --capability-mix \
    --checkpoint-only \
    --steps 135000 \
    --seq-len 512 \
    --batch-size 24 \
    --lr 0.00002 \
    --reset-lr-schedule \
    --max-train-tokens 1658880000 \
    --save-every 10000 \
    2>&1 | tee "$LOG_DIR/train-seq512.log"

python3 train_domain.py \
    --input-dir "$CORPUS" \
    --output-dir "$DOMAIN" \
    --profile echo-2b \
    --resume \
    --weighted-sampling \
    --mask-user-tokens \
    --capability-mix \
    --checkpoint-only \
    --steps 15000 \
    --seq-len 1024 \
    --batch-size 8 \
    --lr 0.000008 \
    --reset-lr-schedule \
    --max-train-tokens 122880000 \
    --save-every 5000 \
    2>&1 | tee "$LOG_DIR/train-seq1024.log"

python3 echo_eval_capability.py \
    --domain-dir "$DOMAIN" \
    --baseline "$BASELINE" \
    --candidate "$DOMAIN/model.pt" \
    --out-json "$LOG_DIR/eval-capability.json" \
    2>&1 | tee "$LOG_DIR/eval-capability.log"

python3 - "$LOG_DIR/eval-capability.json" "$DOMAIN" "$BASELINE" <<'PY'
import json
import os
import shutil
import sys

report_path, domain, baseline = sys.argv[1:]
report = json.load(open(report_path, encoding="utf-8"))
decision = report["decision"]
candidate = os.path.join(domain, "model.pt")
if decision["continue"]:
    print("Capability run passed the continuation gate; keeping candidate model.pt")
else:
    preserved = os.path.join(domain, "model-capability-candidate.pt")
    shutil.copy2(candidate, preserved)
    temporary = os.path.join(domain, "model.pt.restore")
    shutil.copy2(baseline, temporary)
    os.replace(temporary, candidate)
    print(
        "Capability run failed the continuation gate; preserved the candidate at "
        f"{preserved} and restored the 161.5k baseline"
    )
PY

echo "CAPABILITY_PIPELINE_COMPLETE"
