#!/bin/bash
# ============================================================
# RUNPOD DEPLOYMENT SCRIPT — Echo 2B Training Pipeline
# ============================================================
# This script runs on a RunPod GPU instance to:
#   1. Install Ollama + pull Qwen 27B teacher model
#   2. Clone Echo repo + install dependencies
#   3. Distill domain knowledge from Qwen 27B
#   4. Train the 2B Echo model from scratch
#   5. Save checkpoint to /workspace for download
#
# Usage on RunPod:
#   1. Deploy an RTX A6000 (48GB) pod with /workspace volume
#   2. Set custom script in pod template to:
#      bash /workspace/runpod_train.sh
#   3. Or SSH in and run: bash runpod_train.sh
# ============================================================

set -e

# --- Configuration ---
ECHO_REPO="https://github.com/smartgh0/ECHO.git"
TEACHER_MODEL="qwen2.5:32b"  # Qwen 2.5 32B (closest to 27B)
TRAIN_PROFILE="echo-2b"
TRAIN_STEPS=800000            # ~78 hours on A6000
SEQ_LEN=512
DISTILL_COUNT=10000           # Q&A pairs to distill (10K unique prompts available)
DISTILL_WORKERS=10
DOMAIN_PROMPTS_FILE="distill_prompts.txt"

echo "============================================================"
echo "  ECHO 2B TRAINING PIPELINE — RUNPOD"
echo "============================================================"
echo ""

# --- Step 1: Install dependencies ---
echo "[1/5] Installing dependencies..."
apt-get update -qq && apt-get install -y -qq python3 python3-pip git curl > /dev/null 2>&1
pip3 install -q torch sentencepiece numpy

# --- Step 2: Install Ollama + pull teacher model ---
echo "[2/5] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
sleep 5

echo "  Pulling teacher model: $TEACHER_MODEL..."
ollama pull $TEACHER_MODEL
echo "  Teacher model ready."

# --- Step 3: Clone Echo ---
echo "[3/5] Cloning Echo repository..."
cd /workspace
if [ -d "ECHO" ]; then
    cd ECHO && git pull
else
    git clone $ECHO_REPO
    cd ECHO
fi
pip3 install -q sentencepiece torch numpy

# --- Step 4: Distill domain knowledge from Qwen ---
echo "[4/5] Distilling $DISTILL_COUNT Q&A pairs from $TEACHER_MODEL..."
echo "  This will take several hours..."
python3 distill_echo.py \
    --model $TEACHER_MODEL \
    --count $DISTILL_COUNT \
    --workers $DISTILL_WORKERS \
    --output pipeline/input/distill_only/distilled_data.txt \
    --max-tokens 300

echo "  Distillation complete."
wc -c pipeline/input/distill_only/distilled_data.txt

# --- Step 5: Train the 2B model ---
echo "[5/5] Training Echo 2B model ($TRAIN_STEPS steps, seq_len=$SEQ_LEN)..."
echo "  Profile: $TRAIN_PROFILE"
echo "  Estimated time: ~78 hours on A6000"
echo ""

# Stop Ollama to free VRAM for training
ollama stop $TEACHER_MODEL 2>/dev/null || true
sleep 3

python3 train_domain.py \
    --input-dir pipeline/input/distill_only \
    --output-dir /workspace/domain_brain_2b \
    --profile $TRAIN_PROFILE \
    --rebuild-tokens \
    --steps $TRAIN_STEPS \
    --seq-len $SEQ_LEN

echo ""
echo "============================================================"
echo "  TRAINING COMPLETE"
echo "============================================================"
echo ""
echo "Checkpoint saved to: /workspace/domain_brain_2b/"
echo "  model.pt          — 2B model weights"
echo "  echo_domain.model — SentencePiece tokenizer"
echo "  tokens.u32        — encoded corpus"
echo ""
echo "Download from RunPod file browser or use:"
echo "  scp -P <port> root@<runpod-ip>:/workspace/domain_brain_2b/model.pt ."
echo ""
echo "To run locally:"
echo "  cp domain_brain_2b/* domain_brain/"
echo "  ./echo.sh domain"