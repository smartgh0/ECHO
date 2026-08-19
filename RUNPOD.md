# RunPod Training Guide — Echo 2B Model

## Overview

Train a 2B-parameter Echo model on RunPod using Qwen 27B/32B as a teacher for distillation.

**Budget:** $35 → ~78 hours on RTX A6000 (48GB)
**Result:** 2B-parameter domain-focused LLM trained from scratch

## Step 1: Create a RunPod Account & Add Credit

1. Go to [runpod.io](https://runpod.io)
2. Sign up and add $35 credit
3. Go to **Pods** → **Deploy**

## Step 2: Deploy a GPU Pod

**Recommended:** RTX A6000 (48 GB) at $0.45/hr

1. Select template: **PyTorch 2.1+** (or any CUDA-capable template)
2. GPU: **RTX A6000**
3. Disk: 100 GB (for model + data)
4. Volume: 50 GB at `/workspace` (persistent storage)
5. Expose SSH (note the port and host)
6. Click **Deploy**

## Step 3: SSH into the Pod

```bash
ssh root@<runpod-ip> -p <port>
```

## Step 4: Run the Training Pipeline

### Option A: Full automated pipeline (recommended)

```bash
# Download and run the full pipeline
curl -fsSL https://raw.githubusercontent.com/smartgh0/ECHO/main/runpod_train.sh -o /workspace/runpod_train.sh
chmod +x /workspace/runpod_train.sh
bash /workspace/runpod_train.sh
```

This will:
1. Install Ollama + pull Qwen 32B teacher model
2. Clone Echo repository
3. Distill 50,000 Q&A pairs from Qwen 32B
4. Train the 2B model for 800K steps
5. Save checkpoint to `/workspace/domain_brain_2b/`

### Option B: Manual step-by-step

#### Install dependencies
```bash
apt-get update && apt-get install -y python3-pip git curl
pip3 install torch sentencepiece numpy
```

#### Install Ollama + pull teacher
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
sleep 5
ollama pull qwen2.5:32b
```

#### Clone Echo
```bash
cd /workspace
git clone https://github.com/smartgh0/ECHO.git
cd ECHO
```

#### Distill domain knowledge
```bash
# Edit distill_prompts.txt to add your domain-specific questions
# Then distill:
python3 distill_echo.py \
    --model qwen2.5:32b \
    --count 50000 \
    --workers 10 \
    --output pipeline/input/distill_only/distilled_data.txt \
    --max-tokens 300
```

#### Add your domain data
```bash
# Upload any domain-specific text files
# scp -P <port> your_data.txt root@<runpod-ip>:/workspace/ECHO/pipeline/input/distill_only/
```

#### Train the 2B model
```bash
# Stop Ollama to free VRAM
ollama stop qwen2.5:32b

# Start training
python3 train_domain.py \
    --input-dir pipeline/input/distill_only \
    --output-dir /workspace/domain_brain_2b \
    --profile echo-2b \
    --rebuild-tokens \
    --steps 800000 \
    --seq-len 512
```

## Step 5: Monitor Training

```bash
# Watch training progress
tail -f /tmp/echo-domain-train*.log

# Check GPU usage
nvidia-smi

# Check remaining budget
# $35 / $0.45 = 77.8 hours total
```

## Step 6: Download the Checkpoint

When training is complete (or when you want to stop):

```bash
# From your local machine:
scp -P <port> root@<runpod-ip>:/workspace/domain_brain_2b/model.pt ./domain_brain/
scp -P <port> root@<runpod-ip>:/workspace/ECHO/domain_brain/echo_domain.model ./domain_brain/
```

Or use RunPod's file browser in the web UI.

## Step 7: Run Locally

```bash
cd /home/cnos/Documents/ECHO
./echo.sh domain
```

## Cost Breakdown

| Phase | Time | Cost |
|-------|------|------|
| Setup + Ollama pull | 0.5h | $0.23 |
| Distillation (50K pairs) | ~5h | $2.25 |
| Training (800K steps) | ~73h | $32.85 |
| **Total** | **~78h** | **~$35** |

## Tips

- **Save checkpoints periodically** — if the pod crashes, you lose unsaved work
- **Use /workspace** — this persists across pod restarts
- **Stop the pod when not training** — you only pay while it's running
- **Monitor with `nvidia-smi`** — ensure GPU is being used
- **Use `tmux`** — so training continues if SSH disconnects:
  ```bash
  apt-get install -y tmux
  tmux new -s train
  # run training inside tmux
  # detach: Ctrl+B, D
  # reattach: tmux attach -t train
  ```

## Available Teacher Models

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| qwen3.8:latest | ~16GB | Excellent | Medium (~5s/response) |
| qwen2.5:32b | 32B | Excellent | Slow (~10s/response) |
| qwen2.5:14b | 14B | Very good | Medium (~5s/response) |
| qwen2.5:7b | 7B | Good | Fast (~2s/response) |
| qwen2.5:3b | 3B | Decent | Very fast (~1s/response) |

For 50K pairs with 10 workers:
- 32B: ~14 hours ($6.30)
- 14B: ~7 hours ($3.15)
- 7B: ~3 hours ($1.35)

**Recommendation:** Use `qwen2.5:14b` — good quality at reasonable speed, leaves more budget for training.