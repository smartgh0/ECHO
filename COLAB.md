# Echo LLM Training on Google Colab

## Quick Start (copy into Colab cells)

### Cell 1: Setup
```python
!pip install -q sentencepiece torch numpy

# Clone your repo (replace with your GitHub URL)
!git clone https://github.com/YOUR_USERNAME/ECHO.git /content/ECHO
%cd /content/ECHO

# Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
```

### Cell 2: Upload Data
```python
# Option A: Upload files manually via Colab sidebar
# Option B: Upload from Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy your data files
import os, shutil
os.makedirs('pipeline/input/distill_only', exist_ok=True)

# Copy from Google Drive (adjust paths)
# shutil.copy('/content/drive/MyDrive/ECHO/AllCombined.txt', 'pipeline/input/distill_only/')
# shutil.copy('/content/drive/MyDrive/ECHO/distilled_data.txt', 'pipeline/input/distill_only/')

# Or upload directly
from google.colab import files
uploaded = files.upload()
for name in uploaded:
    shutil.copy(name, f'pipeline/input/distill_only/{name}')
```

### Cell 3: Train
```python
# Train the 302M model (fits in 16GB easily)
!python3 train_domain.py \
  --input-dir pipeline/input/distill_only \
  --output-dir domain_brain \
  --profile coherent-150m \
  --rebuild-tokens \
  --steps 20000 \
  --seq-len 512

# Or train the full 0.5B model (fits in 16GB with AdamW)
# !python3 train_domain.py \
#   --input-dir pipeline/input/distill_only \
#   --output-dir domain_brain \
#   --profile from-scratch-0.5b \
#   --rebuild-tokens \
#   --steps 20000 \
#   --seq-len 256
```

### Cell 4: Fine-tune with Q&A (after pre-training)
```python
# Resume with lower LR to learn conversational format
!python3 train_domain.py \
  --input-dir pipeline/input/distill_only \
  --output-dir domain_brain \
  --profile coherent-150m \
  --rebuild-tokens \
  --resume \
  --lr 0.0001 \
  --steps 10000 \
  --seq-len 512
```

### Cell 5: Test the model
```python
!pip install -q sentencepiece
from echo_tokenizer import EchoTokenizer
from echo_transformer import QuantumTransformerLM
import torch

ckpt = torch.load('domain_brain/model.pt', map_location='cpu', weights_only=True)
model = QuantumTransformerLM.from_dict({**ckpt['config'], 'state_dict': ckpt['state_dict']})
tokenizer = EchoTokenizer('domain_brain/echo_domain.model')

def chat(prompt, length=100, temperature=0.7):
    token_ids = tokenizer.encode(f"user: {prompt}\necho:")
    generated = []
    model.eval()
    with torch.no_grad():
        for _ in range(length):
            ctx = torch.tensor(token_ids[-model.max_context:], dtype=torch.long, device=model.device)
            logits = model(ctx)[0, -1] / max(temperature, 0.01)
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            generated.append(next_id)
            token_ids.append(next_id)
            if next_id == 2:
                break
    return tokenizer.decode(generated)

print(chat("What is gravity?"))
print(chat("Who are you?"))
print(chat("What is machine learning?"))
```

### Cell 6: Download checkpoint
```python
# Download to your local machine
from google.colab import files
files.download('domain_brain/model.pt')
files.download('domain_brain/echo_domain.model')

# Or save to Google Drive
import shutil
shutil.copy('domain_brain/model.pt', '/content/drive/MyDrive/ECHO/model.pt')
shutil.copy('domain_brain/echo_domain.model', '/content/drive/MyDrive/ECHO/echo_domain.model')
```

## Colab Free Tier Limits

- **GPU**: T4 (16 GB VRAM) — 2× your local GPU
- **Session**: ~12 hours before disconnect
- **Disk**: ~100 GB temporary storage
- **Tip**: Save checkpoints to Google Drive periodically

## What fits in 16 GB

| Profile | Params | Peak VRAM | Fits Colab? |
|---------|--------|-----------|-------------|
| coherent-150m | 302M | 5.7 GB | Yes (easily) |
| from-scratch-0.5b | 510M | ~8 GB | Yes |
| echo-2b | 1.8B | ~22 GB | No (needs Colab Pro) |