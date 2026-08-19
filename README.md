# ECHO — A Mind That Grows From Your Words

A self-contained, character-level neural language model built entirely from scratch.

The legacy modes use pure Python. The transformer mode uses PyTorch for GPU training.

## What Is Echo?

Echo is a recurrent neural network (RNN) that learns character-by-character from your conversation. Every word you type becomes training data. The more you talk, the more it learns. It saves its "brain" to disk as plain text and JSON files — no databases, no cloud, no API keys.

## Files

```
echo/
├── echo.sh          — Main entry point (shell script)
├── echo_matrix.py   — Matrix math from scratch (matmul, transpose, tanh, softmax)
├── echo_rnn.py      — RNN engine (forward pass, backprop through time, SGD, grow, prune)
├── echo_evolve.py   — Adaptive neurogenesis, synaptic pruning, learning rate mutation
├── echo_brain.py    — Brain orchestration (vocab, memory, training, persistence, evolution)
├── echo_chat.py     — Interactive chat interface with live mutation display
├── echo_dream.py    — Dream mode (background training + evolution)
└── brain/           — Created on first save
    ├── corpus.txt   — Raw training text (human-readable)
    ├── convo.json   — Conversation log
    ├── vocab.json   — Character vocabulary
    ├── model.json   — Model weights
    └── evolution.json — Evolution state (hidden size, LR, mutation history)
```

## Quick Start

```bash
chmod +x echo.sh
./echo.sh
```

### Local GPU Transformer

An 8 GB GPU can train Echo's `local-8gb` profile using CUDA mixed precision:

```bash
ECHO_MODE=quantum_transformer ECHO_TRANSFORMER_PROFILE=local-8gb ./echo.sh transformer
```

This profile is about 202M parameters. The `echo-2b` profile is available for
larger hardware, but is not suitable for full training on an 8 GB GPU.
Transformer checkpoints are stored as binary `model.pt` files.

For memory-efficient adaptation with a frozen base and trainable attention
adapters, use:

```bash
ECHO_MODE=quantum_transformer ECHO_TRANSFORMER_PROFILE=local-8gb-lora ./echo.sh transformer
```

For a roughly 0.5B-parameter model trained from random initialization, use:

```bash
ECHO_MODE=quantum_transformer \
ECHO_TRANSFORMER_PROFILE=from-scratch-0.5b \
./echo.sh transformer
```

This profile trains every parameter with CUDA mixed precision, activation
checkpointing, SGD, context 256, and gradient accumulation. It is designed to
fit an 8 GB GPU, but it will train much more slowly than the smaller profiles.

For domain pretraining from all processed files with a subword tokenizer, run:

```bash
python3 train_domain.py --input-dir pipeline/input/processed \
   --output-dir domain_brain --vocab-size 16384 --seq-len 256 --steps 10000
```

This creates a SentencePiece tokenizer, streams token IDs to
`domain_brain/tokens.u32`, trains all 0.5B parameters from random initialization,
and saves the binary checkpoint to `domain_brain/model.pt`. The resulting
domain model is loaded through the domain trainer artifacts; the legacy
`brain/` character-chat checkpoint is not overwritten.

To add more `.txt` data, place it in `pipeline/input/`, ingest it, rebuild the
domain token stream, and resume:

```bash
python3 pipeline.py ingest
python3 train_domain.py --input-dir pipeline/input/processed \
   --output-dir domain_brain --rebuild-tokens --resume --steps 10000
```

Start the trained domain model with:

```bash
./echo.sh domain
```

Just start typing. Echo learns from every message.

## Commands (inside chat)

| Command | Description |
|---------|-------------|
| `:train 50` | Train 50 epochs on current memory |
| `:dream 100` | Dream (silent training) for 100 cycles |
| `:temp 0.8` | Set temperature (0.1=conservative, 1.5=wild) |
| `:info` | Show brain status + evolution stats |
| `:evolve` | Show mutation history and evolution details |
| `:vocab` | Show all characters in vocabulary |
| `:corpus` | Show training corpus |
| `:save` | Save brain to disk |
| `:reset` | Wipe brain and start over |
| `:quit` | Save and exit |

## Dream Mode

Run in a separate terminal while you're not chatting:

```bash
./echo.sh dream
```

Echo will silently train on its existing memory, consolidating what it's learned — like sleep in biological brains. It occasionally generates "dream snippets" — free-form text from its current state.

## How It Works

1. **Matrix math** (`echo_matrix.py`) — Every operation is hand-written: matrix multiplication, transpose, element-wise ops, tanh, softmax, cross-entropy loss, gradient clipping. No libraries.

2. **RNN engine** (`echo_rnn.py`) — A character-level RNN with one hidden layer:
   - Forward pass: `h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)`, `y_t = softmax(W_hy * h_t + b_y)`
   - Backward pass: Backpropagation Through Time (BPTT), unrolled over the sequence
   - Optimizer: SGD with gradient clipping
   - Sampling: Temperature-controlled character sampling
   - **Neurogenesis**: `grow(n)` adds new neurons with random weights
   - **Pruning**: `prune(idx)` removes a neuron and all its connections

3. **Evolution** (`echo_evolve.py`) — The brain that grows itself:
   - **Neurogenesis**: When loss plateaus, new neurons are born (up to 512 max)
   - **Synaptic pruning**: When the brain is performing well, weak neurons are removed
   - **Learning rate mutation**: LR shifts up and down based on training dynamics
   - **Neuron importance scoring**: Each neuron's contribution is tracked via gradient magnitude
   - All evolution state is persisted in `evolution.json`

4. **Brain** (`echo_brain.py`) — Manages vocabulary, conversation memory, training cycles, evolution integration, and response generation. Builds context from recent conversation turns, primes the model, and samples a response.

5. **Persistence** — Everything saves as plain text and JSON. `corpus.txt` is human-readable — you can literally open it and read everything Echo has learned. `model.json` contains the weights. `evolution.json` tracks how the brain has grown and mutated over time.

## Architecture

```
Input (one-hot char) ──→ [W_xh] ──┐
                                   ├──→ tanh ──→ [W_hy] ──→ softmax ──→ output char
Previous hidden state ──→ [W_hh] ──┘
```

- Hidden size: 64 neurons (configurable)
- Learning rate: 0.01
- Sequence length: 25 characters (truncated BPTT)
- Gradient clipping: max norm 5.0

## The Truth

This is a small model. It won't be GPT-4. It won't be coherent most of the time at first. But it's *yours* — every weight, every gradient, every multiplication is visible and understandable. You can open `corpus.txt` and see exactly what it knows. You can open `model.json` and see every number in its brain.

The more you talk, the more it learns. The more it dreams, the more it consolidates. It's not magic — it's math, and you can see all of it.

## License

Public domain. Build on it. Break it. Make it yours.