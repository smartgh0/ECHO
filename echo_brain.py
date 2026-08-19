# ============================================================
# ECHO BRAIN — Memory, vocabulary, and training orchestration
# Handles: vocab building, text encoding, memory persistence,
#          training loops, and response generation.
# ============================================================

import json
import os
import random
import sys

import torch

from echo_rnn import EchoRNN
from echo_quantum import QuantumRNN
from echo_quantum_layer import QuantumLayerRNN
from echo_transformer import QuantumTransformerLM
from echo_model_config import get_transformer_profile
from echo_evolve import EvolutionTracker

# ----------------------------------------------------------
# Default configuration
# ----------------------------------------------------------

DEFAULT_HIDDEN_SIZE = 48
DEFAULT_LEARNING_RATE = 0.04
DEFAULT_SEQ_LEN = 25
DEFAULT_TEMPERATURE = 0.7
DEFAULT_GEN_LENGTH = 200

# Special characters
CHAR_NEWLINE = "\n"
CHAR_SPACE = " "

# ----------------------------------------------------------
# Vocabulary management
# ----------------------------------------------------------

class Vocabulary:
    """Maps characters to indices and back. Built from training text."""

    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = []
        self.frozen = False

    @property
    def size(self):
        return len(self.idx_to_char)

    def build(self, text):
        """Build vocabulary from a text corpus."""
        chars = sorted(set(text))
        self.idx_to_char = list(chars)
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.frozen = True
        return self.size

    def add_char(self, ch):
        """Add a single character if not present (for expanding vocab)."""
        if ch not in self.char_to_idx:
            self.char_to_idx[ch] = len(self.idx_to_char)
            self.idx_to_char.append(ch)

    def encode(self, text):
        """Encode a string into a list of indices."""
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode(self, indices):
        """Decode a list of indices back into a string."""
        return "".join(self.idx_to_char[i] for i in indices if 0 <= i < len(self.idx_to_char))

    def one_hot(self, idx):
        """Create a one-hot vector for a character index."""
        vec = [0.0] * self.size
        if 0 <= idx < self.size:
            vec[idx] = 1.0
        return vec

    def encode_one_hot(self, text):
        """Encode a string into a list of one-hot vectors."""
        return [self.one_hot(self.char_to_idx.get(c, 0)) for c in text]

    def to_dict(self):
        return {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char
        }

    @classmethod
    def from_dict(cls, d):
        v = cls()
        v.char_to_idx = d['char_to_idx']
        v.idx_to_char = d['idx_to_char']
        v.frozen = True
        return v

# ----------------------------------------------------------
# Brain — wraps the RNN + vocabulary + memory
# ----------------------------------------------------------

class EchoBrain:
    """The complete mind: vocabulary + RNN model + conversation memory + evolution.

    Supports two modes:
    - classical: standard RNN with deterministic weights
    - quantum: superposition weights with Monte Carlo sampling
    """

    def __init__(self, hidden_size=48, learning_rate=0.04,
                 max_hidden=512, min_hidden=16, mode='quantum_layer',
                 quantum_samples=2, dropout_rate=0.0,
                 transformer_profile='small'):
        self.vocab = Vocabulary()
        self.model = None
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.conversation_log = []
        self.training_corpus = ""
        self.mode = mode  # 'classical', 'quantum', or 'quantum_layer'
        self.quantum_samples = quantum_samples
        self.dropout_rate = dropout_rate
        self.transformer_profile = transformer_profile

        # Evolution system
        self.evolution = EvolutionTracker(
            initial_hidden=hidden_size,
            min_hidden=min_hidden,
            max_hidden=max_hidden,
            initial_lr=learning_rate
        )

        self.recent_mutations = []

    # --- Corpus and vocabulary ---

    def add_conversation(self, role, text):
        """Add a conversation turn and append to training corpus."""
        formatted = f"{role}: {text}\n"
        self.conversation_log.append((role, text))
        self.training_corpus += formatted

    def build_vocab(self):
        """Build vocabulary from the current corpus."""
        if not self.training_corpus:
            self.training_corpus = "hello\nuser: hello\necho: hello\n "
        self.vocab.build(self.training_corpus)
        current_hidden = self.evolution.hidden_size

        if self.mode == 'quantum_transformer':
            profile = get_transformer_profile(self.transformer_profile)
            self.model = QuantumTransformerLM(
                vocab_size=self.vocab.size,
                profile_name=self.transformer_profile,
                **profile
            )
        elif self.mode == 'quantum_layer':
            self.model = QuantumLayerRNN(
                vocab_size=self.vocab.size,
                hidden_size=current_hidden,
                learning_rate=self.evolution.lr,
                n_samples=min(4, self.quantum_samples),  # Layer mode needs fewer samples
                dropout_rate=self.dropout_rate
            )
        elif self.mode == 'quantum':
            self.model = QuantumRNN(
                vocab_size=self.vocab.size,
                hidden_size=current_hidden,
                learning_rate=self.evolution.lr,
                n_samples=self.quantum_samples
            )
        else:
            self.model = EchoRNN(
                vocab_size=self.vocab.size,
                hidden_size=current_hidden,
                learning_rate=self.evolution.lr
            )

    def expand_vocab(self, text):
        """Expand vocabulary to cover new characters (requires model resize)."""
        new_chars = set(text) - set(self.vocab.char_to_idx.keys())
        if not new_chars:
            return False
        self.training_corpus += text
        self.vocab.build(self.training_corpus)
        current_hidden = self.evolution.hidden_size

        if self.mode == 'quantum_transformer':
            profile = get_transformer_profile(self.transformer_profile)
            self.model = QuantumTransformerLM(
                vocab_size=self.vocab.size,
                profile_name=self.transformer_profile,
                **profile
            )
        elif self.mode == 'quantum_layer':
            self.model = QuantumLayerRNN(
                vocab_size=self.vocab.size,
                hidden_size=current_hidden,
                learning_rate=self.evolution.lr,
                n_samples=min(4, self.quantum_samples),
                dropout_rate=self.dropout_rate
            )
        elif self.mode == 'quantum':
            self.model = QuantumRNN(
                vocab_size=self.vocab.size,
                hidden_size=current_hidden,
                learning_rate=self.evolution.lr,
                n_samples=self.quantum_samples
            )
        else:
            self.model = EchoRNN(
                vocab_size=self.vocab.size,
                hidden_size=current_hidden,
                learning_rate=self.evolution.lr
            )
        return True

    # --- Training ---

    def train(self, epochs=10, seq_len=DEFAULT_SEQ_LEN, verbose=True):
        """Train the model on the current corpus, with adaptive evolution.

        Works in both classical and quantum modes.
        In quantum mode, each training step samples K configurations
        from the superposition and amplifies the best one.
        """
        if self.model is None:
            self.build_vocab()

        if len(self.training_corpus) < 2:
            if verbose:
                print("  [echo] not enough text to train")
            return

        corpus_indices = self.vocab.encode(self.training_corpus)
        n = len(corpus_indices)

        for epoch in range(epochs):
            # Sync learning rate from evolution
            self.model.learning_rate = self.evolution.lr

            if self.mode == 'quantum_transformer':
                batch_count = (self.model.batch_size *
                               self.model.gradient_accumulation_steps)
                input_batch = []
                target_batch = []
                for _ in range(batch_count):
                    if n <= seq_len:
                        start, end = 0, n - 1
                    else:
                        start = random.randint(0, n - seq_len - 1)
                        end = start + seq_len
                    input_batch.append(corpus_indices[start:end])
                    target_batch.append(corpus_indices[start + 1:end + 1])
                loss, _ = self.model.train_step_batch(input_batch, target_batch)
            else:
                if n <= seq_len:
                    start, end = 0, n - 1
                else:
                    start = random.randint(0, n - seq_len - 1)
                    end = start + seq_len
                inputs = [self.vocab.one_hot(corpus_indices[i]) for i in range(start, end)]
                targets = [corpus_indices[i] for i in range(start + 1, end + 1)]
                loss, _ = self.model.train_step(inputs, targets)

            self.model.total_epochs += 1
            if self.model.smooth_loss is None:
                self.model.smooth_loss = loss
            else:
                self.model.smooth_loss = 0.99 * self.model.smooth_loss + 0.01 * loss

            # --- Evolution check (same for both modes) ---
            mutation = None if self.mode == 'quantum_transformer' else self.evolution.record_loss(loss)
            if mutation is not None:
                self._apply_mutation(mutation)
                if verbose:
                    self._print_mutation(mutation)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                extra = ""
                if self.mode in ('quantum', 'quantum_transformer'):
                    qs = self.model.quantum_stats()
                    extra = f"  entropy={qs['avg_entropy']:.3f}  collapsed={qs.get('collapsed', 0)}"
                print(f"  [echo] epoch {epoch+1}/{epochs}  loss={loss:.4f}  "
                      f"smooth={self.model.smooth_loss:.4f}  "
                      f"neurons={self.model.hidden_size}  "
                      f"lr={self.evolution.lr:.4f}{extra}")

    def _apply_mutation(self, mutation):
        """Apply a mutation event to the model."""
        mtype = mutation['type']

        if mtype == 'GROW':
            # Compound neurogenesis: add new neurons
            n_new = mutation['n_new']
            self.model.grow(n_new)

        elif mtype in ('LR_MUTATE', 'LR_BOOST'):
            # Learning rate mutation — already updated in evolution object
            self.model.learning_rate = self.evolution.lr

        # No PRUNE case — we don't prune. Ever.

        # Record for display
        self.recent_mutations.append(mutation)
        if len(self.recent_mutations) > 20:
            self.recent_mutations.pop(0)

    def _print_mutation(self, mutation):
        """Print a mutation event with appropriate coloring."""
        mtype = mutation['type']
        detail = mutation['detail']

        if mtype == 'GROW':
            print(f"  \033[32m[evolve] {detail}\033[0m")
        elif mtype in ('LR_MUTATE', 'LR_BOOST'):
            print(f"  \033[36m[evolve] {detail}\033[0m")
        # No PRUNE coloring — we don't prune

    def dream(self, cycles=50, seq_len=DEFAULT_SEQ_LEN, verbose=False):
        """Dream mode: train silently on existing memory.
        Like sleep consolidation — the brain rehearses what it knows.
        """
        if self.model is None:
            self.build_vocab()
        if len(self.training_corpus) < 2:
            return

        self.train(epochs=cycles, seq_len=seq_len, verbose=verbose)

    # --- Generation ---

    def respond(self, user_input, temperature=DEFAULT_TEMPERATURE, length=DEFAULT_GEN_LENGTH):
        """Generate a response to user input.
        
        Strategy: prime the model with recent conversation context,
        then sample characters until a newline or max length.
        """
        if self.model is None:
            self.build_vocab()

        # Build context: last few exchanges + the new input
        context = ""
        for role, text in self.conversation_log[-6:]:  # Last 6 turns for context
            context += f"{role}: {text}\n"
        context += f"echo: "

        # Encode context as one-hot
        seed = self.vocab.encode_one_hot(context[-100:])  # Last 100 chars of context

        # Sample
        sampled = self.model.sample(seed, length=length, temperature=temperature)

        # Decode and cut at first newline (end of response)
        raw = self.vocab.decode(sampled)
        if "\n" in raw:
            raw = raw[:raw.index("\n")]

        return raw.strip() if raw.strip() else "..."

    # --- Persistence (plain text + JSON) ---

    def save(self, brain_dir="echo/brain"):
        """Save the brain to disk.
        
        Creates:
            brain_dir/corpus.txt    — raw training text
            brain_dir/convo.json    — conversation log
            brain_dir/model.json    — model weights
            brain_dir/vocab.json    — vocabulary
            brain_dir/evolution.json — evolution state (hidden size, LR, mutations)
        """
        os.makedirs(brain_dir, exist_ok=True)

        # Save corpus as plain text
        with open(os.path.join(brain_dir, "corpus.txt"), "w", encoding="utf-8") as f:
            f.write(self.training_corpus)

        # Save conversation log
        with open(os.path.join(brain_dir, "convo.json"), "w", encoding="utf-8") as f:
            json.dump(self.conversation_log, f, ensure_ascii=False, indent=2)

        # Save vocabulary
        with open(os.path.join(brain_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab.to_dict(), f, ensure_ascii=False)

        # Save model weights. Transformer tensors use a binary checkpoint;
        # JSON is retained for metadata and backwards compatibility.
        if self.model is not None:
            if self.mode == 'quantum_transformer':
                model_data = {
                    'vocab_size': self.model.vocab_size,
                    'd_model': self.model.d_model,
                    'n_layers': self.model.n_layers,
                    'n_heads': self.model.n_heads,
                    'ff_multiplier': self.model.ff_multiplier,
                    'max_context': self.model.max_context,
                    'learning_rate': self.model.learning_rate,
                    'profile_name': self.model.profile_name,
                    'batch_size': self.model.batch_size,
                    'gradient_accumulation_steps': self.model.gradient_accumulation_steps,
                    'lora_rank': self.model.lora_rank,
                    'freeze_base': self.model.freeze_base,
                    'optimizer': self.model.optimizer_name,
                    'gradient_checkpointing': self.model.gradient_checkpointing,
                    'seed': self.model.seed,
                    'total_epochs': self.model.total_epochs,
                    'total_chars_seen': self.model.total_chars_seen,
                    'smooth_loss': self.model.smooth_loss,
                }
                with open(os.path.join(brain_dir, "model.json"), "w", encoding="utf-8") as f:
                    json.dump(model_data, f)
                torch.save(
                    {'state_dict': {key: value.detach().cpu() for key, value in self.model.state_dict().items()}},
                    os.path.join(brain_dir, "model.pt")
                )
            else:
                model_data = self.model.to_dict()
                with open(os.path.join(brain_dir, "model.json"), "w", encoding="utf-8") as f:
                    json.dump(model_data, f)

        # Save brain mode
        with open(os.path.join(brain_dir, "mode.json"), "w", encoding="utf-8") as f:
            json.dump({
                'mode': self.mode,
                'quantum_samples': self.quantum_samples,
                'dropout_rate': self.dropout_rate,
                'transformer_profile': self.transformer_profile
            }, f)

        # Save evolution state
        if self.evolution is not None:
            with open(os.path.join(brain_dir, "evolution.json"), "w", encoding="utf-8") as f:
                json.dump(self.evolution.to_dict(), f)

    @classmethod
    def load(cls, brain_dir="echo/brain"):
        """Load a brain from disk."""
        if not os.path.isdir(brain_dir):
            return None

        brain = cls()

        # Load corpus
        corpus_path = os.path.join(brain_dir, "corpus.txt")
        if os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf-8") as f:
                brain.training_corpus = f.read()

        # Load conversation log
        convo_path = os.path.join(brain_dir, "convo.json")
        if os.path.exists(convo_path):
            with open(convo_path, "r", encoding="utf-8") as f:
                brain.conversation_log = [tuple(x) for x in json.load(f)]

        # Load vocabulary
        vocab_path = os.path.join(brain_dir, "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                brain.vocab = Vocabulary.from_dict(json.load(f))

        # Load evolution state (before model)
        evo_path = os.path.join(brain_dir, "evolution.json")
        if os.path.exists(evo_path):
            with open(evo_path, "r", encoding="utf-8") as f:
                brain.evolution = EvolutionTracker.from_dict(json.load(f))

        # Load brain mode
        mode_path = os.path.join(brain_dir, "mode.json")
        if os.path.exists(mode_path):
            with open(mode_path, "r", encoding="utf-8") as f:
                md = json.load(f)
                brain.mode = md.get('mode', 'classical')
                brain.quantum_samples = md.get('quantum_samples', 8)
                brain.dropout_rate = md.get('dropout_rate', 0.0)
                brain.transformer_profile = md.get('transformer_profile', 'small')

        # Load model (using the right class based on mode)
        model_path = os.path.join(brain_dir, "model.json")
        if os.path.exists(model_path) and brain.vocab.size > 0:
            with open(model_path, "r", encoding="utf-8") as f:
                model_data = json.load(f)
                if brain.mode == 'quantum_layer':
                    brain.model = QuantumLayerRNN.from_dict(model_data)
                elif brain.mode == 'quantum':
                    brain.model = QuantumRNN.from_dict(model_data)
                elif brain.mode == 'quantum_transformer':
                    binary_path = os.path.join(brain_dir, "model.pt")
                    if os.path.exists(binary_path):
                        binary_data = torch.load(binary_path, map_location='cpu', weights_only=True)
                        model_data['state_dict'] = binary_data['state_dict']
                    brain.model = QuantumTransformerLM.from_dict(model_data)
                else:
                    brain.model = EchoRNN.from_dict(model_data)
                if brain.mode == 'quantum_layer':
                    brain.model.dropout_rate = brain.dropout_rate

        return brain

    def exists(brain_dir="echo/brain"):
        """Check if a saved brain exists."""
        return os.path.isdir(brain_dir) and os.path.exists(os.path.join(brain_dir, "model.json"))

    # --- Info ---

    def info(self):
        """Return brain status string."""
        mode_label = {'quantum_layer': 'QUANTUM LAYER', 'quantum': 'QUANTUM', 'classical': 'CLASSICAL'}
        label = mode_label.get(self.mode, self.mode.upper())
        lines = [f"=== ECHO BRAIN STATUS [{label}] ==="]
        lines.append(f"  corpus length:  {len(self.training_corpus):,} chars")
        lines.append(f"  conversation:   {len(self.conversation_log)} turns")
        lines.append(f"  vocab size:     {self.vocab.size} chars")
        if self.model:
            lines.append(self.model.info())
        else:
            lines.append("  model:          (not initialized)")
        if self.evolution:
            lines.append("  --- evolution ---")
            lines.append(self.evolution.stats())
            recent = self.evolution.recent_mutations(3)
            if recent:
                lines.append("  recent mutations:")
                for m in recent:
                    lines.append(f"    {m['detail']}")
        if self.mode in ('quantum', 'quantum_layer', 'quantum_transformer') and self.model:
            qs = self.model.quantum_stats()
            lines.append("  --- quantum ---")
            if self.mode == 'quantum_layer':
                lines.append(f"  layer entropy:   {qs['avg_entropy']:.4f} / {qs['max_entropy']:.4f} ({qs['entropy_ratio']*100:.1f}% superposition)")
                lines.append(f"  collapsed layers:{qs['collapsed_layers']} / {qs['total_layers']}")
                lines.append(f"  amplifications:  {qs['amplifications']:,}")
                lines.append(f"  re-branches:     {qs['rebranches']}")
                lines.append(f"  samples/step:    {qs['samples_per_step']}")
                lines.append(f"  dropout rate:    {qs['dropout_rate']}")
                lines.append(f"  layer alphas:    xh={qs['alpha_xh']:.3f}  hh={qs['alpha_hh']:.3f}  hy={qs['alpha_hy']:.3f}")
            else:
                lines.append(f"  avg entropy:     {qs['avg_entropy']:.4f} / {qs['max_entropy']:.4f} ({qs['entropy_ratio']*100:.1f}% superposition)")
                lines.append(f"  collapsed:       {qs['collapsed']} / {qs['total_weights']} ({qs['collapse_rate']*100:.1f}%)")
                lines.append(f"  amplifications:  {qs['amplifications']:,}")
                lines.append(f"  re-branches:     {qs['rebranches']}")
                lines.append(f"  samples/step:    {qs['samples_per_step']}")
        return "\n".join(lines)

    def show_vocab(self):
        """Display all characters in the vocabulary."""
        chars = []
        for c in self.vocab.idx_to_char:
            if c == "\n":
                chars.append("\\n")
            elif c == "\t":
                chars.append("\\t")
            elif c == " ":
                chars.append("' '")
            else:
                chars.append(c)
        return " ".join(chars)