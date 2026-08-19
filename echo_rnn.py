# ============================================================
# ECHO RNN — Character-level Recurrent Neural Network
# Built from scratch. No PyTorch. No TensorFlow. No NumPy.
# Just pure math, line by line.
# ============================================================

from echo_matrix import (
    zeros, ones, random_matrix, transpose, matmul,
    matmul_transpose_b, matmul_transpose_a,
    add, subtract, scale, hadamard,
    vec_zeros, vec_add, vec_scale, vec_dot,
    tanh, tanh_deriv, softmax, cross_entropy_loss,
    argmax, clamp, clip_gradients
)
import math
import random

class EchoRNN:
    """A character-level RNN with one hidden layer.

    Architecture:
        Input  (vocab_size)  ->  x_t
        Hidden (hidden_size) ->  h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
        Output (vocab_size)  ->  y_t = softmax(W_hy * h_t + b_y)

    Training uses Backpropagation Through Time (BPTT).
    Optimizer: SGD with gradient clipping.
    """

    def __init__(self, vocab_size, hidden_size=64, learning_rate=0.01, seed=42):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.seed = seed

        random.seed(seed)

        # --- Weight initialization (Xavier/Glorot-like) ---
        scale_xh = math.sqrt(1.0 / vocab_size)
        scale_hh = math.sqrt(1.0 / hidden_size)
        scale_hy = math.sqrt(1.0 / hidden_size)

        # Input -> Hidden weights
        self.W_xh = random_matrix(hidden_size, vocab_size, scale_xh)
        # Hidden -> Hidden weights (recurrent)
        self.W_hh = random_matrix(hidden_size, hidden_size, scale_hh)
        # Hidden -> Output weights
        self.W_hy = random_matrix(vocab_size, hidden_size, scale_hy)

        # Biases
        self.b_h = vec_zeros(hidden_size)
        self.b_y = vec_zeros(vocab_size)

        # All weight matrices for gradient clipping
        self.weight_matrices = [self.W_xh, self.W_hh, self.W_hy]

        # Training stats
        self.total_epochs = 0
        self.total_chars_seen = 0
        self.smooth_loss = None  # Exponential moving average of loss

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------

    def forward(self, inputs, h_prev=None):
        """Run forward pass over a sequence of one-hot input vectors.
        
        inputs: list of one-hot vectors (each is a list of vocab_size floats)
        h_prev: initial hidden state (list of hidden_size floats)
        
        Returns: (logits_list, hidden_states, h_final)
            logits_list: list of raw output vectors (pre-softmax) for each timestep
            hidden_states: list of hidden state vectors for each timestep (including initial)
            h_final: final hidden state
        """
        if h_prev is None:
            h_prev = vec_zeros(self.hidden_size)

        logits_list = []
        hidden_states = [h_prev]  # Include initial state for BPTT

        h_t = list(h_prev)

        for t in range(len(inputs)):
            x_t = inputs[t]

            # h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            # W_xh * x_t: (hidden_size x vocab_size) * (vocab_size x 1) = (hidden_size x 1)
            new_h = list(self.b_h)  # Start with bias

            # W_xh @ x_t
            for i in range(self.hidden_size):
                s = 0.0
                W_xh_row = self.W_xh[i]
                for j in range(self.vocab_size):
                    s += W_xh_row[j] * x_t[j]
                new_h[i] += s

            # W_hh @ h_{t-1}
            for i in range(self.hidden_size):
                s = 0.0
                W_hh_row = self.W_hh[i]
                for j in range(self.hidden_size):
                    s += W_hh_row[j] * h_t[j]
                new_h[i] += s

            # Apply tanh activation
            h_t = [tanh(v) for v in new_h]
            hidden_states.append(h_t)

            # y_t = W_hy * h_t + b_y (pre-softmax logits)
            logits = list(self.b_y)
            for i in range(self.vocab_size):
                s = 0.0
                W_hy_row = self.W_hy[i]
                for j in range(self.hidden_size):
                    s += W_hy_row[j] * h_t[j]
                logits[i] += s
            logits_list.append(logits)

        return logits_list, hidden_states, h_t

    # ----------------------------------------------------------
    # Backward pass (BPTT — Backpropagation Through Time)
    # ----------------------------------------------------------

    def backward(self, inputs, targets, logits_list, hidden_states):
        """Compute gradients via BPTT.
        
        inputs: list of one-hot input vectors
        targets: list of target indices (integers)
        logits_list: pre-softmax outputs from forward pass
        hidden_states: hidden states from forward pass (including initial)
        
        Returns: dict of gradients for each parameter
        """
        seq_len = len(inputs)

        # Initialize gradients to zero
        dW_xh = zeros(self.hidden_size, self.vocab_size)
        dW_hh = zeros(self.hidden_size, self.hidden_size)
        dW_hy = zeros(self.vocab_size, self.hidden_size)
        db_h = vec_zeros(self.hidden_size)
        db_y = vec_zeros(self.vocab_size)

        # dh_next carries gradient from future timestep to past
        dh_next = vec_zeros(self.hidden_size)

        # Pre-compute softmax probabilities and their gradients
        for t in range(seq_len):
            probs = softmax(logits_list[t])

            # Gradient of loss w.r.t. logits (softmax + cross-entropy)
            # dL/dy_i = p_i - 1 if i == target, else p_i
            dy = list(probs)
            dy[targets[t]] -= 1.0

            # Gradient for W_hy and b_y
            # dW_hy += dy * h_t^T
            h_t = hidden_states[t + 1]  # +1 because hidden_states includes initial
            for i in range(self.vocab_size):
                dyi = dy[i]
                db_y[i] += dyi
                dW_hy_row = dW_hy[i]
                for j in range(self.hidden_size):
                    dW_hy_row[j] += dyi * h_t[j]

            # Gradient flowing back to hidden state
            # dh = W_hy^T * dy + dh_next
            dh = vec_zeros(self.hidden_size)
            for i in range(self.vocab_size):
                dyi = dy[i]
                W_hy_row = self.W_hy[i]
                for j in range(self.hidden_size):
                    dh[j] += dyi * W_hy_row[j]
            dh = vec_add(dh, dh_next)

            # Backprop through tanh
            # h_t = tanh(pre_h), so dh_pre = dh * (1 - h_t^2)
            h_t = hidden_states[t + 1]
            dh_pre = [dh[i] * (1.0 - h_t[i] * h_t[i]) for i in range(self.hidden_size)]

            # Gradient for b_h
            db_h = vec_add(db_h, dh_pre)

            # Gradient for W_xh: dW_xh += dh_pre * x_t^T
            x_t = inputs[t]
            for i in range(self.hidden_size):
                dh_pre_i = dh_pre[i]
                dW_xh_row = dW_xh[i]
                for j in range(self.vocab_size):
                    dW_xh_row[j] += dh_pre_i * x_t[j]

            # Gradient for W_hh: dW_hh += dh_pre * h_{t-1}^T
            h_prev = hidden_states[t]  # h_{t-1}
            for i in range(self.hidden_size):
                dh_pre_i = dh_pre[i]
                dW_hh_row = dW_hh[i]
                for j in range(self.hidden_size):
                    dW_hh_row[j] += dh_pre_i * h_prev[j]

            # Carry gradient to previous timestep
            # dh_next = W_hh^T * dh_pre
            dh_next = vec_zeros(self.hidden_size)
            for i in range(self.hidden_size):
                dh_pre_i = dh_pre[i]
                W_hh_row = self.W_hh[i]
                for j in range(self.hidden_size):
                    dh_next[j] += dh_pre_i * W_hh_row[j]

        # Clip gradients to prevent explosion
        clip_gradients([dW_xh, dW_hh, dW_hy], threshold=5.0)

        return {
            'dW_xh': dW_xh,
            'dW_hh': dW_hh,
            'dW_hy': dW_hy,
            'db_h': db_h,
            'db_y': db_y
        }

    # ----------------------------------------------------------
    # Parameter update (SGD)
    # ----------------------------------------------------------

    def update_params(self, grads):
        """Apply gradient update using SGD."""
        lr = self.learning_rate

        # Update W_xh
        for i in range(self.hidden_size):
            dW_xh_row = grads['dW_xh'][i]
            W_xh_row = self.W_xh[i]
            for j in range(self.vocab_size):
                W_xh_row[j] -= lr * dW_xh_row[j]

        # Update W_hh
        for i in range(self.hidden_size):
            dW_hh_row = grads['dW_hh'][i]
            W_hh_row = self.W_hh[i]
            for j in range(self.hidden_size):
                W_hh_row[j] -= lr * dW_hh_row[j]

        # Update W_hy
        for i in range(self.vocab_size):
            dW_hy_row = grads['dW_hy'][i]
            W_hy_row = self.W_hy[i]
            for j in range(self.hidden_size):
                W_hy_row[j] -= lr * dW_hy_row[j]

        # Update biases
        db_h = grads['db_h']
        db_y = grads['db_y']
        for i in range(self.hidden_size):
            self.b_h[i] -= lr * db_h[i]
        for i in range(self.vocab_size):
            self.b_y[i] -= lr * db_y[i]

    # ----------------------------------------------------------
    # Training step (one sequence)
    # ----------------------------------------------------------

    def train_step(self, inputs, targets, h_prev=None):
        """Train on a single sequence.
        
        inputs: list of one-hot vectors
        targets: list of target indices
        h_prev: optional initial hidden state
        
        Returns: (loss, h_final)
        """
        # Forward
        logits_list, hidden_states, h_final = self.forward(inputs, h_prev)

        # Compute loss
        total_loss = 0.0
        for t in range(len(targets)):
            probs = softmax(logits_list[t])
            total_loss += cross_entropy_loss(probs, targets[t])
        avg_loss = total_loss / max(len(targets), 1)

        # Backward
        grads = self.backward(inputs, targets, logits_list, hidden_states)

        # Update
        self.update_params(grads)

        # Track stats
        self.total_chars_seen += len(targets)
        if self.smooth_loss is None:
            self.smooth_loss = avg_loss
        else:
            # Exponential moving average
            self.smooth_loss = 0.99 * self.smooth_loss + 0.01 * avg_loss

        return avg_loss, h_final

    # ----------------------------------------------------------
    # Sampling / Generation
    # ----------------------------------------------------------

    def sample(self, seed_input, length=100, temperature=0.8, h_prev=None):
        """Generate text by sampling from the model.
        
        seed_input: list of one-hot vectors to prime the model
        length: number of characters to generate
        temperature: controls randomness (lower = more conservative, higher = more wild)
        h_prev: optional initial hidden state
        
        Returns: list of character indices (sampled)
        """
        if h_prev is None:
            h_prev = vec_zeros(self.hidden_size)

        h_t = list(h_prev)
        sampled_indices = []

        # Prime with seed input
        current_x = None
        for x in seed_input:
            current_x = x
            # Forward one step
            new_h = list(self.b_h)
            for i in range(self.hidden_size):
                s = self.b_h[i]
                W_xh_row = self.W_xh[i]
                for j in range(self.vocab_size):
                    s += W_xh_row[j] * x[j]
                W_hh_row = self.W_hh[i]
                for j in range(self.hidden_size):
                    s += W_hh_row[j] * h_t[j]
                new_h[i] = tanh(s)
            h_t = new_h

        # If no seed, start with a zero input
        if current_x is None:
            current_x = vec_zeros(self.vocab_size)

        # Generate characters
        for _ in range(length):
            # Forward one step
            new_h = list(self.b_h)
            for i in range(self.hidden_size):
                s = self.b_h[i]
                W_xh_row = self.W_xh[i]
                for j in range(self.vocab_size):
                    s += W_xh_row[j] * current_x[j]
                W_hh_row = self.W_hh[i]
                for j in range(self.hidden_size):
                    s += W_hh_row[j] * h_t[j]
                new_h[i] = tanh(s)
            h_t = new_h

            # Compute output logits
            logits = list(self.b_y)
            for i in range(self.vocab_size):
                s = self.b_y[i]
                W_hy_row = self.W_hy[i]
                for j in range(self.hidden_size):
                    s += W_hy_row[j] * h_t[j]
                logits[i] = s

            # Apply temperature
            scaled_logits = [l / max(temperature, 0.01) for l in logits]
            probs = softmax(scaled_logits)

            # Sample from the distribution
            r = random.random()
            cumulative = 0.0
            next_idx = 0
            for i in range(len(probs)):
                cumulative += probs[i]
                if r < cumulative:
                    next_idx = i
                    break

            sampled_indices.append(next_idx)

            # Create one-hot for next input
            current_x = vec_zeros(self.vocab_size)
            current_x[next_idx] = 1.0

        return sampled_indices

    # ----------------------------------------------------------
    # Persistence — save and load as plain text
    # ----------------------------------------------------------

    def to_dict(self):
        """Serialize model state to a dictionary (JSON-safe)."""
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'seed': self.seed,
            'W_xh': self.W_xh,
            'W_hh': self.W_hh,
            'W_hy': self.W_hy,
            'b_h': self.b_h,
            'b_y': self.b_y,
            'total_epochs': self.total_epochs,
            'total_chars_seen': self.total_chars_seen,
            'smooth_loss': self.smooth_loss
        }

    @classmethod
    def from_dict(cls, d):
        """Rebuild model from serialized dictionary."""
        model = cls(
            vocab_size=d['vocab_size'],
            hidden_size=d['hidden_size'],
            learning_rate=d['learning_rate'],
            seed=d.get('seed', 42)
        )
        model.W_xh = d['W_xh']
        model.W_hh = d['W_hh']
        model.W_hy = d['W_hy']
        model.b_h = d['b_h']
        model.b_y = d['b_y']
        model.total_epochs = d.get('total_epochs', 0)
        model.total_chars_seen = d.get('total_chars_seen', 0)
        model.smooth_loss = d.get('smooth_loss', None)
        return model

    # ----------------------------------------------------------
    # Neurogenesis — grow new neurons
    # ----------------------------------------------------------

    def grow(self, n_new):
        """Add n_new new neurons to the hidden layer.
        
        Preserves existing weights. New neurons get random weights
        so they can explore new pattern spaces.
        """
        old_hidden = self.hidden_size
        new_hidden = old_hidden + n_new
        vocab = self.vocab_size

        import math as _math
        scale_xh = _math.sqrt(1.0 / vocab)
        scale_hh = _math.sqrt(1.0 / new_hidden)
        scale_hy = _math.sqrt(1.0 / new_hidden)

        # --- Expand W_xh: add n_new rows ---
        new_rows_xh = [[random.gauss(0, 1) * scale_xh for _ in range(vocab)]
                       for _ in range(n_new)]
        self.W_xh = self.W_xh + new_rows_xh

        # --- Expand W_hh: add n_new rows AND n_new columns ---
        # Add n_new columns to each existing row
        for i in range(old_hidden):
            self.W_hh[i] = self.W_hh[i] + [random.gauss(0, 1) * scale_hh for _ in range(n_new)]
        # Add n_new new rows (connecting new neurons to all neurons)
        new_rows_hh = [[random.gauss(0, 1) * scale_hh for _ in range(new_hidden)]
                       for _ in range(n_new)]
        self.W_hh = self.W_hh + new_rows_hh

        # --- Expand W_hy: add n_new columns ---
        for i in range(vocab):
            self.W_hy[i] = self.W_hy[i] + [random.gauss(0, 1) * scale_hy for _ in range(n_new)]

        # --- Expand biases ---
        self.b_h = self.b_h + [0.0] * n_new  # New neurons start with zero bias

        # Update hidden size
        self.hidden_size = new_hidden

    # ----------------------------------------------------------
    # Synaptic Pruning — remove weak neurons
    # ----------------------------------------------------------

    def prune(self, neuron_idx):
        """Remove a neuron at the given index.
        
        Removes the corresponding row from W_xh and W_hh,
        the corresponding column from W_hy and W_hh,
        and the bias entry.
        """
        if neuron_idx < 0 or neuron_idx >= self.hidden_size:
            return

        # Remove row from W_xh
        self.W_xh.pop(neuron_idx)

        # Remove row from W_hh
        self.W_hh.pop(neuron_idx)
        # Remove column from W_hh (from all remaining rows)
        for i in range(len(self.W_hh)):
            self.W_hh[i].pop(neuron_idx)

        # Remove column from W_hy (from all rows)
        for i in range(len(self.W_hy)):
            self.W_hy[i].pop(neuron_idx)

        # Remove bias entry
        self.b_h.pop(neuron_idx)

        # Update hidden size
        self.hidden_size -= 1

    # ----------------------------------------------------------
    # Info
    # ----------------------------------------------------------

    def info(self):
        """Return a string describing the model's current state."""
        loss_str = f"{self.smooth_loss:.4f}" if self.smooth_loss is not None else "N/A"
        param_count = (
            self.hidden_size * self.vocab_size +      # W_xh
            self.hidden_size * self.hidden_size +     # W_hh
            self.vocab_size * self.hidden_size +      # W_hy
            self.hidden_size +                        # b_h
            self.vocab_size                            # b_y
        )
        return (
            f"  vocab_size:     {self.vocab_size}\n"
            f"  hidden_size:    {self.hidden_size}\n"
            f"  parameters:     {param_count:,}\n"
            f"  learning_rate:  {self.learning_rate}\n"
            f"  epochs:         {self.total_epochs}\n"
            f"  chars seen:     {self.total_chars_seen:,}\n"
            f"  smooth_loss:    {loss_str}\n"
        )