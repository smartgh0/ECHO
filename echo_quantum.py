# ============================================================
# ECHO QUANTUM — Superposition-weighted RNN
# Each weight exists in 2 states simultaneously:
#   W = alpha|w1> + beta|w2>   where |alpha|^2 + |beta|^2 = 1
#
# Forward pass samples K configurations from the superposition.
# Gradient updates amplify branches that produce lower loss.
# Collapsed weights re-branch to maintain exploration.
#
# No PyTorch. No TensorFlow. No NumPy. No Qiskit.
# Just pure math, quantum-inspired, running on classical hardware.
# ============================================================

import math
import random
from echo_matrix import (
    zeros, random_matrix, vec_zeros, vec_add,
    tanh, softmax, cross_entropy_loss, clip_gradients
)

# ----------------------------------------------------------
# QuantumWeight — a single weight in superposition
# ----------------------------------------------------------

class QuantumWeight:
    """A single weight in quantum superposition of two values.

    State: (alpha, w1, beta, w2)
    where |alpha|^2 + |beta|^2 = 1
    The weight is w1 with probability |alpha|^2, w2 with probability |beta|^2.

    Operations:
        sample() → (value, branch_index)
        amplify(branch) → shift probability toward that branch
        collapse_check() → re-branch if one branch dominates > 97.5%
        expected_value() → probability-weighted average (for generation)
    """

    __slots__ = ['alpha', 'w1', 'beta', 'w2', 'collapses', 'amplifications']

    def __init__(self, w1, w2, alpha=None, beta=None):
        if alpha is None:
            alpha = math.sqrt(0.5)
            beta = math.sqrt(0.5)
        self.alpha = alpha  # amplitude for branch 0
        self.w1 = w1        # weight value for branch 0
        self.beta = beta    # amplitude for branch 1
        self.w2 = w2        # weight value for branch 1
        self.collapses = 0
        self.amplifications = 0

    def sample(self):
        """Collapse the superposition to one value (Monte Carlo).
        Returns (value, branch_index)."""
        if random.random() < self.alpha * self.alpha:
            return self.w1, 0
        else:
            return self.w2, 1

    def amplify(self, branch, rate=0.001):
        """Shift probability toward the given branch.
        This is the quantum measurement update — branches that produce
        lower loss get amplified, others get diminished."""
        if branch == 0:
            self.alpha = min(0.99, self.alpha + rate)
            self.beta = math.sqrt(max(0.0001, 1.0 - self.alpha * self.alpha))
        else:
            self.beta = min(0.99, self.beta + rate)
            self.alpha = math.sqrt(max(0.0001, 1.0 - self.beta * self.beta))
        self.amplifications += 1

    def update_value(self, branch, gradient, lr):
        """Apply gradient update to the specified branch's weight value."""
        if branch == 0:
            self.w1 -= lr * gradient
        else:
            self.w2 -= lr * gradient

    def check_collapse(self, rebranch_scale=0.05):
        """Check if this weight has collapsed (>97.5% on one branch).
        If so, re-branch: create a new superposition around the dominant value.
        Returns True if re-branching occurred."""
        if self.alpha > 0.975:
            # Branch 0 dominant — re-branch around w1
            self.alpha = math.sqrt(0.6)
            self.beta = math.sqrt(0.4)
            self.w2 = self.w1 + random.gauss(0, rebranch_scale)
            self.collapses += 1
            return True
        elif self.beta > 0.975:
            # Branch 1 dominant — re-branch around w2
            self.beta = math.sqrt(0.6)
            self.alpha = math.sqrt(0.4)
            self.w1 = self.w2 + random.gauss(0, rebranch_scale)
            self.collapses += 1
            return True
        return False

    def expected_value(self):
        """Return the probability-weighted expected value of this weight.
        Used for deterministic generation (no sampling)."""
        return self.alpha * self.alpha * self.w1 + self.beta * self.beta * self.w2

    def entropy(self):
        """Shannon entropy of the superposition.
        0.693 = maximum (perfect 50/50 superposition)
        0.000 = fully collapsed (classical behavior)"""
        p = self.alpha * self.alpha
        q = self.beta * self.beta
        if p < 1e-10: p = 1e-10
        if q < 1e-10: q = 1e-10
        return -p * math.log(p) - q * math.log(q)

    def to_list(self):
        """Serialize for JSON persistence."""
        return [self.alpha, self.w1, self.beta, self.w2]

    @classmethod
    def from_list(cls, data):
        qw = cls(data[1], data[3], data[0], data[2])
        return qw

    def __repr__(self):
        return f"QW(α={self.alpha:.3f}|{self.w1:+.4f}> + β={self.beta:.3f}|{self.w2:+.4f}>)"


# ----------------------------------------------------------
# QuantumRNN — Superposition-weighted character-level RNN
# ----------------------------------------------------------

class QuantumRNN:
    """A character-level RNN where every weight is in quantum superposition.

    Architecture (same as EchoRNN):
        Input  (vocab_size)  ->  x_t
        Hidden (hidden_size) ->  h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
        Output (vocab_size)  ->  y_t = softmax(W_hy * h_t + b_y)

    But each weight W[i][j] is a QuantumWeight:
        W[i][j] = alpha|w1> + beta|w2>

    Training:
        1. Sample K configurations from the superposition
        2. Run forward pass on each configuration
        3. Compute loss for each
        4. Amplify branches from the best configuration
        5. Apply gradient updates to the best configuration's branches
        6. Check for collapse and re-branch

    This is Monte Carlo quantum simulation on classical hardware.
    """

    def __init__(self, vocab_size, hidden_size=24, learning_rate=0.02,
                 n_samples=8, seed=42):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_samples = n_samples  # K: Monte Carlo samples per step
        self.seed = seed

        random.seed(seed)

        sx = math.sqrt(1.0 / vocab_size)
        sh = math.sqrt(1.0 / hidden_size)

        # Initialize all weights as quantum superpositions
        # Start at 50/50 (max entropy = max exploration)
        self.Q_xh = [[QuantumWeight(random.gauss(0,1)*sx, random.gauss(0,1)*sx)
                      for _ in range(vocab_size)] for _ in range(hidden_size)]
        self.Q_hh = [[QuantumWeight(random.gauss(0,1)*sh, random.gauss(0,1)*sh)
                      for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.Q_hy = [[QuantumWeight(random.gauss(0,1)*sh, random.gauss(0,1)*sh)
                      for _ in range(hidden_size)] for _ in range(vocab_size)]

        self.Q_bh = [QuantumWeight(random.gauss(0,0.01), random.gauss(0,0.01))
                     for _ in range(hidden_size)]
        self.Q_by = [QuantumWeight(random.gauss(0,0.01), random.gauss(0,0.01))
                     for _ in range(vocab_size)]

        # Training stats
        self.total_epochs = 0
        self.total_chars_seen = 0
        self.smooth_loss = None
        self.total_amplifications = 0
        self.total_collapses = 0
        self.total_rebranches = 0

    # ----------------------------------------------------------
    # Configuration sampling
    # ----------------------------------------------------------

    def _sample_config(self):
        """Sample one complete classical configuration from all quantum weights.
        Returns a dict of plain (classical) weight matrices + branch tracking."""
        # Sample W_xh
        W_xh = zeros(self.hidden_size, self.vocab_size)
        branches_xh = zeros(self.hidden_size, self.vocab_size)
        for i in range(self.hidden_size):
            for j in range(self.vocab_size):
                val, br = self.Q_xh[i][j].sample()
                W_xh[i][j] = val
                branches_xh[i][j] = br

        # Sample W_hh
        W_hh = zeros(self.hidden_size, self.hidden_size)
        branches_hh = zeros(self.hidden_size, self.hidden_size)
        for i in range(self.hidden_size):
            for j in range(self.hidden_size):
                val, br = self.Q_hh[i][j].sample()
                W_hh[i][j] = val
                branches_hh[i][j] = br

        # Sample W_hy
        W_hy = zeros(self.vocab_size, self.hidden_size)
        branches_hy = zeros(self.vocab_size, self.hidden_size)
        for i in range(self.vocab_size):
            for j in range(self.hidden_size):
                val, br = self.Q_hy[i][j].sample()
                W_hy[i][j] = val
                branches_hy[i][j] = br

        # Sample biases
        b_h = [0.0] * self.hidden_size
        branches_bh = [0] * self.hidden_size
        for i in range(self.hidden_size):
            val, br = self.Q_bh[i].sample()
            b_h[i] = val
            branches_bh[i] = br

        b_y = [0.0] * self.vocab_size
        branches_by = [0] * self.vocab_size
        for i in range(self.vocab_size):
            val, br = self.Q_by[i].sample()
            b_y[i] = val
            branches_by[i] = br

        return {
            'W_xh': W_xh, 'W_hh': W_hh, 'W_hy': W_hy,
            'b_h': b_h, 'b_y': b_y,
            'br_xh': branches_xh, 'br_hh': branches_hh, 'br_hy': branches_hy,
            'br_bh': branches_bh, 'br_by': branches_by
        }

    # ----------------------------------------------------------
    # Forward pass (with a specific classical config)
    # ----------------------------------------------------------

    def _forward_with_config(self, inputs, config, h_prev=None):
        """Run forward pass using a sampled classical configuration."""
        if h_prev is None:
            h_prev = [0.0] * self.hidden_size

        W_xh = config['W_xh']
        W_hh = config['W_hh']
        W_hy = config['W_hy']
        b_h = config['b_h']
        b_y = config['b_y']

        h_t = list(h_prev)
        logits_list = []
        hidden_states = [list(h_t)]

        for t in range(len(inputs)):
            x_t = inputs[t]
            new_h = list(b_h)

            for i in range(self.hidden_size):
                s = b_h[i]
                W_xh_row = W_xh[i]
                for j in range(self.vocab_size):
                    s += W_xh_row[j] * x_t[j]
                W_hh_row = W_hh[i]
                for j in range(self.hidden_size):
                    s += W_hh_row[j] * h_t[j]
                new_h[i] = tanh(s)

            h_t = new_h
            hidden_states.append(list(h_t))

            logits = list(b_y)
            for i in range(self.vocab_size):
                s = b_y[i]
                W_hy_row = W_hy[i]
                for j in range(self.hidden_size):
                    s += W_hy_row[j] * h_t[j]
                logits[i] = s
            logits_list.append(logits)

        return logits_list, hidden_states, h_t

    # ----------------------------------------------------------
    # Backward pass (BPTT) on best config
    # ----------------------------------------------------------

    def _backward_with_config(self, inputs, targets, logits_list, hidden_states, config):
        """Compute gradients via BPTT for a specific configuration."""
        seq_len = len(inputs)
        H = self.hidden_size
        V = self.vocab_size

        dW_xh = zeros(H, V)
        dW_hh = zeros(H, H)
        dW_hy = zeros(V, H)
        db_h = [0.0] * H
        db_y = [0.0] * V
        dh_next = [0.0] * H

        for t in range(seq_len - 1, -1, -1):
            probs = softmax(logits_list[t])
            dy = list(probs)
            dy[targets[t]] -= 1.0

            h_t = hidden_states[t + 1]

            for i in range(V):
                dyi = dy[i]
                db_y[i] += dyi
                dW_hy_row = dW_hy[i]
                for j in range(H):
                    dW_hy_row[j] += dyi * h_t[j]

            # Backprop to hidden
            dh = [0.0] * H
            for i in range(V):
                dyi = dy[i]
                W_hy_row = config['W_hy'][i]
                for j in range(H):
                    dh[j] += dyi * W_hy_row[j]
            dh = vec_add(dh, dh_next)

            # Through tanh
            dh_pre = [dh[i] * (1.0 - h_t[i] * h_t[i]) for i in range(H)]

            # Accumulate gradients for input, recurrent, and hidden bias.
            for i in range(H):
                dh_pre_i = dh_pre[i]
                db_h[i] += dh_pre_i
                for j in range(V):
                    dW_xh[i][j] += dh_pre_i * inputs[t][j]
                for j in range(H):
                    dW_hh[i][j] += dh_pre_i * hidden_states[t][j]

            # Carry gradient to the previous timestep.
            W_hh = config['W_hh']
            dh_next = [0.0] * H
            for i in range(H):
                dh_pre_i = dh_pre[i]
                for j in range(H):
                    dh_next[j] += dh_pre_i * W_hh[i][j]

        clip_gradients([dW_xh, dW_hh, dW_hy], threshold=5.0)

        return {
            'dW_xh': dW_xh, 'dW_hh': dW_hh, 'dW_hy': dW_hy,
            'db_h': db_h, 'db_y': db_y
        }

    def _update_config(self, config, grads):
        """Apply one accumulated BPTT update to the sampled branches."""
        for i in range(self.hidden_size):
            for j in range(self.vocab_size):
                self.Q_xh[i][j].update_value(
                    int(config['br_xh'][i][j]), grads['dW_xh'][i][j], self.learning_rate)
            for j in range(self.hidden_size):
                self.Q_hh[i][j].update_value(
                    int(config['br_hh'][i][j]), grads['dW_hh'][i][j], self.learning_rate)
            self.Q_bh[i].update_value(
                int(config['br_bh'][i]), grads['db_h'][i], self.learning_rate)

        for i in range(self.vocab_size):
            for j in range(self.hidden_size):
                self.Q_hy[i][j].update_value(
                    int(config['br_hy'][i][j]), grads['dW_hy'][i][j], self.learning_rate)
            self.Q_by[i].update_value(
                int(config['br_by'][i]), grads['db_y'][i], self.learning_rate)
    # ----------------------------------------------------------
    # Training step — the quantum engine
    # ----------------------------------------------------------

    def train_step(self, inputs, targets, h_prev=None):
        """Quantum training step.

        1. Sample K configurations from the superposition
        2. Run forward pass on each
        3. Find the best (lowest loss)
        4. Amplify the best config's branches
        5. Apply gradient updates to the best config
        6. Check for collapses and re-branch

        Returns: (average_loss, h_final)
        """
        K = self.n_samples
        results = []

        for _ in range(K):
            config = self._sample_config()
            logits_list, hidden_states, h_final = self._forward_with_config(
                inputs, config, h_prev)

            loss = 0.0
            for t in range(len(targets)):
                probs = softmax(logits_list[t])
                loss += cross_entropy_loss(probs, targets[t])
            loss /= max(len(targets), 1)

            results.append((config, loss, logits_list, hidden_states))

        # Sort by loss (best first)
        results.sort(key=lambda r: r[1])

        best_config, best_loss, best_logits, best_hidden = results[0]

        # --- AMPLIFY: shift probability toward best config's branches ---
        amp_rate = self.learning_rate * 0.1

        # Amplify W_hy (most direct impact on loss)
        for i in range(self.vocab_size):
            for j in range(self.hidden_size):
                br = int(best_config['br_hy'][i][j])
                self.Q_hy[i][j].amplify(br, amp_rate)
                self.total_amplifications += 1

        # Amplify W_xh
        for i in range(self.hidden_size):
            for j in range(self.vocab_size):
                br = int(best_config['br_xh'][i][j])
                self.Q_xh[i][j].amplify(br, amp_rate * 0.5)  # Less aggressive
                self.total_amplifications += 1

        # Amplify W_hh
        for i in range(self.hidden_size):
            for j in range(self.hidden_size):
                br = int(best_config['br_hh'][i][j])
                self.Q_hh[i][j].amplify(br, amp_rate * 0.5)
                self.total_amplifications += 1

        # Amplify biases
        for i in range(self.hidden_size):
            br = int(best_config['br_bh'][i])
            self.Q_bh[i].amplify(br, amp_rate)
        for i in range(self.vocab_size):
            br = int(best_config['br_by'][i])
            self.Q_by[i].amplify(br, amp_rate)

        # --- GRADIENT UPDATE on best config ---
        grads = self._backward_with_config(
            inputs, targets, best_logits, best_hidden, best_config)
        self._update_config(best_config, grads)

        # --- CHECK COLLAPSES and re-branch ---
        all_weights = (
            [qw for row in self.Q_xh for qw in row] +
            [qw for row in self.Q_hh for qw in row] +
            [qw for row in self.Q_hy for qw in row] +
            self.Q_bh + self.Q_by
        )
        for qw in all_weights:
            if qw.check_collapse():
                self.total_rebranches += 1

        # Count collapsed weights
        self.total_collapses = sum(
            1 for qw in all_weights if qw.alpha > 0.975 or qw.beta > 0.975
        )

        # Track stats
        self.total_epochs += 1
        self.total_chars_seen += len(targets)
        if self.smooth_loss is None:
            self.smooth_loss = best_loss
        else:
            self.smooth_loss = 0.99 * self.smooth_loss + 0.01 * best_loss

        return best_loss, h_final

    # ----------------------------------------------------------
    # Generation — uses expected (probability-weighted) weights
    # ----------------------------------------------------------

    def _get_expected_weights(self):
        """Extract the expected (classical) weight matrices from superposition.
        This is what the brain 'looks like' when you measure it."""
        W_xh = [[self.Q_xh[i][j].expected_value()
                 for j in range(self.vocab_size)] for i in range(self.hidden_size)]
        W_hh = [[self.Q_hh[i][j].expected_value()
                 for j in range(self.hidden_size)] for i in range(self.hidden_size)]
        W_hy = [[self.Q_hy[i][j].expected_value()
                 for j in range(self.hidden_size)] for i in range(self.vocab_size)]
        b_h = [self.Q_bh[i].expected_value() for i in range(self.hidden_size)]
        b_y = [self.Q_by[i].expected_value() for i in range(self.vocab_size)]
        return W_xh, W_hh, W_hy, b_h, b_y

    def sample(self, seed_input, length=100, temperature=0.5, h_prev=None):
        """Generate text using expected (measured) weights."""
        W_xh, W_hh, W_hy, b_h, b_y = self._get_expected_weights()

        if h_prev is None:
            h_prev = [0.0] * self.hidden_size

        h_t = list(h_prev)
        current_x = None
        sampled_indices = []

        # Prime with seed
        for x in seed_input:
            current_x = x
            new_h = list(b_h)
            for i in range(self.hidden_size):
                s = b_h[i]
                for j in range(self.vocab_size):
                    s += W_xh[i][j] * x[j]
                for j in range(self.hidden_size):
                    s += W_hh[i][j] * h_t[j]
                new_h[i] = tanh(s)
            h_t = new_h

        if current_x is None:
            current_x = [0.0] * self.vocab_size

        # Generate
        for _ in range(length):
            new_h = list(b_h)
            for i in range(self.hidden_size):
                s = b_h[i]
                for j in range(self.vocab_size):
                    s += W_xh[i][j] * current_x[j]
                for j in range(self.hidden_size):
                    s += W_hh[i][j] * h_t[j]
                new_h[i] = tanh(s)
            h_t = new_h

            logits = list(b_y)
            for i in range(self.vocab_size):
                s = b_y[i]
                for j in range(self.hidden_size):
                    s += W_hy[i][j] * h_t[j]
                logits[i] = s

            scaled = [l / max(temperature, 0.01) for l in logits]
            probs = softmax(scaled)

            r = random.random()
            cumulative = 0.0
            next_idx = 0
            for i in range(len(probs)):
                cumulative += probs[i]
                if r < cumulative:
                    next_idx = i
                    break

            sampled_indices.append(next_idx)
            current_x = [0.0] * self.vocab_size
            current_x[next_idx] = 1.0

        return sampled_indices

    # ----------------------------------------------------------
    # Quantum statistics
    # ----------------------------------------------------------

    def quantum_stats(self):
        """Return quantum-specific statistics."""
        all_weights = (
            [qw for row in self.Q_xh for qw in row] +
            [qw for row in self.Q_hh for qw in row] +
            [qw for row in self.Q_hy for qw in row] +
            self.Q_bh + self.Q_by
        )
        total = len(all_weights)
        if total == 0:
            return {'entropy': 0, 'collapsed': 0, 'collapse_rate': 0}

        total_entropy = sum(qw.entropy() for qw in all_weights)
        collapsed = sum(1 for qw in all_weights
                       if qw.alpha > 0.975 or qw.beta > 0.975)
        avg_alpha = sum(qw.alpha for qw in all_weights) / total

        return {
            'avg_entropy': total_entropy / total,
            'max_entropy': 0.693,  # ln(2)
            'entropy_ratio': (total_entropy / total) / 0.693,
            'collapsed': collapsed,
            'total_weights': total,
            'collapse_rate': collapsed / total,
            'avg_alpha': avg_alpha,
            'amplifications': self.total_amplifications,
            'rebranches': self.total_rebranches,
            'samples_per_step': self.n_samples
        }

    # ----------------------------------------------------------
    # Neurogenesis (grow) — same as EchoRNN but with quantum weights
    # ----------------------------------------------------------

    def grow(self, n_new):
        """Add n_new new neurons to the hidden layer."""
        old_hidden = self.hidden_size
        new_hidden = old_hidden + n_new
        V = self.vocab_size

        sx = math.sqrt(1.0 / V)
        sh = math.sqrt(1.0 / new_hidden)

        # New rows for Q_xh
        new_Q_xh_rows = [[QuantumWeight(random.gauss(0,1)*sx, random.gauss(0,1)*sx)
                          for _ in range(V)] for _ in range(n_new)]
        self.Q_xh = self.Q_xh + new_Q_xh_rows

        # Expand Q_hh: add rows AND columns
        for i in range(old_hidden):
            for _ in range(n_new):
                self.Q_hh[i].append(
                    QuantumWeight(random.gauss(0,1)*sh, random.gauss(0,1)*sh))
        new_Q_hh_rows = [[QuantumWeight(random.gauss(0,1)*sh, random.gauss(0,1)*sh)
                          for _ in range(new_hidden)] for _ in range(n_new)]
        self.Q_hh = self.Q_hh + new_Q_hh_rows

        # Expand Q_hy: add columns
        for i in range(V):
            for _ in range(n_new):
                self.Q_hy[i].append(
                    QuantumWeight(random.gauss(0,1)*sh, random.gauss(0,1)*sh))

        # Expand biases
        for _ in range(n_new):
            self.Q_bh.append(QuantumWeight(random.gauss(0,0.01), random.gauss(0,0.01)))

        self.hidden_size = new_hidden

    # NOTE: No prune() method. We don't prune. Ever.

    # ----------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------

    def to_dict(self):
        """Serialize quantum model state."""
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'n_samples': self.n_samples,
            'seed': self.seed,
            'Q_xh': [[qw.to_list() for qw in row] for row in self.Q_xh],
            'Q_hh': [[qw.to_list() for qw in row] for row in self.Q_hh],
            'Q_hy': [[qw.to_list() for qw in row] for row in self.Q_hy],
            'Q_bh': [qw.to_list() for qw in self.Q_bh],
            'Q_by': [qw.to_list() for qw in self.Q_by],
            'total_epochs': self.total_epochs,
            'total_chars_seen': self.total_chars_seen,
            'smooth_loss': self.smooth_loss,
            'total_amplifications': self.total_amplifications,
            'total_collapses': self.total_collapses,
            'total_rebranches': self.total_rebranches
        }

    @classmethod
    def from_dict(cls, d):
        """Rebuild quantum model from serialized state."""
        model = cls(
            vocab_size=d['vocab_size'],
            hidden_size=d['hidden_size'],
            learning_rate=d['learning_rate'],
            n_samples=d.get('n_samples', 8),
            seed=d.get('seed', 42)
        )
        model.Q_xh = [[QuantumWeight.from_list(qw) for qw in row]
                      for row in d['Q_xh']]
        model.Q_hh = [[QuantumWeight.from_list(qw) for qw in row]
                      for row in d['Q_hh']]
        model.Q_hy = [[QuantumWeight.from_list(qw) for qw in row]
                      for row in d['Q_hy']]
        model.Q_bh = [QuantumWeight.from_list(qw) for qw in d['Q_bh']]
        model.Q_by = [QuantumWeight.from_list(qw) for qw in d['Q_by']]
        model.total_epochs = d.get('total_epochs', 0)
        model.total_chars_seen = d.get('total_chars_seen', 0)
        model.smooth_loss = d.get('smooth_loss', None)
        model.total_amplifications = d.get('total_amplifications', 0)
        model.total_collapses = d.get('total_collapses', 0)
        model.total_rebranches = d.get('total_rebranches', 0)
        return model

    # ----------------------------------------------------------
    # Info
    # ----------------------------------------------------------

    def info(self):
        """Return model status string."""
        loss_str = f"{self.smooth_loss:.4f}" if self.smooth_loss is not None else "N/A"
        # Count total quantum weights
        n_w = (self.hidden_size * self.vocab_size +
               self.hidden_size * self.hidden_size +
               self.vocab_size * self.hidden_size +
               self.hidden_size + self.vocab_size)
        return (
            f"  [QUANTUM RNN]\n"
            f"  vocab_size:     {self.vocab_size}\n"
            f"  hidden_size:    {self.hidden_size}\n"
            f"  quantum weights:{n_w:,} (each in 2-state superposition)\n"
            f"  configurations: 2^{n_w} theoretical | {self.n_samples} sampled/step\n"
            f"  learning_rate:  {self.learning_rate}\n"
            f"  epochs:         {self.total_epochs}\n"
            f"  chars seen:     {self.total_chars_seen:,}\n"
            f"  smooth_loss:    {loss_str}\n"
            f"  amplifications: {self.total_amplifications:,}\n"
            f"  re-branches:    {self.total_rebranches}\n"
        )