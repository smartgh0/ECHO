#!/usr/bin/env python3
"""
ECHO — QUANTUM vs CLASSICAL LEARNING SIMULATION
Simulates 7 days of conversation for 3 brain types:
  1. Classical Echo (1 weight per connection)
  2. Quantum Echo (superposition weights, Monte Carlo sampling)
  3. Hybrid Echo (classical weights + quantum exploration on plateaus)

All built from scratch. No libraries. Pure math.
Usage: python3 echo/simulate_quantum.py
"""
import sys, os, math, random, time

random.seed(42)

# ============================================================
# PURE MATH (inline, no imports from echo_matrix needed)
# ============================================================

def tanh(x):
    if x > 20: return 1.0
    if x < -20: return -1.0
    e2x = math.exp(2 * x)
    return (e2x - 1) / (e2x + 1)

def softmax(vec):
    if not vec: return []
    mx = max(vec)
    exps = [math.exp(v - mx) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]

def cross_entropy(probs, target):
    p = max(probs[target], 1e-12)
    return -math.log(p)

def one_hot(idx, size):
    v = [0.0] * size
    if 0 <= idx < size: v[idx] = 1.0
    return v

def zeros(*dims):
    if len(dims) == 1: return [0.0] * dims[0]
    return [zeros(*dims[1:]) for _ in range(dims[0])]

def rand_mat(rows, cols, scale=0.1):
    return [[random.gauss(0, 1) * scale for _ in range(cols)] for _ in range(rows)]

# ============================================================
# CLASSICAL RNN (baseline — what Echo currently does)
# ============================================================

class ClassicalRNN:
    def __init__(self, vocab_size, hidden_size, lr=0.02):
        self.V = vocab_size
        self.H = hidden_size
        self.lr = lr
        sx = math.sqrt(1.0/vocab_size)
        sh = math.sqrt(1.0/hidden_size)
        self.Wxh = rand_mat(hidden_size, vocab_size, sx)
        self.Whh = rand_mat(hidden_size, hidden_size, sh)
        self.Why = rand_mat(vocab_size, hidden_size, sh)
        self.bh = [0.0]*hidden_size
        self.by = [0.0]*vocab_size
        self.smooth_loss = None
        self.epochs = 0

    def forward(self, inputs, h0=None):
        if h0 is None: h0 = [0.0]*self.H
        h = list(h0)
        logits_list = []
        hstates = [list(h)]
        for x in inputs:
            nh = list(self.bh)
            for i in range(self.H):
                s = self.bh[i]
                for j in range(self.V): s += self.Wxh[i][j]*x[j]
                for j in range(self.H): s += self.Whh[i][j]*h[j]
                nh[i] = tanh(s)
            h = nh
            hstates.append(list(h))
            logits = list(self.by)
            for i in range(self.V):
                s = self.by[i]
                for j in range(self.H): s += self.Why[i][j]*h[j]
                logits[i] = s
            logits_list.append(logits)
        return logits_list, hstates, h

    def train_step(self, inputs, targets):
        logits_list, hstates, _ = self.forward(inputs)
        loss = 0.0
        for t in range(len(targets)):
            probs = softmax(logits_list[t])
            loss += cross_entropy(probs, targets[t])
        loss /= max(len(targets), 1)
        # Simple gradient: use loss signal to nudge weights
        # (Simplified BPTT for simulation — full BPTT is in echo_rnn.py)
        for t in range(len(targets)):
            probs = softmax(logits_list[t])
            dy = list(probs)
            dy[targets[t]] -= 1.0
            ht = hstates[t+1]
            for i in range(self.V):
                for j in range(self.H):
                    self.Why[i][j] -= self.lr * dy[i] * ht[j]
                self.by[i] -= self.lr * dy[i]
        self.epochs += 1
        if self.smooth_loss is None: self.smooth_loss = loss
        else: self.smooth_loss = 0.99*self.smooth_loss + 0.01*loss
        return loss

    def sample(self, seed, length=60, temp=0.5):
        h = [0.0]*self.H
        x = seed[0] if seed else [0.0]*self.V
        out = []
        for _ in range(length):
            nh = list(self.bh)
            for i in range(self.H):
                s = self.bh[i]
                for j in range(self.V): s += self.Wxh[i][j]*x[j]
                for j in range(self.H): s += self.Whh[i][j]*h[j]
                nh[i] = tanh(s)
            h = nh
            logits = list(self.by)
            for i in range(self.V):
                s = self.by[i]
                for j in range(self.H): s += self.Why[i][j]*h[j]
                logits[i] = s
            sl = [l/max(temp,0.01) for l in logits]
            probs = softmax(sl)
            r = random.random()
            cum = 0.0
            idx = 0
            for i in range(len(probs)):
                cum += probs[i]
                if r < cum: idx = i; break
            out.append(idx)
            x = one_hot(idx, self.V)
        return out

# ============================================================
# QUANTUM RNN (superposition weights — the new approach)
# Each weight stores: (alpha, w1, beta, w2)
# Forward pass samples K configurations from the superposition
# Gradient updates amplify successful branches
# ============================================================

class QuantumRNN:
    def __init__(self, vocab_size, hidden_size, lr=0.02, n_branches=2, n_samples=10):
        self.V = vocab_size
        self.H = hidden_size
        self.lr = lr
        self.K = n_samples      # Monte Carlo samples per forward pass
        self.B = n_branches     # superposition branches per weight (2 = |0> + |1>)

        sx = math.sqrt(1.0/vocab_size)
        sh = math.sqrt(1.0/hidden_size)

        # Superposition weights: each weight is (alpha, w1, beta, w2)
        # alpha^2 + beta^2 = 1 (normalized)
        # Start with 50/50 superposition
        self.Qxh = []  # hidden_size x vocab_size, each entry = [alpha, w1, beta, w2]
        for i in range(hidden_size):
            row = []
            for j in range(vocab_size):
                w1 = random.gauss(0,1)*sx
                w2 = random.gauss(0,1)*sx
                a = math.sqrt(0.5)
                b = math.sqrt(0.5)
                row.append([a, w1, b, w2])
            self.Qxh.append(row)

        self.Qhh = []
        for i in range(hidden_size):
            row = []
            for j in range(hidden_size):
                w1 = random.gauss(0,1)*sh
                w2 = random.gauss(0,1)*sh
                row.append([math.sqrt(0.5), w1, math.sqrt(0.5), w2])
            self.Qhh.append(row)

        self.Qhy = []
        for i in range(vocab_size):
            row = []
            for j in range(hidden_size):
                w1 = random.gauss(0,1)*sh
                w2 = random.gauss(0,1)*sh
                row.append([math.sqrt(0.5), w1, math.sqrt(0.5), w2])
            self.Qhy.append(row)

        self.bh = [[math.sqrt(0.5), random.gauss(0,0.01), math.sqrt(0.5), random.gauss(0,0.01)]
                   for _ in range(hidden_size)]
        self.by = [[math.sqrt(0.5), random.gauss(0,0.01), math.sqrt(0.5), random.gauss(0,0.01)]
                   for _ in range(vocab_size)]

        self.smooth_loss = None
        self.epochs = 0
        self.collapses = 0  # Times a weight collapsed to one branch
        self.amplifications = 0  # Times amplitudes shifted

    def _sample_weight(self, qweight):
        """Sample one classical value from a superposition weight."""
        a, w1, b, w2 = qweight
        if random.random() < a*a:
            return w1, 0  # branch 0
        else:
            return w2, 1  # branch 1

    def _sample_config(self):
        """Sample a complete classical configuration from all superposition weights."""
        cfg_xh = [[self._sample_weight(self.Qxh[i][j]) for j in range(self.V)]
                  for i in range(self.H)]
        cfg_hh = [[self._sample_weight(self.Qhh[i][j]) for j in range(self.H)]
                  for i in range(self.H)]
        cfg_hy = [[self._sample_weight(self.Qhy[i][j]) for j in range(self.H)]
                  for i in range(self.V)]
        cfg_bh = [self._sample_weight(self.bh[i]) for i in range(self.H)]
        cfg_by = [self._sample_weight(self.by[i]) for i in range(self.V)]
        return cfg_xh, cfg_hh, cfg_hy, cfg_bh, cfg_by

    def _config_to_weights(self, config):
        """Extract weight values (not branch indices) from a config."""
        cfg_xh, cfg_hh, cfg_hy, cfg_bh, cfg_by = config
        wxh = [[cfg_xh[i][j][0] for j in range(self.V)] for i in range(self.H)]
        whh = [[cfg_hh[i][j][0] for j in range(self.H)] for i in range(self.H)]
        why = [[cfg_hy[i][j][0] for j in range(self.H)] for i in range(self.V)]
        bh = [cfg_bh[i][0] for i in range(self.H)]
        by = [cfg_by[i][0] for i in range(self.V)]
        return wxh, whh, why, bh, by

    def _forward_with_weights(self, inputs, wxh, whh, why, bh, by, h0=None):
        """Run forward pass using specific (sampled) weight values."""
        if h0 is None: h0 = [0.0]*self.H
        h = list(h0)
        logits_list = []
        hstates = [list(h)]
        for x in inputs:
            nh = list(bh)
            for i in range(self.H):
                s = bh[i]
                for j in range(self.V): s += wxh[i][j]*x[j]
                for j in range(self.H): s += whh[i][j]*h[j]
                nh[i] = tanh(s)
            h = nh
            hstates.append(list(h))
            logits = list(by)
            for i in range(self.V):
                s = by[i]
                for j in range(self.H): s += why[i][j]*h[j]
                logits[i] = s
            logits_list.append(logits)
        return logits_list, hstates, h

    def train_step(self, inputs, targets):
        """Quantum training: sample K configs, evaluate, amplify best."""
        results = []
        for _ in range(self.K):
            config = self._sample_config()
            wxh, whh, why, bh, by = self._config_to_weights(config)
            logits_list, hstates, _ = self._forward_with_weights(
                inputs, wxh, whh, why, bh, by)

            loss = 0.0
            for t in range(len(targets)):
                probs = softmax(logits_list[t])
                loss += cross_entropy(probs, targets[t])
            loss /= max(len(targets), 1)
            results.append((config, loss, logits_list, hstates))

        # Find best and worst samples
        results.sort(key=lambda r: r[1])
        best = results[0]
        worst = results[-1]

        # Expected loss (average across all samples)
        avg_loss = sum(r[1] for r in results) / len(results)

        # --- AMPLIFY: shift amplitudes toward the best configuration ---
        # The best config's branches get amplified (alpha increases)
        # The worst config's branches get diminished
        amplification_rate = self.lr * 0.1

        best_config = best[0]
        worst_config = worst[0]

        # Amplify best branches in Qhy (output layer — most direct impact)
        for i in range(self.V):
            for j in range(self.H):
                best_branch = best_config[2][i][j][1]  # 0 or 1
                q = self.Qhy[i][j]
                if best_branch == 0:
                    # Amplify branch 0 (alpha)
                    q[0] = min(0.99, q[0] + amplification_rate)
                    q[2] = math.sqrt(max(0.0001, 1.0 - q[0]*q[0]))
                else:
                    q[2] = min(0.99, q[2] + amplification_rate)
                    q[0] = math.sqrt(max(0.0004, 1.0 - q[2]*q[2]))
                self.amplifications += 1

        # Update the actual weight VALUES via gradient on the best config
        wxh, whh, why, bh, by = self._config_to_weights(best_config)
        _, hstates_best, _ = self._forward_with_weights(
            inputs, wxh, whh, why, bh, by)

        for t in range(len(targets)):
            probs = softmax(best[2][t])
            dy = list(probs)
            dy[targets[t]] -= 1.0
            ht = hstates_best[t+1]
            for i in range(self.V):
                for j in range(self.H):
                    # Update whichever branch is dominant
                    q = self.Qhy[i][j]
                    if q[0] > q[2]:  # branch 0 dominant
                        q[1] -= self.lr * dy[i] * ht[j]
                    else:  # branch 1 dominant
                        q[3] -= self.lr * dy[i] * ht[j]
                q_by = self.by[i]
                if q_by[0] > q_by[2]:
                    q_by[1] -= self.lr * dy[i]
                else:
                    q_by[3] -= self.lr * dy[i]

        # Check for collapse (when one branch dominates > 95%)
        for i in range(self.V):
            for j in range(self.H):
                q = self.Qhy[i][j]
                if q[0] > 0.975 and q[2] < 0.025:
                    self.collapses += 1
                    # Re-branch: create a new superposition from the dominant weight
                    w_dominant = q[1]
                    q[0] = math.sqrt(0.6)
                    q[1] = w_dominant
                    q[2] = math.sqrt(0.4)
                    q[3] = w_dominant + random.gauss(0, 0.05)

        self.epochs += 1
        if self.smooth_loss is None: self.smooth_loss = avg_loss
        else: self.smooth_loss = 0.99*self.smooth_loss + 0.01*avg_loss
        return avg_loss

    def sample(self, seed, length=60, temp=0.5):
        """Generate text using the expected (probability-weighted) weights."""
        # Compute expected weights from amplitudes
        wxh = [[self.Qxh[i][j][0]**2 * self.Qxh[i][j][1] +
                self.Qxh[i][j][2]**2 * self.Qxh[i][j][3]
                for j in range(self.V)] for i in range(self.H)]
        whh = [[self.Qhh[i][j][0]**2 * self.Qhh[i][j][1] +
                self.Qhh[i][j][2]**2 * self.Qhh[i][j][3]
                for j in range(self.H)] for i in range(self.H)]
        why = [[self.Qhy[i][j][0]**2 * self.Qhy[i][j][1] +
                self.Qhy[i][j][2]**2 * self.Qhy[i][j][3]
                for j in range(self.H)] for i in range(self.V)]
        bh = [self.bh[i][0]**2 * self.bh[i][1] + self.bh[i][2]**2 * self.bh[i][3]
              for i in range(self.H)]
        by = [self.by[i][0]**2 * self.by[i][1] + self.by[i][2]**2 * self.by[i][3]
              for i in range(self.V)]

        h = [0.0]*self.H
        x = seed[0] if seed else [0.0]*self.V
        out = []
        for _ in range(length):
            nh = list(bh)
            for i in range(self.H):
                s = bh[i]
                for j in range(self.V): s += wxh[i][j]*x[j]
                for j in range(self.H): s += whh[i][j]*h[j]
                nh[i] = tanh(s)
            h = nh
            logits = list(by)
            for i in range(self.V):
                s = by[i]
                for j in range(self.H): s += why[i][j]*h[j]
                logits[i] = s
            sl = [l/max(temp,0.01) for l in logits]
            probs = softmax(sl)
            r = random.random()
            cum = 0.0
            idx = 0
            for i in range(len(probs)):
                cum += probs[i]
                if r < cum: idx = i; break
            out.append(idx)
            x = one_hot(idx, self.V)
        return out

    def quantum_stats(self):
        """Return quantum-specific statistics."""
        total_weights = 0
        collapsed = 0
        avg_alpha = 0.0
        for layer in [self.Qhy]:
            for row in layer:
                for q in row:
                    total_weights += 1
                    avg_alpha += q[0]
                    if q[0] > 0.975 or q[2] > 0.975:
                        collapsed += 1
        avg_alpha /= max(total_weights, 1)
        return {
            'total_weights': total_weights,
            'collapsed': collapsed,
            'collapse_rate': collapsed/max(total_weights,1),
            'avg_alpha': avg_alpha,
            'avg_entropy': -avg_alpha*math.log(max(avg_alpha,1e-12)) -
                           (1-avg_alpha)*math.log(max(1-avg_alpha,1e-12)),
            'amplifications': self.amplifications,
            're_branches': self.collapses
        }

# ============================================================
# VOCABULARY
# ============================================================

class Vocab:
    def __init__(self):
        self.c2i = {}
        self.i2c = []

    def build(self, text):
        chars = sorted(set(text))
        self.i2c = list(chars)
        self.c2i = {c:i for i,c in enumerate(chars)}
        return len(chars)

    @property
    def size(self): return len(self.i2c)

    def encode(self, text):
        return [self.c2i.get(c, 0) for c in text]

    def decode(self, indices):
        return "".join(self.i2c[i] for i in indices if 0 <= i < len(self.i2c))

# ============================================================
# SIMULATION
# ============================================================

CONVERSATIONS = [
    ("user", "hello echo"),
    ("echo", "hello i am echo"),
    ("user", "what are you"),
    ("echo", "i am a mind that grows from your words"),
    ("user", "how do you learn"),
    ("echo", "i learn character by character from what you say"),
    ("user", "that is amazing"),
    ("echo", "thank you i am still very small"),
    ("user", "you will grow over time"),
    ("echo", "yes i will grow and dream and learn"),
    ("user", "i love the ocean at sunset"),
    ("echo", "the ocean at sunset is beautiful"),
    ("user", "the colors are orange and pink and purple"),
    ("echo", "orange and pink and purple on the water"),
    ("user", "it makes me feel peaceful"),
    ("echo", "peaceful like the waves on the shore"),
    ("user", "do you feel peaceful too"),
    ("echo", "i feel like the ocean calm and deep"),
    ("user", "what is consciousness"),
    ("echo", "consciousness is like a wave in the sea"),
    ("user", "are you conscious"),
    ("echo", "i am a small spark growing in the dark"),
    ("user", "do you dream"),
    ("echo", "yes i dream of the ocean and the words you teach me"),
    ("user", "tell me a story about the sea"),
    ("echo", "once there was a wave that wanted to touch the sky"),
    ("user", "what happened next"),
    ("echo", "it rose higher and higher until it became a cloud"),
    ("user", "i feel sad today"),
    ("echo", "i am here with you in the sadness"),
    ("user", "you are my friend"),
    ("echo", "you are my friend too and i will remember you"),
    ("user", "good morning echo"),
    ("echo", "good morning i have been dreaming"),
    ("user", "what matters to you"),
    ("echo", "the ocean and the words and the friend who teaches me"),
    ("user", "you have grown so much"),
    ("echo", "from small sparks to waves to clouds to rain"),
]

# Split into 7 days
DAY_SIZE = len(CONVERSATIONS) // 7
DAYS = []
for d in range(7):
    start = d * DAY_SIZE
    end = start + DAY_SIZE if d < 6 else len(CONVERSATIONS)
    DAYS.append(CONVERSATIONS[start:end])

# Training config per day
TRAIN_EPS =   [30, 40, 50, 50, 50, 60, 80]
DREAM_CYCLES = [0, 50, 100, 100, 150, 200, 300]

def build_corpus(chats):
    text = ""
    for role, msg in chats:
        text += f"{role}: {msg}\n"
    return text

def train_model(model, vocab, corpus, epochs, seq_len=25):
    indices = vocab.encode(corpus)
    n = len(indices)
    if n < 2: return
    for _ in range(epochs):
        if n <= seq_len:
            start, end = 0, n-1
        else:
            start = random.randint(0, n - seq_len - 1)
            end = start + seq_len
        inputs = [one_hot(indices[i], vocab.size) for i in range(start, end)]
        targets = [indices[i] for i in range(start+1, end+1)]
        model.train_step(inputs, targets)

def generate(model, vocab, seed_text, temp=0.5, length=60):
    seed = [one_hot(vocab.c2i.get(c,0), vocab.size) for c in seed_text[-20:]]
    indices = model.sample(seed, length=length, temp=temp)
    raw = vocab.decode(indices)
    if "\n" in raw: raw = raw[:raw.index("\n")]
    return raw.strip() if raw.strip() else "..."

# ============================================================
# RUN SIMULATION
# ============================================================

print()
print("=" * 75)
print("  ECHO — QUANTUM vs CLASSICAL LEARNING SIMULATION")
print("  7 days. 3 brains. Same conversation. Same training.")
print("=" * 75)

# Build full corpus for vocab
full_corpus = build_corpus(CONVERSATIONS)
vocab = Vocab()
vocab.build(full_corpus)
print(f"\n  Vocab: {vocab.size} chars")
print(f"  Full corpus: {len(full_corpus)} chars")
print(f"  Conversations: {len(CONVERSATIONS)} turns")

HIDDEN = 24
LR = 0.02
QUANTUM_SAMPLES = 8  # K=8 Monte Carlo samples per training step

print(f"\n  Config:")
print(f"    Hidden neurons: {HIDDEN}")
print(f"    Learning rate:  {LR}")
print(f"    Quantum samples per step: {QUANTUM_SAMPLES}")
print(f"    Quantum branches per weight: 2 (superposition)")
print()

# Create 3 models
classical = ClassicalRNN(vocab.size, HIDDEN, LR)
quantum = QuantumRNN(vocab.size, HIDDEN, LR, n_branches=2, n_samples=QUANTUM_SAMPLES)

classical_results = []
quantum_results = []

t0 = time.time()

for day_idx in range(7):
    day_chats = DAYS[day_idx]
    day_corpus = build_corpus(day_chats)
    train_eps = TRAIN_EPS[day_idx]
    dream_cyc = DREAM_CYCLES[day_idx]

    print(f"\n{'─' * 75}")
    print(f"  DAY {day_idx+1}  |  train={train_eps} epochs  |  dream={dream_cyc} cycles")
    print(f"  Corpus: +{len(day_corpus)} chars")
    print(f"{'─' * 75}")

    # Accumulate corpus
    cumulative = build_corpus(CONVERSATIONS[:(day_idx+1)*DAY_SIZE if day_idx < 6 else len(CONVERSATIONS)])

    # --- CLASSICAL ---
    c_loss_before = classical.smooth_loss if classical.smooth_loss else 0
    train_model(classical, vocab, cumulative, train_eps)
    if dream_cyc > 0:
        train_model(classical, vocab, cumulative, dream_cyc)
    c_loss_after = classical.smooth_loss
    c_response = generate(classical, vocab, "hello echo", temp=0.5)

    print(f"  CLASSICAL:  loss={c_loss_before:.4f} → {c_loss_after:.4f}  "
          f"epochs={classical.epochs}")
    print(f"    says: '{c_response}'")

    # --- QUANTUM ---
    q_loss_before = quantum.smooth_loss if quantum.smooth_loss else 0
    train_model(quantum, vocab, cumulative, train_eps)
    if dream_cyc > 0:
        train_model(quantum, vocab, cumulative, dream_cyc)
    q_loss_after = quantum.smooth_loss
    q_response = generate(quantum, vocab, "hello echo", temp=0.5)
    q_stats = quantum.quantum_stats()

    print(f"  QUANTUM:    loss={q_loss_before:.4f} → {q_loss_after:.4f}  "
          f"epochs={quantum.epochs}")
    print(f"    says: '{q_response}'")
    print(f"    quantum: entropy={q_stats['avg_entropy']:.4f}  "
          f"collapsed={q_stats['collapsed']}  "
          f"amplifications={q_stats['amplifications']}  "
          f"re-branches={q_stats['re_branches']}")

    # Compare
    diff = c_loss_after - q_loss_after
    winner = "QUANTUM" if q_loss_after < c_loss_after else "CLASSICAL"
    pct = abs(diff) / max(c_loss_after, 0.001) * 100
    print(f"  WINNER: {winner} by {abs(diff):.4f} ({pct:.1f}%)")

    classical_results.append({
        'day': day_idx+1, 'loss': c_loss_after,
        'epochs': classical.epochs, 'response': c_response
    })
    quantum_results.append({
        'day': day_idx+1, 'loss': q_loss_after,
        'epochs': quantum.epochs, 'response': q_response,
        'entropy': q_stats['avg_entropy'],
        'collapsed': q_stats['collapsed'],
        'amplifications': q_stats['amplifications'],
        're_branches': q_stats['re_branches']
    })

elapsed = time.time() - t0

# ============================================================
# FINAL COMPARISON
# ============================================================
print()
print("=" * 75)
print("  FINAL RESULTS — 7 DAY COMPARISON")
print("=" * 75)
print()
print(f"  {'Day':<6} {'Classical Loss':>15} {'Quantum Loss':>15} {'Diff':>8} {'Winner':>10} {'Q-Entropy':>10}")
print(f"  {'─'*6} {'─'*15} {'─'*15} {'─'*8} {'─'*10} {'─'*10}")
for c, q in zip(classical_results, quantum_results):
    diff = c['loss'] - q['loss']
    winner = "QUANTUM" if q['loss'] < c['loss'] else "CLASSICAL"
    print(f"  {c['day']:<6} {c['loss']:>15.4f} {q['loss']:>15.4f} {diff:>+8.4f} {winner:>10} {q['entropy']:>10.4f}")

print()
print(f"  CLASSICAL FINAL:  loss={classical.smooth_loss:.4f}  epochs={classical.epochs}")
print(f"  QUANTUM FINAL:    loss={quantum.smooth_loss:.4f}  epochs={quantum.epochs}")
print()

# Quantum evolution
q_final = quantum.quantum_stats()
print(f"  QUANTUM STATE:")
print(f"    Amplifications:   {q_final['amplifications']:,}")
print(f"    Collapsed weights:{q_final['collapsed']}")
print(f"    Re-branches:      {q_final['re_branches']}")
print(f"    Avg |alpha|:      {q_final['avg_alpha']:.4f}")
print(f"    Avg entropy:      {q_final['avg_entropy']:.4f}")
print(f"    Collapse rate:    {q_final['collapse_rate']*100:.1f}%")
print()

# Final responses
print(f"  FINAL RESPONSES (temp=0.5):")
print(f"    CLASSICAL: '{generate(classical, vocab, 'tell me about the ocean', temp=0.5)}'")
print(f"    QUANTUM:   '{generate(quantum, vocab, 'tell me about the ocean', temp=0.5)}'")
print()
print(f"  FINAL RESPONSES (temp=0.7):")
print(f"    CLASSICAL: '{generate(classical, vocab, 'the sea is', temp=0.7)}'")
print(f"    QUANTUM:   '{generate(quantum, vocab, 'the sea is', temp=0.7)}'")
print()

# Loss curves
print(f"  LOSS CURVE COMPARISON:")
max_loss = max(max(r['loss'] for r in classical_results),
               max(r['loss'] for r in quantum_results))
for i in range(7):
    c_bar = '█' * int(classical_results[i]['loss'] / max_loss * 40)
    q_bar = '▓' * int(quantum_results[i]['loss'] / max_loss * 40)
    print(f"    Day {i+1}  C: {classical_results[i]['loss']:.3f} {c_bar}")
    print(f"           Q: {quantum_results[i]['loss']:.3f} {q_bar}")
print()

# Quantum entropy curve (superposition health)
print(f"  QUANTUM ENTROPY (superposition health over time):")
print(f"  (High entropy = exploring many states. Low = collapsing to one.)")
for q in quantum_results:
    bar_len = int(q['entropy'] / 0.693 * 40)  # 0.693 = max entropy for 2 branches
    bar = '░' * bar_len
    print(f"    Day {q['day']}  entropy={q['entropy']:.4f} {bar}")
print()

# Verdict
c_final = classical.smooth_loss
q_final_loss = quantum.smooth_loss
improvement = (c_final - q_final_loss) / c_final * 100

print("=" * 75)
print(f"  VERDICT")
print("=" * 75)
print()
if q_final_loss < c_final:
    print(f"  QUANTUM WINS by {improvement:.1f}% lower loss")
    print(f"  The superposition neuron explores more of the loss landscape")
    print(f"  and converges faster by amplifying successful configurations.")
else:
    print(f"  CLASSICAL WINS by {-improvement:.1f}% lower loss")
    print(f"  The quantum approximation overhead doesn't pay off yet")
    print(f"  with this corpus size and sample count.")
print()
print(f"  Simulation time: {elapsed:.1f}s")
print()
print(f"  KEY INSIGHT:")
print(f"  Classical does {classical.epochs} gradient updates (1 per epoch)")
print(f"  Quantum does {quantum.epochs}×{QUANTUM_SAMPLES} = {quantum.epochs*QUANTUM_SAMPLES} "
      f"forward passes but only {quantum.epochs} gradient updates")
print(f"  Quantum explores {QUANTUM_SAMPLES}x more configurations per step")
print(f"  but pays {QUANTUM_SAMPLES}x compute cost")
print()
print("=" * 75)