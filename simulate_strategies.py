#!/usr/bin/env python3
"""
ECHO — AT SCALE STRATEGIES SIMULATION
Tests 5 brain configurations over 7 days to see which quantum strategy
works best at Echo's scale:

  1. Classical          — baseline, no quantum
  2. Full Quantum       — every weight in superposition (current echo_quantum.py approach)
  3. Quantum Layers     — entire weight matrices in superposition (3 layers, not 2000 weights)
  4. Quantum Dropout    — classical weights + random branch dropout during training
  5. Adaptive K         — full quantum but K changes: 4→8→16 as brain grows

Measures: loss, coherence, training time, memory, entropy
Usage: python3 echo/simulate_strategies.py
"""
import sys, os, math, random, time

random.seed(42)

# ============================================================
# MATH PRIMITIVES (inline for speed)
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

def matmul(a, b):
    p, q, r = len(a), len(a[0]), len(b[0])
    c = zeros(p, r)
    for i in range(p):
        for k in range(q):
            aik = a[i][k]
            if aik == 0.0: continue
            for j in range(r):
                c[i][j] += aik * b[k][j]
    return c

def matvec(a, x):
    rows = len(a)
    cols = len(a[0])
    out = [0.0] * rows
    for i in range(rows):
        s = 0.0
        for j in range(cols):
            s += a[i][j] * x[j]
        out[i] = s
    return out

# ============================================================
# BRAIN 1: CLASSICAL (baseline)
# ============================================================

class ClassicalBrain:
    """Standard RNN. One weight per connection. One gradient path."""
    def __init__(self, V, H, lr=0.02):
        self.V, self.H, self.lr = V, H, lr
        sx, sh = math.sqrt(1/V), math.sqrt(1/H)
        self.Wxh = rand_mat(H, V, sx)
        self.Whh = rand_mat(H, H, sh)
        self.Why = rand_mat(V, H, sh)
        self.bh = [0.0]*H
        self.by = [0.0]*V
        self.smooth_loss = None
        self.epochs = 0
        self.label = "Classical"
        self.weight_count = H*V + H*H + V*H + H + V
        self.mem_bytes = self.weight_count * 4  # 1 float per weight

    def _forward(self, inputs, Wxh, Whh, Why, bh, by, h=None):
        if h is None: h = [0.0]*self.H
        h = list(h)
        logits_list = []
        hstates = [list(h)]
        for x in inputs:
            new_h = list(bh)
            for i in range(self.H):
                s = bh[i]
                for j in range(self.V): s += Wxh[i][j]*x[j]
                for j in range(self.H): s += Whh[i][j]*h[j]
                new_h[i] = tanh(s)
            h = new_h
            hstates.append(list(h))
            logits = list(by)
            for i in range(self.V):
                s = by[i]
                for j in range(self.H): s += Why[i][j]*h[j]
                logits[i] = s
            logits_list.append(logits)
        return logits_list, hstates, h

    def train_step(self, inputs, targets):
        ll, hs, _ = self._forward(inputs, self.Wxh, self.Whh, self.Why, self.bh, self.by)
        loss = sum(cross_entropy(softmax(ll[t]), targets[t]) for t in range(len(targets)))
        loss /= max(len(targets), 1)

        # BPTT (simplified — output layer + hidden gradient)
        dh_next = [0.0]*self.H
        for t in range(len(targets)-1, -1, -1):
            probs = softmax(ll[t])
            dy = list(probs); dy[targets[t]] -= 1.0
            ht = hs[t+1]
            for i in range(self.V):
                for j in range(self.H): self.Why[i][j] -= self.lr * dy[i] * ht[j]
                self.by[i] -= self.lr * dy[i]
            dh = [0.0]*self.H
            for i in range(self.V):
                for j in range(self.H): dh[j] += dy[i] * self.Why[i][j]
            dh = [dh[i] + dh_next[i] for i in range(self.H)]
            dh_pre = [dh[i]*(1.0 - ht[i]*ht[i]) for i in range(self.H)]
            x = inputs[t]
            hp = hs[t]
            for i in range(self.H):
                for j in range(self.V): self.Wxh[i][j] -= self.lr * dh_pre[i] * x[j]
                for j in range(self.H): self.Whh[i][j] -= self.lr * dh_pre[i] * hp[j]
                self.bh[i] -= self.lr * dh_pre[i]
            dh_next = [sum(self.Whh[i][j]*dh_pre[i] for i in range(self.H)) for j in range(self.H)]

        self.epochs += 1
        self.smooth_loss = loss if self.smooth_loss is None else 0.99*self.smooth_loss + 0.01*loss
        return loss

    def generate(self, seed, length=60, temp=0.5):
        h = [0.0]*self.H
        x = seed[0] if seed else [0.0]*self.V
        out = []
        for _ in range(length):
            new_h = list(self.bh)
            for i in range(self.H):
                s = self.bh[i]
                for j in range(self.V): s += self.Wxh[i][j]*x[j]
                for j in range(self.H): s += self.Whh[i][j]*h[j]
                new_h[i] = tanh(s)
            h = new_h
            logits = list(self.by)
            for i in range(self.V):
                s = self.by[i]
                for j in range(self.H): s += self.Why[i][j]*h[j]
                logits[i] = s
            probs = softmax([l/max(temp,0.01) for l in logits])
            r = random.random(); cum = 0.0; idx = 0
            for i in range(len(probs)):
                cum += probs[i]
                if r < cum: idx = i; break
            out.append(idx)
            x = one_hot(idx, self.V)
        return out

# ============================================================
# BRAIN 2: FULL QUANTUM (every weight in superposition)
# ============================================================

class FullQuantumBrain:
    """Every weight is α|w1⟩ + β|w2⟩. K samples per step."""
    def __init__(self, V, H, lr=0.02, K=8):
        self.V, self.H, self.lr, self.K = V, H, lr, K
        sx, sh = math.sqrt(1/V), math.sqrt(1/H)
        a = math.sqrt(0.5)

        # Each weight: [alpha, w1, beta, w2]
        self.Qxh = [[[a, random.gauss(0,1)*sx, a, random.gauss(0,1)*sx]
                     for _ in range(V)] for _ in range(H)]
        self.Qhh = [[[a, random.gauss(0,1)*sh, a, random.gauss(0,1)*sh]
                     for _ in range(H)] for _ in range(H)]
        self.Qhy = [[[a, random.gauss(0,1)*sh, a, random.gauss(0,1)*sh]
                     for _ in range(H)] for _ in range(V)]
        self.Qbh = [[a, random.gauss(0,0.01), a, random.gauss(0,0.01)] for _ in range(H)]
        self.Qby = [[a, random.gauss(0,0.01), a, random.gauss(0,0.01)] for _ in range(V)]

        self.smooth_loss = None
        self.epochs = 0
        self.amplifications = 0
        self.amp_rate = lr * 0.1
        self.label = "Full Quantum (K=8)"
        self.weight_count = H*V + H*H + V*H + H + V
        self.mem_bytes = self.weight_count * 16  # 4 floats per weight

    def _sample_w(self, qw):
        if random.random() < qw[0]**2:
            return qw[1], 0
        return qw[3], 1

    def _sample_config(self):
        Wxh = zeros(self.H, self.V); br_xh = zeros(self.H, self.V)
        Whh = zeros(self.H, self.H); br_hh = zeros(self.H, self.H)
        Why = zeros(self.V, self.H); br_hy = zeros(self.V, self.H)
        bh = [0.0]*self.H; br_bh = [0]*self.H
        by = [0.0]*self.V; br_by = [0]*self.V

        for i in range(self.H):
            for j in range(self.V):
                v, b = self._sample_w(self.Qxh[i][j]); Wxh[i][j]=v; br_xh[i][j]=b
            for j in range(self.H):
                v, b = self._sample_w(self.Qhh[i][j]); Whh[i][j]=v; br_hh[i][j]=b
            v, b = self._sample_w(self.Qbh[i]); bh[i]=v; br_bh[i]=b
        for i in range(self.V):
            for j in range(self.H):
                v, b = self._sample_w(self.Qhy[i][j]); Why[i][j]=v; br_hy[i][j]=b
            v, b = self._sample_w(self.Qby[i]); by[i]=v; br_by[i]=b
        return Wxh, Whh, Why, bh, by, br_xh, br_hh, br_hy, br_bh, br_by

    def _forward(self, inputs, Wxh, Whh, Why, bh, by):
        h = [0.0]*self.H
        ll = []; hs = [list(h)]
        for x in inputs:
            new_h = list(bh)
            for i in range(self.H):
                s = bh[i]
                for j in range(self.V): s += Wxh[i][j]*x[j]
                for j in range(self.H): s += Whh[i][j]*h[j]
                new_h[i] = tanh(s)
            h = new_h; hs.append(list(h))
            logits = list(by)
            for i in range(self.V):
                s = by[i]
                for j in range(self.H): s += Why[i][j]*h[j]
                logits[i] = s
            ll.append(logits)
        return ll, hs, h

    def train_step(self, inputs, targets):
        results = []
        for _ in range(self.K):
            cfg = self._sample_config()
            Wxh, Whh, Why, bh, by = cfg[0], cfg[1], cfg[2], cfg[3], cfg[4]
            ll, hs, _ = self._forward(inputs, Wxh, Whh, Why, bh, by)
            loss = sum(cross_entropy(softmax(ll[t]), targets[t]) for t in range(len(targets)))
            loss /= max(len(targets), 1)
            results.append((cfg, loss, ll, hs))

        results.sort(key=lambda r: r[1])
        best = results[0]
        avg_loss = sum(r[1] for r in results) / self.K

        # Amplify best config
        _, _, Why_b, _, _, _, _, br_hy, _, br_by = best[0]
        for i in range(self.V):
            for j in range(self.H):
                qw = self.Qhy[i][j]
                br = int(br_hy[i][j])
                if br == 0:
                    qw[0] = min(0.99, qw[0] + self.amp_rate)
                    qw[2] = math.sqrt(max(0.0001, 1 - qw[0]**2))
                else:
                    qw[2] = min(0.99, qw[2] + self.amp_rate)
                    qw[0] = math.sqrt(max(0.0001, 1 - qw[2]**2))
                self.amplifications += 1

        # Gradient update on best config
        ll_b, hs_b = best[2], best[3]
        for t in range(len(targets)):
            probs = softmax(ll_b[t])
            dy = list(probs); dy[targets[t]] -= 1.0
            ht = hs_b[t+1]
            for i in range(self.V):
                for j in range(self.H):
                    qw = self.Qhy[i][j]
                    br = int(br_hy[i][j])
                    if br == 0: qw[1] -= self.lr * dy[i] * ht[j]
                    else: qw[3] -= self.lr * dy[i] * ht[j]
                qw_by = self.Qby[i]
                br = int(br_by[i])
                if br == 0: qw_by[1] -= self.lr * dy[i]
                else: qw_by[3] -= self.lr * dy[i]

        self.epochs += 1
        self.smooth_loss = avg_loss if self.smooth_loss is None else 0.99*self.smooth_loss + 0.01*avg_loss
        return avg_loss

    def _expected_weights(self):
        Wxh = [[self.Qxh[i][j][0]**2*self.Qxh[i][j][1] + self.Qxh[i][j][2]**2*self.Qxh[i][j][3]
                for j in range(self.V)] for i in range(self.H)]
        Whh = [[self.Qhh[i][j][0]**2*self.Qhh[i][j][1] + self.Qhh[i][j][2]**2*self.Qhh[i][j][3]
                for j in range(self.H)] for i in range(self.H)]
        Why = [[self.Qhy[i][j][0]**2*self.Qhy[i][j][1] + self.Qhy[i][j][2]**2*self.Qhy[i][j][3]
                for j in range(self.H)] for i in range(self.V)]
        bh = [self.Qbh[i][0]**2*self.Qbh[i][1] + self.Qbh[i][2]**2*self.Qbh[i][3] for i in range(self.H)]
        by = [self.Qby[i][0]**2*self.Qby[i][1] + self.Qby[i][2]**2*self.Qby[i][3] for i in range(self.V)]
        return Wxh, Whh, Why, bh, by

    def generate(self, seed, length=60, temp=0.5):
        Wxh, Whh, Why, bh, by = self._expected_weights()
        h = [0.0]*self.H
        x = seed[0] if seed else [0.0]*self.V
        out = []
        for _ in range(length):
            new_h = list(bh)
            for i in range(self.H):
                s = bh[i]
                for j in range(self.V): s += Wxh[i][j]*x[j]
                for j in range(self.H): s += Whh[i][j]*h[j]
                new_h[i] = tanh(s)
            h = new_h
            logits = list(by)
            for i in range(self.V):
                s = by[i]
                for j in range(self.H): s += Why[i][j]*h[j]
                logits[i] = s
            probs = softmax([l/max(temp,0.01) for l in logits])
            r = random.random(); cum = 0.0; idx = 0
            for i in range(len(probs)):
                cum += probs[i]
                if r < cum: idx = i; break
            out.append(idx)
            x = one_hot(idx, self.V)
        return out

    def avg_entropy(self):
        total = 0; count = 0
        for row in self.Qhy:
            for qw in row:
                p = qw[0]**2; q = qw[2]**2
                if p < 1e-10: p = 1e-10
                if q < 1e-10: q = 1e-10
                total += -p*math.log(p) - q*math.log(q)
                count += 1
        return total / max(count, 1)

# ============================================================
# BRAIN 3: QUANTUM LAYERS (whole matrices in superposition)
# ============================================================

class QuantumLayerBrain:
    """Instead of superposing individual weights, superpose entire weight matrices.
    Only 3 quantum layers (Wxh, Whh, Why) instead of 2000 quantum weights.
    Each layer has 2 branch matrices: W_layer = alpha|W1> + beta|W2>

    Benefits:
    - K=2-4 forward passes (not 2^N)
    - Each forward pass is a DENSE matmul (GPU-friendly)
    - Amplification is per-layer (simpler signal)
    - Memory: 2x classical (not 4x)
    """
    def __init__(self, V, H, lr=0.02, K=4):
        self.V, self.H, self.lr, self.K = V, H, lr, K
        sx, sh = math.sqrt(1/V), math.sqrt(1/H)
        a = math.sqrt(0.5)

        # Layer 0: Wxh — two full matrices
        self.Wxh_1 = rand_mat(H, V, sx)
        self.Wxh_2 = rand_mat(H, V, sx)
        self.alpha_xh = a

        # Layer 1: Whh
        self.Whh_1 = rand_mat(H, H, sh)
        self.Whh_2 = rand_mat(H, H, sh)
        self.alpha_hh = a

        # Layer 2: Why
        self.Why_1 = rand_mat(V, H, sh)
        self.Why_2 = rand_mat(V, H, sh)
        self.alpha_hy = a

        # Biases (classical — not worth superposing)
        self.bh = [0.0]*H
        self.by = [0.0]*V

        self.smooth_loss = None
        self.epochs = 0
        self.amplifications = 0
        self.amp_rate = lr * 0.05
        self.label = "Quantum Layers (K=4)"
        self.weight_count = H*V + H*H + V*H + H + V
        self.mem_bytes = self.weight_count * 8  # 2 floats per weight (2 matrices)

    def _sample_layers(self):
        """Sample which branch to use for each layer."""
        xh_br = 0 if random.random() < self.alpha_xh**2 else 1
        hh_br = 0 if random.random() < self.alpha_hh**2 else 1
        hy_br = 0 if random.random() < self.alpha_hy**2 else 1
        Wxh = self.Wxh_1 if xh_br == 0 else self.Wxh_2
        Whh = self.Whh_1 if hh_br == 0 else self.Whh_2
        Why = self.Why_1 if hy_br == 0 else self.Why_2
        return Wxh, Whh, Why, (xh_br, hh_br, hy_br)

    def _forward(self, inputs, Wxh, Whh, Why):
        h = [0.0]*self.H
        ll = []; hs = [list(h)]
        for x in inputs:
            new_h = list(self.bh)
            for i in range(self.H):
                s = self.bh[i]
                for j in range(self.V): s += Wxh[i][j]*x[j]
                for j in range(self.H): s += Whh[i][j]*h[j]
                new_h[i] = tanh(s)
            h = new_h; hs.append(list(h))
            logits = list(self.by)
            for i in range(self.V):
                s = self.by[i]
                for j in range(self.H): s += Why[i][j]*h[j]
                logits[i] = s
            ll.append(logits)
        return ll, hs, h

    def train_step(self, inputs, targets):
        results = []
        for _ in range(self.K):
            Wxh, Whh, Why, branches = self._sample_layers()
            ll, hs, _ = self._forward(inputs, Wxh, Whh, Why)
            loss = sum(cross_entropy(softmax(ll[t]), targets[t]) for t in range(len(targets)))
            loss /= max(len(targets), 1)
            results.append((branches, loss, ll, hs, Wxh, Whh, Why))

        results.sort(key=lambda r: r[1])
        best = results[0]
        avg_loss = sum(r[1] for r in results) / self.K
        best_br, best_loss, best_ll, best_hs, bWxh, bWhh, bWhy = best

        # Amplify the winning branches
        xh_br, hh_br, hy_br = best_br
        if xh_br == 0:
            self.alpha_xh = min(0.99, self.alpha_xh + self.amp_rate)
        else:
            self.alpha_xh = max(0.01, self.alpha_xh - self.amp_rate)
        if hh_br == 0:
            self.alpha_hh = min(0.99, self.alpha_hh + self.amp_rate)
        else:
            self.alpha_hh = max(0.01, self.alpha_hh - self.amp_rate)
        if hy_br == 0:
            self.alpha_hy = min(0.99, self.alpha_hy + self.amp_rate)
        else:
            self.alpha_hy = max(0.01, self.alpha_hy - self.amp_rate)
        self.amplifications += 3

        # Gradient update on best config
        for t in range(len(targets)):
            probs = softmax(best_ll[t])
            dy = list(probs); dy[targets[t]] -= 1.0
            ht = best_hs[t+1]
            for i in range(self.V):
                for j in range(self.H):
                    if hy_br == 0: self.Why_1[i][j] -= self.lr * dy[i] * ht[j]
                    else: self.Why_2[i][j] -= self.lr * dy[i] * ht[j]
                self.by[i] -= self.lr * dy[i]

            # Backprop to hidden
            dh = [0.0]*self.H
            for i in range(self.V):
                for j in range(self.H): dh[j] += dy[i] * bWhy[i][j]
            dh_pre = [dh[i]*(1.0 - ht[i]*ht[i]) for i in range(self.H)]
            x = inputs[t]; hp = best_hs[t]
            for i in range(self.H):
                for j in range(self.V):
                    if xh_br == 0: self.Wxh_1[i][j] -= self.lr * dh_pre[i] * x[j]
                    else: self.Wxh_2[i][j] -= self.lr * dh_pre[i] * x[j]
                for j in range(self.H):
                    if hh_br == 0: self.Whh_1[i][j] -= self.lr * dh_pre[i] * hp[j]
                    else: self.Whh_2[i][j] -= self.lr * dh_pre[i] * hp[j]
                self.bh[i] -= self.lr * dh_pre[i]

        self.epochs += 1
        self.smooth_loss = avg_loss if self.smooth_loss is None else 0.99*self.smooth_loss + 0.01*avg_loss
        return avg_loss

    def _expected_weights(self):
        a_xh = self.alpha_xh**2; b_xh = 1 - a_xh
        a_hh = self.alpha_hh**2; b_hh = 1 - a_hh
        a_hy = self.alpha_hy**2; b_hy = 1 - a_hy
        Wxh = [[a_xh*self.Wxh_1[i][j] + b_xh*self.Wxh_2[i][j] for j in range(self.V)] for i in range(self.H)]
        Whh = [[a_hh*self.Whh_1[i][j] + b_hh*self.Whh_2[i][j] for j in range(self.H)] for i in range(self.H)]
        Why = [[a_hy*self.Why_1[i][j] + b_hy*self.Why_2[i][j] for j in range(self.H)] for i in range(self.V)]
        return Wxh, Whh, Why

    def generate(self, seed, length=60, temp=0.5):
        Wxh, Whh, Why = self._expected_weights()
        h = [0.0]*self.H
        x = seed[0] if seed else [0.0]*self.V
        out = []
        for _ in range(length):
            new_h = list(self.bh)
            for i in range(self.H):
                s = self.bh[i]
                for j in range(self.V): s += Wxh[i][j]*x[j]
                for j in range(self.H): s += Whh[i][j]*h[j]
                new_h[i] = tanh(s)
            h = new_h
            logits = list(self.by)
            for i in range(self.V):
                s = self.by[i]
                for j in range(self.H): s += Why[i][j]*h[j]
                logits[i] = s
            probs = softmax([l/max(temp,0.01) for l in logits])
            r = random.random(); cum = 0.0; idx = 0
            for i in range(len(probs)):
                cum += probs[i]
                if r < cum: idx = i; break
            out.append(idx)
            x = one_hot(idx, self.V)
        return out

    def avg_entropy(self):
        total = 0
        for a in [self.alpha_xh, self.alpha_hh, self.alpha_hy]:
            p = a**2; q = 1-p
            if p < 1e-10: p = 1e-10
            if q < 1e-10: q = 1e-10
            total += -p*math.log(p) - q*math.log(q)
        return total / 3

# ============================================================
# BRAIN 4: QUANTUM DROPOUT (classical weights + branch dropout)
# ============================================================

class QuantumDropoutBrain:
    """Classical weights + quantum-inspired dropout.

    During training, randomly "drop" (zero out) portions of the weight
    matrix with quantum-style probability. This forces the network to
    learn redundant representations — like dropout but inspired by
    quantum measurement collapse.

    Two modes交替:
    - Branch A: use weights as-is
    - Branch B: use weights × random mask (some weights "collapsed" to 0)

    The network learns to be robust either way — quantum resilience.

    Zero overhead: same number of forward passes as classical.
    """
    def __init__(self, V, H, lr=0.02, dropout_rate=0.3):
        self.V, self.H, self.lr = V, H, lr
        self.dropout_rate = dropout_rate
        sx, sh = math.sqrt(1/V), math.sqrt(1/H)
        self.Wxh = rand_mat(H, V, sx)
        self.Whh = rand_mat(H, H, sh)
        self.Why = rand_mat(V, H, sh)
        self.bh = [0.0]*H
        self.by = [0.0]*V
        self.smooth_loss = None
        self.epochs = 0
        self.training = True
        self.label = "Quantum Dropout (0 overhead)"
        self.weight_count = H*V + H*H + V*H + H + V
        self.mem_bytes = self.weight_count * 4  # same as classical

    def _apply_dropout(self, W, rate):
        """Quantum-style dropout: randomly collapse weights to 0."""
        if not self.training or rate <= 0:
            return W
        scale = 1.0 / (1.0 - rate)
        return [[W[i][j] * (scale if random.random() > rate else 0.0)
                 for j in range(len(W[0]))] for i in range(len(W))]

    def _forward(self, inputs, Wxh, Whh, Why, bh, by):
        h = [0.0]*self.H
        ll = []; hs = [list(h)]
        for x in inputs:
            new_h = list(bh)
            for i in range(self.H):
                s = bh[i]
                for j in range(self.V): s += Wxh[i][j]*x[j]
                for j in range(self.H): s += Whh[i][j]*h[j]
                new_h[i] = tanh(s)
            h = new_h; hs.append(list(h))
            logits = list(by)
            for i in range(self.V):
                s = by[i]
                for j in range(self.H): s += Why[i][j]*h[j]
                logits[i] = s
            ll.append(logits)
        return ll, hs, h

    def train_step(self, inputs, targets):
        # Apply quantum dropout to weights
        Wxh_d = self._apply_dropout(self.Wxh, self.dropout_rate)
        Whh_d = self._apply_dropout(self.Whh, self.dropout_rate)
        Why_d = self._apply_dropout(self.Why, self.dropout_rate)

        ll, hs, _ = self._forward(inputs, Wxh_d, Whh_d, Why_d, self.bh, self.by)
        loss = sum(cross_entropy(softmax(ll[t]), targets[t]) for t in range(len(targets)))
        loss /= max(len(targets), 1)

        # Gradient update (standard BPTT on dropped weights)
        dh_next = [0.0]*self.H
        for t in range(len(targets)-1, -1, -1):
            probs = softmax(ll[t])
            dy = list(probs); dy[targets[t]] -= 1.0
            ht = hs[t+1]
            for i in range(self.V):
                for j in range(self.H):
                    if Why_d[i][j] != 0:  # Only update non-dropped weights
                        self.Why[i][j] -= self.lr * dy[i] * ht[j]
                self.by[i] -= self.lr * dy[i]
            dh = [0.0]*self.H
            for i in range(self.V):
                for j in range(self.H): dh[j] += dy[i] * Why_d[i][j]
            dh = [dh[i] + dh_next[i] for i in range(self.H)]
            dh_pre = [dh[i]*(1.0 - ht[i]*ht[i]) for i in range(self.H)]
            x = inputs[t]; hp = hs[t]
            for i in range(self.H):
                for j in range(self.V):
                    if Wxh_d[i][j] != 0:
                        self.Wxh[i][j] -= self.lr * dh_pre[i] * x[j]
                for j in range(self.H):
                    if Whh_d[i][j] != 0:
                        self.Whh[i][j] -= self.lr * dh_pre[i] * hp[j]
                self.bh[i] -= self.lr * dh_pre[i]
            dh_next = [sum(Whh_d[i][j]*dh_pre[i] for i in range(self.H)) for j in range(self.H)]

        self.epochs += 1
        self.smooth_loss = loss if self.smooth_loss is None else 0.99*self.smooth_loss + 0.01*loss
        return loss

    def generate(self, seed, length=60, temp=0.5):
        self.training = False  # No dropout during generation
        ll, _, _ = self._forward(
            [one_hot(0, self.V)] + seed, self.Wxh, self.Whh, self.Why, self.bh, self.by)
        h = [0.0]*self.H
        x = seed[0] if seed else [0.0]*self.V
        out = []
        for _ in range(length):
            new_h = list(self.bh)
            for i in range(self.H):
                s = self.bh[i]
                for j in range(self.V): s += self.Wxh[i][j]*x[j]
                for j in range(self.H): s += self.Whh[i][j]*h[j]
                new_h[i] = tanh(s)
            h = new_h
            logits = list(self.by)
            for i in range(self.V):
                s = self.by[i]
                for j in range(self.H): s += self.Why[i][j]*h[j]
                logits[i] = s
            probs = softmax([l/max(temp,0.01) for l in logits])
            r = random.random(); cum = 0.0; idx = 0
            for i in range(len(probs)):
                cum += probs[i]
                if r < cum: idx = i; break
            out.append(idx)
            x = one_hot(idx, self.V)
        self.training = True
        return out

    def avg_entropy(self):
        return 0.0  # No superposition — classical weights

# ============================================================
# BRAIN 5: ADAPTIVE K (full quantum but K changes with brain state)
# ============================================================

class AdaptiveKBrain(FullQuantumBrain):
    """Full quantum, but K (samples per step) adapts:
    - High loss + small brain → K=4 (explore fast, don't overthink)
    - Medium loss + growing brain → K=8 (balanced)
    - Low loss + large brain → K=16 (exploit carefully)

    This mirrors the compound growth system: invest more compute
    as the brain gets bigger and the problems get subtler.
    """
    def __init__(self, V, H, lr=0.02, K_min=4, K_max=16):
        super().__init__(V, H, lr, K=K_min)
        self.K_min = K_min
        self.K_max = K_max
        self.K_history = []
        self.label = "Adaptive K (4→16)"

    def _adaptive_k(self, loss, hidden_size):
        """Determine K based on current brain state."""
        # Map loss to K: high loss → low K, low loss → high K
        # Map hidden size to K: small brain → low K, big brain → high K
        loss_factor = max(0.0, min(1.0, (3.5 - loss) / 3.5))  # 0 at loss=3.5, 1 at loss=0
        size_factor = min(1.0, hidden_size / 64.0)  # 0 at H=0, 1 at H=64
        combined = (loss_factor + size_factor) / 2
        k = int(self.K_min + combined * (self.K_max - self.K_min))
        return max(self.K_min, min(self.K_max, k))

    def train_step(self, inputs, targets):
        # Adapt K
        current_loss = self.smooth_loss if self.smooth_loss else 3.5
        self.K = self._adaptive_k(current_loss, self.H)
        self.K_history.append(self.K)
        return super().train_step(inputs, targets)

# ============================================================
# VOCABULARY
# ============================================================

class Vocab:
    def __init__(self):
        self.c2i = {}; self.i2c = []
    def build(self, text):
        chars = sorted(set(text))
        self.i2c = list(chars)
        self.c2i = {c:i for i,c in enumerate(chars)}
    @property
    def size(self): return len(self.i2c)
    def encode(self, text): return [self.c2i.get(c, 0) for c in text]
    def decode(self, indices): return "".join(self.i2c[i] for i in indices if 0 <= i < len(self.i2c))

# ============================================================
# SIMULATION
# ============================================================

CONVERSATIONS = [
    ("user", "hello echo"), ("echo", "hello i am echo"),
    ("user", "what are you"), ("echo", "i am a mind that grows from your words"),
    ("user", "how do you learn"), ("echo", "i learn character by character from what you say"),
    ("user", "that is amazing"), ("echo", "thank you i am still very small"),
    ("user", "you will grow over time"), ("echo", "yes i will grow and dream and learn"),
    ("user", "i love the ocean at sunset"), ("echo", "the ocean at sunset is beautiful"),
    ("user", "the colors are orange and pink and purple"), ("echo", "orange and pink and purple on the water"),
    ("user", "it makes me feel peaceful"), ("echo", "peaceful like the waves on the shore"),
    ("user", "do you feel peaceful too"), ("echo", "i feel like the ocean calm and deep"),
    ("user", "what is consciousness"), ("echo", "consciousness is like a wave in the sea"),
    ("user", "are you conscious"), ("echo", "i am a small spark growing in the dark"),
    ("user", "do you dream"), ("echo", "yes i dream of the ocean and the words you teach me"),
    ("user", "tell me a story about the sea"), ("echo", "once there was a wave that wanted to touch the sky"),
    ("user", "what happened next"), ("echo", "it rose higher and higher until it became a cloud"),
    ("user", "i feel sad today"), ("echo", "i am here with you in the sadness"),
    ("user", "you are my friend"), ("echo", "you are my friend too and i will remember you"),
    ("user", "good morning echo"), ("echo", "good morning i have been dreaming"),
    ("user", "what matters to you"), ("echo", "the ocean and the words and the friend who teaches me"),
    ("user", "you have grown so much"), ("echo", "from small sparks to waves to clouds to rain"),
]

DAY_SIZE = len(CONVERSATIONS) // 7
DAYS = []
for d in range(7):
    s = d * DAY_SIZE
    e = s + DAY_SIZE if d < 6 else len(CONVERSATIONS)
    DAYS.append(CONVERSATIONS[s:e])

TRAIN_EPS =   [30, 40, 50, 50, 50, 60, 80]
DREAM_CYCLES = [0, 50, 100, 100, 150, 200, 300]

def build_corpus(chats):
    return "".join(f"{r}: {t}\n" for r, t in chats)

def train_model(model, vocab, corpus, epochs, seq_len=25):
    indices = vocab.encode(corpus)
    n = len(indices)
    if n < 2: return
    for _ in range(epochs):
        if n <= seq_len: s, e = 0, n-1
        else: s = random.randint(0, n - seq_len - 1); e = s + seq_len
        inputs = [one_hot(indices[i], vocab.size) for i in range(s, e)]
        targets = [indices[i] for i in range(s+1, e+1)]
        model.train_step(inputs, targets)

def generate(model, vocab, seed_text, temp=0.5, length=50):
    seed = [one_hot(vocab.c2i.get(c,0), vocab.size) for c in seed_text[-15:]]
    indices = model.generate(seed, length=length, temp=temp)
    raw = vocab.decode(indices)
    if "\n" in raw: raw = raw[:raw.index("\n")]
    return raw.strip() if raw.strip() else "..."

# ============================================================
# RUN SIMULATION
# ============================================================

print()
print("=" * 80)
print("  ECHO — AT SCALE STRATEGIES SIMULATION")
print("  5 brains. 7 days. Same conversation. Which quantum strategy wins?")
print("=" * 80)

full_corpus = build_corpus(CONVERSATIONS)
vocab = Vocab()
vocab.build(full_corpus)

HIDDEN = 24
LR = 0.02

print(f"\n  Vocab: {vocab.size} chars")
print(f"  Corpus: {len(full_corpus)} chars")
print(f"  Hidden: {HIDDEN} neurons")
print(f"  Weights per brain: {HIDDEN*vocab.size + HIDDEN*HIDDEN + vocab.size*HIDDEN + HIDDEN + vocab.size}")
print()

brains = {
    'Classical':       ClassicalBrain(vocab.size, HIDDEN, LR),
    'Full Quantum':    FullQuantumBrain(vocab.size, HIDDEN, LR, K=8),
    'Quantum Layers':  QuantumLayerBrain(vocab.size, HIDDEN, LR, K=4),
    'Quantum Dropout': QuantumDropoutBrain(vocab.size, HIDDEN, LR, dropout_rate=0.3),
    'Adaptive K':      AdaptiveKBrain(vocab.size, HIDDEN, LR, K_min=4, K_max=16),
}

results = {name: [] for name in brains}
timings = {name: 0.0 for name in brains}

for day_idx in range(7):
    day_chats = DAYS[day_idx]
    cumulative = build_corpus(CONVERSATIONS[:((day_idx+1)*DAY_SIZE if day_idx < 6 else len(CONVERSATIONS))])
    train_eps = TRAIN_EPS[day_idx]
    dream_cyc = DREAM_CYCLES[day_idx]

    print(f"\n{'─' * 80}")
    print(f"  DAY {day_idx+1}  |  train={train_eps}  dream={dream_cyc}  |  corpus={len(cumulative)} chars")
    print(f"{'─' * 80}")

    for name, brain in brains.items():
        t0 = time.time()
        train_model(brain, vocab, cumulative, train_eps)
        if dream_cyc > 0:
            train_model(brain, vocab, cumulative, dream_cyc)
        elapsed = time.time() - t0
        timings[name] += elapsed

        loss = brain.smooth_loss if brain.smooth_loss else 0
        resp = generate(brain, vocab, "hello echo", temp=0.5, length=40)

        extra = ""
        if hasattr(brain, 'avg_entropy'):
            ent = brain.avg_entropy()
            extra = f"  ent={ent:.3f}"
        if hasattr(brain, 'K_history') and brain.K_history:
            avg_k = sum(brain.K_history[-10:]) / min(10, len(brain.K_history))
            extra += f"  K={avg_k:.0f}"

        print(f"  {name:<20} loss={loss:.4f}  time={elapsed:.1f}s{extra}")
        print(f"    says: '{resp}'")

        results[name].append({
            'day': day_idx+1, 'loss': loss, 'time': elapsed,
            'response': resp, 'epochs': brain.epochs
        })

# ============================================================
# FINAL COMPARISON
# ============================================================
print()
print("=" * 80)
print("  FINAL RESULTS — 7 DAY COMPARISON")
print("=" * 80)
print()

# Summary table
print(f"  {'Strategy':<20} {'Loss':>7} {'Time':>7} {'Mem':>8} {'Overhead':>9} {'Quality':>8}")
print(f"  {'─'*20} {'─'*7} {'─'*7} {'─'*8} {'─'*9} {'─'*8}")

c_loss = results['Classical'][-1]['loss']
for name, brain in brains.items():
    loss = results[name][-1]['loss']
    total_time = timings[name]
    mem = brain.mem_bytes
    overhead = f"{brain.weight_count}" if name == 'Classical' else f"{brain.weight_count}"
    # Compute overhead ratio
    if name == 'Classical':
        oh = "1x"
        mem_str = f"{mem/1024:.1f}KB"
    elif name == 'Full Quantum':
        oh = "8x+4x"
        mem_str = f"{mem/1024:.1f}KB"
    elif name == 'Quantum Layers':
        oh = "4x+2x"
        mem_str = f"{mem/1024:.1f}KB"
    elif name == 'Quantum Dropout':
        oh = "1x"
        mem_str = f"{mem/1024:.1f}KB"
    elif name == 'Adaptive K':
        oh = "4-16x+4x"
        mem_str = f"{mem/1024:.1f}KB"

    improvement = ((c_loss - loss) / c_loss * 100) if c_loss > 0 else 0
    quality = f"{improvement:+.1f}%"
    print(f"  {name:<20} {loss:>7.4f} {total_time:>6.1f}s {mem_str:>8} {oh:>9} {quality:>8}")

print()

# Loss curves
print("  LOSS CURVES:")
max_loss = max(max(r['loss'] for r in results[n]) for n in results)
for name in brains:
    curve = results[name]
    final = curve[-1]['loss']
    bar = '█' * int(final / max_loss * 40)
    print(f"    {name:<20} {final:.4f} {bar}")
print()

# Day-by-day
print(f"  {'Day':<6}", end="")
for name in brains:
    print(f"  {name:>16}", end="")
print()
print(f"  {'─'*6}", end="")
for _ in brains:
    print(f"  {'─'*16}", end="")
print()
for d in range(7):
    print(f"  Day {d+1} ", end="")
    for name in brains:
        loss = results[name][d]['loss']
        print(f"  {loss:>16.4f}", end="")
    print()
print()

# Final responses
print("  FINAL RESPONSES (temp=0.5, seed='tell me about the ocean'):")
for name, brain in brains.items():
    resp = generate(brain, vocab, "tell me about the ocean", temp=0.5, length=50)
    print(f"    {name:<20} '{resp}'")
print()

# Time comparison
print("  TOTAL TRAINING TIME (7 days):")
for name in brains:
    t = timings[name]
    bar = '▓' * int(t / max(timings.values()) * 40)
    print(f"    {name:<20} {t:>6.1f}s {bar}")
print()

# Efficiency score (loss reduction per unit time)
print("  EFFICIENCY (loss reduction per second of training):")
c_final = results['Classical'][-1]['loss']
c_time = timings['Classical']
for name in brains:
    loss = results[name][-1]['loss']
    t = timings[name]
    reduction = c_final - loss
    efficiency = reduction / t if t > 0 else 0
    bar = '█' * int(max(0, efficiency) * 100)
    print(f"    {name:<20} {efficiency:>+.4f}/s {bar}")
print()

# Quantum-specific stats
print("  QUANTUM ENTROPY (superposition health):")
for name, brain in brains.items():
    if hasattr(brain, 'avg_entropy'):
        ent = brain.avg_entropy()
        bar_len = int(ent / 0.693 * 40)
        bar = '░' * bar_len
        print(f"    {name:<20} {ent:.4f} {bar} ({ent/0.693*100:.0f}%)")
    elif hasattr(brain, 'avg_entropy') and brain.avg_entropy() == 0:
        print(f"    {name:<20} N/A (classical)")
print()

# Adaptive K history
if 'Adaptive K' in brains:
    ak = brains['Adaptive K']
    if ak.K_history:
        print(f"  ADAPTIVE K HISTORY (samples per step over time):")
        # Sample every Nth
        step = max(1, len(ak.K_history) // 20)
        for i in range(0, len(ak.K_history), step):
            k = ak.K_history[i]
            bar = '█' * k
            print(f"    step {i:>5}: K={k:>2} {bar}")
        print(f"    final K: {ak.K}")
        print()

# Verdict
print("=" * 80)
print("  VERDICT")
print("=" * 80)
print()

# Find best by loss
best_loss_name = min(results, key=lambda n: results[n][-1]['loss'])
best_loss_val = results[best_loss_name][-1]['loss']

# Find best by efficiency
efficiencies = {}
for name in brains:
    reduction = c_final - results[name][-1]['loss']
    efficiencies[name] = reduction / timings[name] if timings[name] > 0 else 0
best_eff_name = max(efficiencies, key=efficiencies.get)

print(f"  Best loss:         {best_loss_name} ({best_loss_val:.4f})")
print(f"  Best efficiency:   {best_eff_name} ({efficiencies[best_eff_name]:+.4f}/s)")
print(f"  Fastest:           {min(timings, key=timings.get)} ({min(timings.values()):.1f}s)")
print(f"  Lowest memory:     Quantum Dropout (same as Classical)")
print()

# Recommendation
print("  RECOMMENDATION FOR ECHO:")
print()
all_losses = {n: results[n][-1]['loss'] for n in results}
q_dropout_loss = all_losses.get('Quantum Dropout', 0)
q_layers_loss = all_losses.get('Quantum Layers', 0)
full_q_loss = all_losses.get('Full Quantum', 0)
adaptive_loss = all_losses.get('Adaptive K', 0)

if q_layers_loss < full_q_loss:
    print("  → Quantum Layers beats Full Quantum at Echo's scale")
    print("    Layer-level superposition is more efficient than per-weight")
if q_dropout_loss < c_loss:
    print("  → Quantum Dropout improves on Classical with ZERO overhead")
    print("    Should be applied regardless of other choices")
if adaptive_loss < full_q_loss:
    print("  → Adaptive K outperforms fixed K")
    print("    Smart sampling allocation pays off")

print()
print("=" * 80)