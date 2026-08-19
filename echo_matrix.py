# ============================================================
# ECHO MATRIX — Pure Python matrix math, zero dependencies
# No NumPy. No SciPy. No nothing. Just lists and loops.
# ============================================================

import math
import random

# ----------------------------------------------------------
# Core matrix operations
# ----------------------------------------------------------

def zeros(rows, cols):
    """Create a rows x cols matrix filled with zeros."""
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def ones(rows, cols):
    """Create a rows x cols matrix filled with ones."""
    return [[1.0 for _ in range(cols)] for _ in range(rows)]

def random_matrix(rows, cols, scale=0.1):
    """Create a rows x cols matrix with random values."""
    return [[random.gauss(0, 1) * scale for _ in range(cols)] for _ in range(rows)]

def identity(n):
    """Create an n x n identity matrix."""
    m = zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m

def transpose(m):
    """Transpose a 2D list."""
    if not m:
        return []
    return [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]

def matmul(a, b):
    """Matrix multiply: a (p x q) * b (q x r) = c (p x r)."""
    if not a or not b:
        return []
    p = len(a)
    q = len(a[0])
    r = len(b[0])
    # Pre-allocate result
    c = zeros(p, r)
    for i in range(p):
        for k in range(q):
            aik = a[i][k]
            if aik == 0.0:
                continue
            for j in range(r):
                c[i][j] += aik * b[k][j]
    return c

def matmul_transpose_b(a, b):
    """Multiply a by b^T without computing the transpose explicitly.
    a is (p x q), b is (r x q), result is (p x r)."""
    p = len(a)
    q = len(a[0])
    r = len(b)
    c = zeros(p, r)
    for i in range(p):
        for j in range(r):
            s = 0.0
            for k in range(q):
                s += a[i][k] * b[j][k]
            c[i][j] = s
    return c

def matmul_transpose_a(a, b):
    """Multiply a^T by b without computing the transpose explicitly.
    a is (q x p), b is (q x r), result is (p x r)."""
    q = len(a)
    p = len(a[0])
    r = len(b[0])
    c = zeros(p, r)
    for k in range(q):
        for i in range(p):
            aki = a[k][i]
            if aki == 0.0:
                continue
            for j in range(r):
                c[i][j] += aki * b[k][j]
    return c

def add(a, b):
    """Element-wise addition."""
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def subtract(a, b):
    """Element-wise subtraction."""
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def scale(a, s):
    """Scale matrix by scalar."""
    return [[a[i][j] * s for j in range(len(a[0]))] for i in range(len(a))]

def hadamard(a, b):
    """Element-wise (Hadamard) product."""
    return [[a[i][j] * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

# ----------------------------------------------------------
# Vector operations (1D lists)
# ----------------------------------------------------------

def vec_zeros(n):
    return [0.0] * n

def vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]

def vec_scale(a, s):
    return [x * s for x in a]

def vec_dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))

def vec_to_matrix(v):
    """Convert 1D vector to 1 x n matrix."""
    return [list(v)]

def matrix_to_vec(m):
    """Convert 1 x n (or n x 1) matrix to 1D vector."""
    if len(m) == 1:
        return list(m[0])
    return [m[i][0] for i in range(len(m))]

# ----------------------------------------------------------
# Activation functions and their derivatives
# ----------------------------------------------------------

def tanh(x):
    """Hyperbolic tangent — squash to [-1, 1]."""
    if x > 20:
        return 1.0
    if x < -20:
        return -1.0
    e2x = math.exp(2 * x)
    return (e2x - 1) / (e2x + 1)

def tanh_deriv(x):
    """Derivative of tanh: 1 - tanh^2."""
    t = tanh(x)
    return 1.0 - t * t

def softmax(vec):
    """Numerically stable softmax."""
    if not vec:
        return []
    max_val = max(vec)
    exps = [math.exp(v - max_val) for v in vec]
    s = sum(exps)
    return [e / s for e in exps]

def softmax_matrix(m):
    """Apply softmax to each row of a matrix (treating each row as a vector)."""
    return [softmax(row) for row in m]

# ----------------------------------------------------------
# Loss functions
# ----------------------------------------------------------

def cross_entropy_loss(probs, target_idx):
    """Cross-entropy loss for a single prediction.
    probs: list of probabilities, target_idx: correct class index."""
    p = max(probs[target_idx], 1e-12)
    return -math.log(p)

# ----------------------------------------------------------
# Utility
# ----------------------------------------------------------

def argmax(vec):
    """Return index of maximum value."""
    best = 0
    best_val = vec[0]
    for i in range(1, len(vec)):
        if vec[i] > best_val:
            best_val = vec[i]
            best = i
    return best

def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

def clip_gradients(weights, threshold=5.0):
    """Global gradient clipping to prevent exploding gradients."""
    total_sq = 0.0
    for w in weights:
        for row in w:
            for val in row:
                total_sq += val * val
    norm = math.sqrt(total_sq)
    if norm > threshold:
        scale_factor = threshold / (norm + 1e-8)
        for w in weights:
            for row in w:
                for j in range(len(row)):
                    row[j] *= scale_factor

def matrix_to_str(m):
    """Pretty print a matrix (for debugging)."""
    lines = []
    for row in m:
        lines.append("  [" + ", ".join(f"{v:8.4f}" for v in row) + "]")
    return "[\n" + "\n".join(lines) + "\n]"

# ----------------------------------------------------------
# Self-test (run directly to verify)
# ----------------------------------------------------------

if __name__ == "__main__":
    print("=== ECHO MATRIX SELF-TEST ===\n")

    # Test matmul
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    c = matmul(a, b)
    print(f"matmul([[1,2],[3,4]] x [[5,6],[7,8]]) = {c}")
    assert c == [[19, 22], [43, 50]], "matmul FAILED"
    print("  PASS\n")

    # Test transpose
    t = transpose(a)
    print(f"transpose([[1,2],[3,4]]) = {t}")
    assert t == [[1, 3], [2, 4]], "transpose FAILED"
    print("  PASS\n")

    # Test tanh
    print(f"tanh(0) = {tanh(0):.6f} (should be 0.0)")
    assert abs(tanh(0)) < 1e-10, "tanh(0) FAILED"
    print("  PASS\n")

    # Test softmax
    s = softmax([1.0, 2.0, 3.0])
    print(f"softmax([1,2,3]) = [{s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}]")
    assert abs(sum(s) - 1.0) < 1e-10, "softmax sum FAILED"
    print("  PASS\n")

    # Test random matrix
    r = random_matrix(3, 4, 0.1)
    print(f"random_matrix(3,4) shape: {len(r)}x{len(r[0])}")
    assert len(r) == 3 and len(r[0]) == 4, "random_matrix FAILED"
    print("  PASS\n")

    # Test argmax
    print(f"argmax([0.1, 0.9, 0.2]) = {argmax([0.1, 0.9, 0.2])} (should be 1)")
    assert argmax([0.1, 0.9, 0.2]) == 1, "argmax FAILED"
    print("  PASS\n")

    print("=== ALL TESTS PASSED ===")