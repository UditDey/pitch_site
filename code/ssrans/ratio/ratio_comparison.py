"""
Compression Ratio Comparison: Bitmap vs Vanilla rANS vs SSrANS
Generates line chart of bits-per-symbol across sparsity levels.
8-bit data (N=256).
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter

# ============================================================
# rANS Parameters
# ============================================================
PROB_BITS = 12
M = 1 << PROB_BITS  # 4096
RANS_L = 1 << 23
N = 256  # 8-bit symbols


# ============================================================
# Data Generation
# ============================================================
def generate_sparse_data(length, sparsity_frac, seed=42):
    rng = np.random.default_rng(seed)
    data = np.zeros(length, dtype=np.int32)
    num_nonzero = int(length * (1 - sparsity_frac))
    if num_nonzero > 0:
        indices = rng.choice(length, num_nonzero, replace=False)
        data[indices] = rng.integers(1, N, size=num_nonzero)
    return data


# ============================================================
# rANS Encode / Decode
# ============================================================
def rans_encode(data, freq_fn, cfreq_fn):
    output = []
    x = RANS_L
    for s in reversed(data):
        f = freq_fn(s)
        c = cfreq_fn(s)
        x_max = ((RANS_L >> PROB_BITS) << 8) * f
        while x >= x_max:
            output.append(x & 0xFF)
            x >>= 8
        x = (x // f) * M + c + (x % f)
    for _ in range(4):
        output.append(x & 0xFF)
        x >>= 8
    return output


def rans_decode(encoded, length, freq_fn, cfreq_fn, inv_cfreq_fn):
    enc = list(reversed(encoded))
    idx = 0
    x = 0
    for _ in range(4):
        x = (x << 8) | enc[idx]
        idx += 1
    result = []
    for _ in range(length):
        slot = x & (M - 1)
        s = inv_cfreq_fn(slot)
        f = freq_fn(s)
        c = cfreq_fn(s)
        x = (x >> PROB_BITS) * f + slot - c
        while x < RANS_L and idx < len(enc):
            x = (x << 8) | enc[idx]
            idx += 1
        result.append(s)
    return result


# ============================================================
# Vanilla rANS: exact frequency tables
# ============================================================
def build_freq_table(data):
    counts = Counter(int(s) for s in data)
    total = len(data)
    freq = {}
    for s in range(N):
        freq[s] = max(1, round(counts.get(s, 0) / total * M))
    current_sum = sum(freq.values())
    most_freq_sym = max(freq, key=freq.get)
    freq[most_freq_sym] -= (current_sum - M)
    assert freq[most_freq_sym] >= 1
    return freq


def build_cfreq_table(freq):
    cfreq = {}
    c = 0
    for s in range(N):
        cfreq[s] = c
        c += freq[s]
    return cfreq


def build_inv_cfreq_table(freq, cfreq):
    inv = [0] * M
    for s in range(N):
        for i in range(cfreq[s], cfreq[s] + freq[s]):
            inv[i] = s
    return inv


# ============================================================
# SSrANS: sparsity probability model
# ============================================================
def compute_ssrans_params(data):
    zero_count = int(np.sum(data == 0))
    nonzero_count = len(data) - zero_count
    if nonzero_count == 0:
        return M, 0
    S = zero_count / nonzero_count
    F_nonzero = max(1, round(M / ((N - 1) * (S + 1))))
    F_zero = M - (N - 1) * F_nonzero
    if F_zero < 1:
        F_zero = 1
        F_nonzero = (M - 1) // (N - 1)
        F_zero = M - (N - 1) * F_nonzero
    return F_zero, F_nonzero


def make_ssrans_functions(F_zero, F_nonzero):
    def freq(s):
        return F_zero if s == 0 else F_nonzero

    def cfreq(s):
        return 0 if s == 0 else F_zero + (s - 1) * F_nonzero

    def inv_cfreq(slot):
        return 0 if slot < F_zero else 1 + (slot - F_zero) // F_nonzero

    return freq, cfreq, inv_cfreq


# ============================================================
# Bitmap & Entropy
# ============================================================
def bitmap_bps(data):
    n = len(data)
    num_nonzero = int(np.count_nonzero(data))
    return (n + num_nonzero * 8) / n


def shannon_entropy(data):
    counts = Counter(int(s) for s in data)
    total = len(data)
    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            H -= p * math.log2(p)
    return H


# ============================================================
# Experiment
# ============================================================
LENGTH = 50_000  # shorter for speed across many sparsity levels

sparsity_levels = np.arange(0.40, 1.00, 0.02)

results = {'sparsity': [], 'entropy': [], 'bitmap': [], 'vanilla': [], 'ssrans': []}

print(f"Running experiment: {len(sparsity_levels)} sparsity levels, {LENGTH} symbols each (N={N})")
print(f"{'Sparsity':>10} {'Entropy':>10} {'Bitmap':>10} {'V-rANS':>10} {'SSrANS':>10}")
print("-" * 55)

for sp in sparsity_levels:
    data = generate_sparse_data(LENGTH, sp)

    H = shannon_entropy(data)
    bmp = bitmap_bps(data)

    # Vanilla rANS
    ft = build_freq_table(data)
    ct = build_cfreq_table(ft)
    it = build_inv_cfreq_table(ft, ct)
    ve = rans_encode(data, lambda s, ft=ft: ft[s], lambda s, ct=ct: ct[s])
    v_bps = len(ve) * 8 / LENGTH

    # Verify roundtrip
    vd = rans_decode(ve, LENGTH, lambda s, ft=ft: ft[s], lambda s, ct=ct: ct[s], lambda sl, it=it: it[sl])
    assert vd == list(data), f"Vanilla rANS mismatch at sparsity {sp:.0%}"

    # SSrANS
    Fz, Fnz = compute_ssrans_params(data)
    sf, sc, si = make_ssrans_functions(Fz, Fnz)
    se = rans_encode(data, sf, sc)
    ss_bps = len(se) * 8 / LENGTH

    sd = rans_decode(se, LENGTH, sf, sc, si)
    assert sd == list(data), f"SSrANS mismatch at sparsity {sp:.0%}"

    results['sparsity'].append(sp * 100)
    results['entropy'].append(H)
    results['bitmap'].append(bmp)
    results['vanilla'].append(v_bps)
    results['ssrans'].append(ss_bps)

    print(f"{sp:>10.0%} {H:>10.3f} {bmp:>10.3f} {v_bps:>10.3f} {ss_bps:>10.3f}")

print("\nAll roundtrip checks passed. Generating chart...")

# ============================================================
# Plot
# ============================================================
sp = results['sparsity']
ratio_bitmap  = [8 / b for b in results['bitmap']]
ratio_vanilla = [8 / v for v in results['vanilla']]
ratio_ssrans  = [8 / s for s in results['ssrans']]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(sp, ratio_bitmap,  color='#e74c3c', linewidth=2.2, label='Bitmap')
ax.plot(sp, ratio_vanilla, color='#3498db', linewidth=2.2, label='Vanilla rANS')
ax.plot(sp, ratio_ssrans,  color='#2ecc71', linewidth=2.2, label='SSrANS')

ax.set_xlabel('Sparsity (%)', fontsize=13)
ax.set_ylabel('Compression Ratio (×)', fontsize=13)
ax.set_title('Compression Ratio (8-bit data, N=256)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.set_xlim(40, 100)
ax.set_yscale('log', base=2)
ax.set_ylim(1, None)
import matplotlib.ticker as ticker
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.0f}×'))
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/ssrans_compression_ratio.png', dpi=150)
print("Chart saved.")