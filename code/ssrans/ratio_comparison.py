import random
import math
import argparse
import numpy as np
from collections import Counter

# ============================================================
# rANS Constants
# ============================================================
LOG_M = 12
M = 1 << LOG_M          # 4096 — probability resolution
RANS_L = 1 << 23        # Lower bound of rANS state range
ALPHABET = 256           # uint8 symbol space


# ============================================================
# Data Generation
# ============================================================
def gen_data(length, sparsity, seed=42):
    rng = random.Random(seed)
    data = []
    for _ in range(length):
        if rng.random() < sparsity:
            data.append(0)
        else:
            data.append(rng.randint(1, 255))
    return data


# ============================================================
# Calculate Shannon Entropy (bits per symbol)
# ============================================================
def shannon(data):
    histogram = Counter(data)
    entropy = 0.0
    for count in histogram.values():
        p = count / len(data)
        entropy += -p * math.log2(p)
    return entropy


# ============================================================
# Build Vanilla rANS Frequency Tables
# ============================================================
def build_vanilla_tables(data):
    histogram = Counter(data)

    freq = [0] * ALPHABET
    for s in range(ALPHABET):
        if histogram.get(s, 0) != 0:
            freq[s] = max(1, round(histogram[s] * M / len(data)))

    # Adjust most-frequent symbol so sum == M
    current_sum = sum(freq)
    most_freq_sym = max(range(ALPHABET), key=lambda s: freq[s])
    freq[most_freq_sym] += M - current_sum

    assert freq[most_freq_sym] >= 1, "M too small for this data"
    assert sum(freq) == M

    # Cumulative frequencies
    cfreq = [0] * ALPHABET
    for s in range(1, ALPHABET):
        cfreq[s] = cfreq[s - 1] + freq[s - 1]

    # Inverse cumulative frequency table (slot -> symbol)
    inv_cfreq = [0] * M
    for s in range(ALPHABET):
        for i in range(freq[s]):
            inv_cfreq[cfreq[s] + i] = s

    return freq, cfreq, inv_cfreq


# ============================================================
# Vanilla rANS Codec
# ============================================================
def vanilla_rans_encode(data, freq, cfreq):
    state = RANS_L
    stream = []

    for s in reversed(data):
        f = freq[s]
        cf = cfreq[s]
        assert f > 0

        # Renormalize: push bytes until state is small enough
        x_max = ((RANS_L >> LOG_M) << 8) * f
        while state >= x_max:
            stream.append(state & 0xFF)
            state >>= 8
        
        # Core rANS encode step
        state = (state // f) * M + cf + (state % f)
    
    # Flush final state (4 bytes, big-endian push)
    for _ in range(4):
        stream.append(state & 0xFF)
        state >>= 8

    return stream

def vanilla_rans_decode(stream, freq, cfreq, inv_cfreq, n_symbols):
    stream = list(stream)

    state = 0
    for _ in range(4):
        state = (state << 8) | stream.pop()
    
    decoded = []
    for _ in range(n_symbols):
        # Core rANS decode step
        slot = state & (M - 1) # state % M (M is power of 2)
        s = inv_cfreq[slot]
        f = freq[s]
        cf = cfreq[s]
        state = (state >> LOG_M) * f + slot - cf

        # Renormalize: pull bytes until state >= RANS_L
        while state < RANS_L:
            state = (state << 8) | stream.pop()

        decoded.append(s)

    return decoded


# ============================================================
# SSrANS Probability Model
# ============================================================
def round_ssrans_params(M, N, S):
    F_nonzero = max(1, round(M * (1 - S) / (N - 1)))
    F_zero = M - (N - 1) * F_nonzero
    if F_zero < 1:
        F_zero = 1
        F_nonzero = (M - F_zero) // (N - 1)
        F_zero = M - (N - 1) * F_nonzero
    return F_zero, F_nonzero


def ssrans_freq(s, F_zero, F_nonzero):
    return F_zero if s == 0 else F_nonzero

def ssrans_cfreq(s, F_zero, F_nonzero):
    if s == 0:
        return 0
    return F_zero + (s - 1) * F_nonzero

def ssrans_inv_cfreq(slot, F_zero, F_nonzero):
    if slot < F_zero:
        return 0
    return 1 + (slot - F_zero) // F_nonzero


# ============================================================
# SSrANS Codec
# ============================================================
def ssrans_encode(data, F_zero, F_nonzero):
    state = RANS_L
    stream = []

    for s in reversed(data):
        # Key difference between SSrANS and vanilla rANS!!
        f = ssrans_freq(s, F_zero, F_nonzero)
        cf = ssrans_cfreq(s, F_zero, F_nonzero)
        assert f > 0

        # Renormalize: push bytes until state is small enough
        x_max = ((RANS_L >> LOG_M) << 8) * f
        while state >= x_max:
            stream.append(state & 0xFF)
            state >>= 8

        # Core rANS encode step
        state = (state // f) * M + cf + (state % f)

    for _ in range(4):
        stream.append(state & 0xFF)
        state >>= 8

    return stream

def ssrans_decode(stream, F_zero, F_nonzero, n_symbols):
    stream = list(stream)

    state = 0
    for _ in range(4):
        state = (state << 8) | stream.pop()

    decoded = []
    for _ in range(n_symbols):
        slot = state & (M - 1)
        s = ssrans_inv_cfreq(slot, F_zero, F_nonzero)
        f = ssrans_freq(s, F_zero, F_nonzero)
        cf = ssrans_cfreq(s, F_zero, F_nonzero)
        state = (state >> LOG_M) * f + slot - cf

        while state < RANS_L:
            state = (state << 8) | stream.pop()

        decoded.append(s)

    return decoded


# ============================================================
# Bitmap Codec
# ============================================================
def bitmap_encode(data):
    bitmap = []
    dense = []
    for val in data:
        if val != 0:
            bitmap.append(1)
            dense.append(val)
        else:
            bitmap.append(0)
    return bitmap, dense

def bitmap_decode(bitmap, dense):
    dense_idx = 0
    result = []
    for bit in bitmap:
        if bit:
            result.append(dense[dense_idx])
            dense_idx += 1
        else:
            result.append(0)
    assert dense_idx == len(dense)
    return result


# ============================================================
# Test harness
# ============================================================
def test_all(length, sparsity):
    print(f"{'=' * 70}")
    print(f"  Data length: {length}  |  Target sparsity: {sparsity}")
    print(f"{'=' * 70}")

    data = gen_data(length, sparsity)
    actual_sparsity = data.count(0) / len(data)
    n_unique = len(set(data))

    entropy_bps = shannon(data)
    original_bytes = length  # 1 byte per symbol
    entropy_bytes = entropy_bps * length / 8

    print(f"  Actual sparsity: {actual_sparsity:.4f}")
    print(f"  Unique symbols:  {n_unique}")
    print(f"  Shannon entropy: {entropy_bps:.4f} bits/symbol")
    print(f"  Original size:   {original_bytes} bytes")
    print(f"  Entropy limit:   {entropy_bytes:.1f} bytes  "
          f"(ratio {original_bytes / entropy_bytes:.3f}x)")
    print()

    # --- Vanilla rANS ---
    freq, cfreq, inv_cfreq = build_vanilla_tables(data)
    stream = vanilla_rans_encode(data, freq, cfreq)
    decoded = vanilla_rans_decode(stream, freq, cfreq, inv_cfreq, length)
    assert decoded == data, "Vanilla rANS roundtrip FAILED!"
    byte_count = len(stream)
    print(f"  Vanilla rANS:  {byte_count:>6} bytes  |  "
          f"ratio {original_bytes / byte_count:.3f}x  |  "
          f"overhead vs entropy {(byte_count - entropy_bytes) / entropy_bytes * 100:+.1f}%  |  PASS")

    # --- SSrANS (NEW) ---
    F_zero, F_nonzero = round_ssrans_params(M, ALPHABET, sparsity)
    stream_ss = ssrans_encode(data, F_zero, F_nonzero)
    decoded_ss = ssrans_decode(stream_ss, F_zero, F_nonzero, length)
    assert decoded_ss == data, "SSrANS roundtrip FAILED!"
    byte_count_ss = len(stream_ss)
    print(f"  SSrANS:        {byte_count_ss:>6} bytes  |  "
          f"ratio {original_bytes / byte_count_ss:.3f}x  |  "
          f"overhead vs entropy {(byte_count_ss - entropy_bytes) / entropy_bytes * 100:+.1f}%  |  PASS")

    # --- Bitmap ---
    bitmap, dense = bitmap_encode(data)
    decoded = bitmap_decode(bitmap, dense)
    assert decoded == data, "Bitmap roundtrip FAILED!"
    bm_bytes = math.ceil(len(bitmap) / 8) + len(dense)
    print(f"  Bitmap:        {bm_bytes:>6} bytes  |  "
          f"ratio {original_bytes / bm_bytes:.3f}x  |  "
          f"overhead vs entropy {(bm_bytes - entropy_bytes) / entropy_bytes * 100:+.1f}%  |  PASS")
    print(f"    (bitmap={math.ceil(len(bitmap) / 8)} bytes, "
          f"dense={len(dense)} bytes)")

    print()


def sweep_ratios(length, sparsities):
    results = {"entropy": [], "vanilla": [], "ssrans": [], "bitmap": []}

    for sparsity in sparsities:
        data = gen_data(length, sparsity)
        original_bytes = length

        entropy_bps = shannon(data)
        entropy_bytes = entropy_bps * length / 8
        results["entropy"].append(original_bytes / entropy_bytes)

        # Vanilla rANS
        freq, cfreq, inv_cfreq = build_vanilla_tables(data)
        stream = vanilla_rans_encode(data, freq, cfreq)
        results["vanilla"].append(original_bytes / len(stream))

        # SSrANS
        F_zero, F_nonzero = round_ssrans_params(M, ALPHABET, sparsity)
        stream_ss = ssrans_encode(data, F_zero, F_nonzero)
        results["ssrans"].append(original_bytes / len(stream_ss))

        # Bitmap
        bitmap, dense = bitmap_encode(data)
        bm_bytes = math.ceil(len(bitmap) / 8) + len(dense)
        results["bitmap"].append(original_bytes / bm_bytes)

    return results


def plot_sweep(length=10000):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    sparsities = np.arange(0.40, 0.991, 0.01)
    results = sweep_ratios(length, sparsities)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sparsities, results["vanilla"], "-", color="#1f77b4",
            linewidth=1.5, label="Vanilla rANS")
    ax.plot(sparsities, results["ssrans"], "-", color="#2ca02c",
            linewidth=1.5, label="SSrANS")
    ax.plot(sparsities, results["bitmap"], "-", color="#d62728",
            linewidth=1.5, label="Bitmap")

    ax.set_yscale("log", base=2)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.set_xlabel("Sparsity (%)", fontsize=12)
    ax.set_ylabel("Compression Ratio", fontsize=12)
    ax.set_title("Compression Ratio vs Sparsity", fontsize=14)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("compression_ratio_sweep.png", dpi=180)
    print("Plot saved to compression_ratio_sweep.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rANS / SSrANS / Bitmap benchmark")
    parser.add_argument("--plot", action="store_true", help="Plot compression ratio sweep curve")
    args = parser.parse_args()

    print("rANS / SSrANS / Bitmap — Roundtrip Correctness & Compression Ratio")
    print(f"Parameters: LOG_M={LOG_M}, M={M}, RANS_L=2^{int(math.log2(RANS_L))}, "
          f"ALPHABET={ALPHABET}\n")
    
    for sparsity in [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]:
        test_all(length=10000, sparsity=sparsity)

    if args.plot:
        plot_sweep()
