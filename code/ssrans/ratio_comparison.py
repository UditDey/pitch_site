"""
Roundtrip correctness test and compression ratio comparison for:
  1. Vanilla rANS (with exact frequency tables from data histogram)
  2. SSrANS (rANS with sparsity probability model instead of tables)
  3. Bitmap (bitmap + packed nonzero values)

Generates random sparse uint8 data, compresses/decompresses through all 3,
asserts exact roundtrip, and compares compressed sizes against Shannon entropy.
"""

import math
import random
from collections import Counter

# ============================================================
# Constants
# ============================================================
LOG_M = 12
M = 1 << LOG_M          # 4096 — probability resolution
RANS_L = 1 << 23        # Lower bound of rANS state range
ALPHABET = 256           # uint8 symbol space


# ============================================================
# Data generation
# ============================================================
def generate_sparse_data(length, sparsity, seed=42):
    """Generate random uint8 data where `sparsity` fraction of symbols are 0."""
    rng = random.Random(seed)
    data = []
    for _ in range(length):
        if rng.random() < sparsity:
            data.append(0)
        else:
            data.append(rng.randint(1, 255))
    return data


# ============================================================
# Shannon entropy (theoretical lower bound)
# ============================================================
def shannon_entropy_bps(data):
    """Returns Shannon entropy in bits per symbol."""
    counts = Counter(data)
    total = len(data)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


# ============================================================
# Frequency table builders
# ============================================================
def build_vanilla_tables(data):
    """Build exact frequency tables from data histogram. Only symbols that
    appear get freq >= 1. Absent symbols get freq = 0."""
    counts = Counter(data)
    total = len(data)

    # Initial proportional assignment, minimum 1 per present symbol
    freq = [0] * ALPHABET
    for s in range(ALPHABET):
        if counts.get(s, 0) > 0:
            freq[s] = max(1, round(counts[s] * M / total))

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


def build_ssrans_tables(sparsity):
    """Build frequency tables from the SSrANS probability model.

    Model: freq[0] = F_zero, freq[s] = F_nonzero for s in 1..255
    where F_zero models the elevated frequency of the zero symbol
    and F_nonzero is uniform across all other symbols.

    From the article:
        F_nonzero = M * (1 - S) / (N - 1)
        F_zero    = M - (N - 1) * F_nonzero
    """
    N = ALPHABET
    S = sparsity

    # Round F_nonzero, enforce minimum 1
    F_nonzero = max(1, round(M * (1 - S) / (N - 1)))

    # Derive F_zero from constraint: M = F_zero + (N-1) * F_nonzero
    F_zero = M - (N - 1) * F_nonzero

    # If F_zero collapsed, fix it
    if F_zero < 1:
        F_zero = 1
        F_nonzero = (M - F_zero) // (N - 1)
        F_zero = M - (N - 1) * F_nonzero

    assert F_zero >= 1 and F_nonzero >= 1
    assert F_zero + (N - 1) * F_nonzero == M

    # Build standard rANS tables from the model
    freq = [F_nonzero] * ALPHABET
    freq[0] = F_zero

    cfreq = [0] * ALPHABET
    for s in range(1, ALPHABET):
        cfreq[s] = cfreq[s - 1] + freq[s - 1]

    inv_cfreq = [0] * M
    for s in range(ALPHABET):
        for i in range(freq[s]):
            inv_cfreq[cfreq[s] + i] = s

    # Also verify the inline formulas match the tables
    for s in range(ALPHABET):
        assert freq[s] == (F_zero if s == 0 else F_nonzero)
        expected_cf = 0 if s == 0 else F_zero + (s - 1) * F_nonzero
        assert cfreq[s] == expected_cf, f"cfreq mismatch at s={s}: {cfreq[s]} != {expected_cf}"

    for slot in range(M):
        if slot < F_zero:
            expected_sym = 0
        else:
            expected_sym = 1 + (slot - F_zero) // F_nonzero
        assert inv_cfreq[slot] == expected_sym, \
            f"inv_cfreq mismatch at slot={slot}: {inv_cfreq[slot]} != {expected_sym}"

    return freq, cfreq, inv_cfreq, F_zero, F_nonzero


# ============================================================
# rANS encoder / decoder (used by both vanilla and SSrANS)
# ============================================================
def rans_encode(data, freq, cfreq):
    """Encode data into a byte stream using rANS. Processes symbols in reverse."""
    state = RANS_L
    stream = []  # output byte stream (built in encode order, reversed for decode)

    for s in reversed(data):
        f = freq[s]
        cf = cfreq[s]
        assert f > 0, f"Cannot encode symbol {s} with freq=0"

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


def rans_decode(stream, freq, cfreq, inv_cfreq, n_symbols):
    """Decode n_symbols from a byte stream using rANS."""
    stream = list(stream)  # copy, we'll pop from the end

    # Recover initial state (4 bytes)
    state = 0
    for _ in range(4):
        state = (state << 8) | stream.pop()

    decoded = []
    for _ in range(n_symbols):
        # Core rANS decode step
        slot = state & (M - 1)           # state % M (M is power of 2)
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
# Bitmap encoder / decoder
# ============================================================
def bitmap_compress(data):
    """Compress sparse data into (bitmap, dense_values)."""
    bitmap = []
    dense = []
    for val in data:
        if val != 0:
            bitmap.append(1)
            dense.append(val)
        else:
            bitmap.append(0)
    return bitmap, dense


def bitmap_decompress(bitmap, dense):
    """Decompress (bitmap, dense_values) back to original data."""
    dense_idx = 0
    result = []
    for bit in bitmap:
        if bit:
            result.append(dense[dense_idx])
            dense_idx += 1
        else:
            result.append(0)
    assert dense_idx == len(dense), "Dense array not fully consumed"
    return result


def bitmap_compressed_bytes(bitmap, dense, element_bytes=1):
    """Total compressed size in bytes: packed bitmap + dense values."""
    bitmap_bytes = math.ceil(len(bitmap) / 8)
    dense_bytes = len(dense) * element_bytes
    return bitmap_bytes + dense_bytes


# ============================================================
# Test harness
# ============================================================
def test_all(length, sparsity):
    print(f"{'=' * 70}")
    print(f"  Data length: {length}  |  Target sparsity: {sparsity}")
    print(f"{'=' * 70}")

    data = generate_sparse_data(length, sparsity)
    actual_sparsity = data.count(0) / len(data)
    n_unique = len(set(data))

    entropy_bps = shannon_entropy_bps(data)
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
    v_freq, v_cfreq, v_inv_cfreq = build_vanilla_tables(data)
    v_stream = rans_encode(data, v_freq, v_cfreq)
    v_decoded = rans_decode(v_stream, v_freq, v_cfreq, v_inv_cfreq, length)
    assert v_decoded == data, "Vanilla rANS roundtrip FAILED!"
    v_bytes = len(v_stream)
    print(f"  Vanilla rANS:  {v_bytes:>6} bytes  |  "
          f"ratio {original_bytes / v_bytes:.3f}x  |  "
          f"overhead vs entropy {(v_bytes - entropy_bytes) / entropy_bytes * 100:+.1f}%  |  PASS")

    # --- SSrANS ---
    ss_freq, ss_cfreq, ss_inv_cfreq, F_zero, F_nonzero = build_ssrans_tables(sparsity)
    ss_stream = rans_encode(data, ss_freq, ss_cfreq)
    ss_decoded = rans_decode(ss_stream, ss_freq, ss_cfreq, ss_inv_cfreq, length)
    assert ss_decoded == data, "SSrANS roundtrip FAILED!"
    ss_bytes = len(ss_stream)
    print(f"  SSrANS:        {ss_bytes:>6} bytes  |  "
          f"ratio {original_bytes / ss_bytes:.3f}x  |  "
          f"overhead vs entropy {(ss_bytes - entropy_bytes) / entropy_bytes * 100:+.1f}%  |  PASS")
    print(f"    (F_zero={F_zero}, F_nonzero={F_nonzero}, "
          f"model_sparsity={F_zero / M:.4f})")

    # --- Bitmap ---
    bm_bitmap, bm_dense = bitmap_compress(data)
    bm_decoded = bitmap_decompress(bm_bitmap, bm_dense)
    assert bm_decoded == data, "Bitmap roundtrip FAILED!"
    bm_bytes = bitmap_compressed_bytes(bm_bitmap, bm_dense)
    print(f"  Bitmap:        {bm_bytes:>6} bytes  |  "
          f"ratio {original_bytes / bm_bytes:.3f}x  |  "
          f"overhead vs entropy {(bm_bytes - entropy_bytes) / entropy_bytes * 100:+.1f}%  |  PASS")
    print(f"    (bitmap={math.ceil(len(bm_bitmap) / 8)} bytes, "
          f"dense={len(bm_dense)} bytes)")

    print()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("rANS / SSrANS / Bitmap — Roundtrip Correctness & Compression Ratio")
    print(f"Parameters: LOG_M={LOG_M}, M={M}, RANS_L=2^{int(math.log2(RANS_L))}, "
          f"ALPHABET={ALPHABET}")
    print()

    for sparsity in [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]:
        test_all(length=10000, sparsity=sparsity)