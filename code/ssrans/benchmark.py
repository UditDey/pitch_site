"""
TPU Microbenchmark: SSrANS vs Bitmap vs Vanilla rANS

Compares three decompression methods on a sparse probability model
(80% sparsity, alphabet=256, M=4096). All methods produce the same
number of output symbols per run, enabling direct comparison.

  SSrANS:  S parallel streams, each decoding TOTAL_SYMBOLS/S symbols
           via pure arithmetic (no table lookups).

  Bitmap:  128 packed uint32 words decoded via popcount, prefix-sum,
           and data-dependent gather from a dense value array.

  vRaNS:   S parallel streams using table-driven rANS decode
           (3 table lookups per symbol).

The key result: SSrANS throughput scales with stream count S.
Bitmap throughput is fixed (gather-bound). Crossover at S ≈ 8.
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
import functools
import timeit

# ============================================================
# Model parameters
# ============================================================
LOG_M    = 12
M        = 1 << LOG_M       # 4096
ALPHABET = 256
SPARSITY = 0.8

F_NONZERO = max(1, round(M * (1 - SPARSITY) / (ALPHABET - 1)))
F_ZERO    = M - (ALPHABET - 1) * F_NONZERO
assert F_ZERO >= 1 and F_NONZERO >= 1
assert F_ZERO + (ALPHABET - 1) * F_NONZERO == M

# Vanilla rANS lookup tables
_freq  = [F_NONZERO] * ALPHABET; _freq[0] = F_ZERO
_cfreq = [0] * ALPHABET
for _s in range(1, ALPHABET):
    _cfreq[_s] = _cfreq[_s - 1] + _freq[_s - 1]
_inv_cfreq = [0] * M
for _s in range(ALPHABET):
    for _i in range(_freq[_s]):
        _inv_cfreq[_cfreq[_s] + _i] = _s

FREQ_TABLE      = jnp.array(_freq, dtype=jnp.int32)
CFREQ_TABLE     = jnp.array(_cfreq, dtype=jnp.int32)
INV_CFREQ_TABLE = jnp.array(_inv_cfreq, dtype=jnp.int32)

# ============================================================
# Benchmark parameters
# ============================================================
TOTAL_SYMBOLS = 4096
BITS_PER_WORD = 32
N_WORDS       = TOTAL_SYMBOLS // BITS_PER_WORD  # 128
N_RUNS        = 200
N_WARMUP      = 3
N_TRIALS      = 5
STREAM_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

# Precomputed bit masks (uint32 to avoid signed overflow at bit 31)
BIT_POSITIONS = jnp.array([1 << j for j in range(BITS_PER_WORD)], dtype=jnp.uint32)
MASKS_BELOW   = jnp.array([(1 << j) - 1 for j in range(BITS_PER_WORD)], dtype=jnp.uint32)


# ============================================================
# Kernels
# ============================================================
def popcount32(x):
    return lax.population_count(x.astype(jnp.uint32)).astype(jnp.int32)


@functools.partial(jax.jit, static_argnums=(1, 2))
def bench_ssrans(init_states, n_symbols_per_stream, n_runs):
    """SSrANS: table-free decode via arithmetic on sparse model."""
    def decode_one(carry, _):
        state, cksum = carry
        slot = state & (M - 1)
        is_zero = slot < F_ZERO
        sym = jnp.where(is_zero, 0, 1 + (slot - F_ZERO) // F_NONZERO)
        f   = jnp.where(is_zero, F_ZERO, F_NONZERO)
        cf  = jnp.where(is_zero, jnp.int32(0), F_ZERO + (sym - 1) * F_NONZERO)
        new_state = (state >> LOG_M) * f + slot - cf
        return (new_state, cksum ^ sym), None

    def one_run(carry, _):
        init_state, cksum = carry
        (final_state, cksum), _ = lax.scan(
            decode_one, (init_state, cksum), None, length=n_symbols_per_stream)
        next_init = init_state ^ (final_state & 0xFF)
        return (next_init, cksum), None

    (_, cksum), _ = lax.scan(
        one_run, (init_states, jnp.zeros_like(init_states)), None, length=n_runs)
    return cksum.sum()


@functools.partial(jax.jit, static_argnums=(4, 5))
def bench_vanilla_rans(init_states, freq_table, cfreq_table, inv_cfreq_table,
                       n_symbols_per_stream, n_runs):
    """Vanilla rANS: table-driven decode (3 lookups per symbol)."""
    def decode_one(carry, _):
        state, cksum = carry
        slot = state & (M - 1)
        sym = inv_cfreq_table[slot]
        f   = freq_table[sym]
        cf  = cfreq_table[sym]
        new_state = (state >> LOG_M) * f + slot - cf
        return (new_state, cksum ^ sym), None

    def one_run(carry, _):
        init_state, cksum = carry
        (final_state, cksum), _ = lax.scan(
            decode_one, (init_state, cksum), None, length=n_symbols_per_stream)
        next_init = init_state ^ (final_state & 0xFF)
        return (next_init, cksum), None

    (_, cksum), _ = lax.scan(
        one_run, (init_states, jnp.zeros_like(init_states)), None, length=n_runs)
    return cksum.sum()


@functools.partial(jax.jit, static_argnums=(3,))
def bench_bitmap(packed_bitmap, dense_vals, perturbations, n_runs):
    """Packed bitmap: popcount + prefix-sum + gather via (128,32) broadcast."""
    def one_run(cksum, perturb):
        bm = packed_bitmap ^ (perturb.astype(jnp.uint32) & jnp.uint32(1))

        # Popcount each word, exclusive prefix sum for word offsets
        counts = popcount32(bm)
        word_offsets = jnp.cumsum(counts) - counts

        # Broadcast to (N_WORDS, 32) and decode each bit position
        bm_col      = bm[:, None]
        offsets_col  = word_offsets[:, None]
        bit_set      = (bm_col & BIT_POSITIONS[None, :]) != 0
        local_rank   = popcount32(bm_col & MASKS_BELOW[None, :])
        gather_idx   = (offsets_col + local_rank).astype(jnp.int32)
        gather_idx   = jnp.minimum(gather_idx, dense_vals.shape[0] - 1)
        gathered     = dense_vals[gather_idx.ravel()].reshape(N_WORDS, BITS_PER_WORD)
        symbols      = jnp.where(bit_set, gathered, 0)

        return cksum ^ symbols.sum(), None

    cksum, _ = lax.scan(one_run, jnp.int32(0), perturbations)
    return cksum


# ============================================================
# Data generation
# ============================================================
def make_rans_states(key, S):
    return jax.random.randint(key, (S,), minval=M, maxval=2*M, dtype=jnp.int32)


def make_bitmap_data(key, total_symbols, n_runs, sparsity):
    k1, k2, k3 = jax.random.split(key, 3)
    n_words = total_symbols // BITS_PER_WORD

    bits = jax.random.bernoulli(k1, p=(1 - sparsity), shape=(total_symbols,))
    bits_reshaped = bits.reshape(n_words, BITS_PER_WORD).astype(jnp.uint32)
    packed = (bits_reshaped * BIT_POSITIONS[None, :]).sum(axis=-1).astype(jnp.uint32)

    nnz = int(bits.sum())
    dense_vals = jax.random.randint(k2, (max(nnz, 1),), minval=1, maxval=256, dtype=jnp.int32)
    perturbations = jax.random.randint(k3, (n_runs,), minval=0, maxval=2, dtype=jnp.int32)

    return packed, dense_vals, perturbations, nnz


# ============================================================
# Timing
# ============================================================
def time_fn(fn, args):
    for _ in range(N_WARMUP):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(N_TRIALS):
        t0 = timeit.default_timer()
        jax.block_until_ready(fn(*args))
        times.append(timeit.default_timer() - t0)
    return sum(times) / len(times)


# ============================================================
# Main
# ============================================================
def main():
    print(f"JAX {jax.__version__}  |  {jax.devices()[0]}  |  Backend: {jax.default_backend()}")
    print()
    print(f"Model:  M={M}, alphabet={ALPHABET}, sparsity={SPARSITY} "
          f"(F_zero={F_ZERO}, F_nonzero={F_NONZERO})")
    print(f"Output: {TOTAL_SYMBOLS:,} symbols/run × {N_RUNS} runs  |  "
          f"Bitmap: {N_WORDS} packed uint32 words")
    print(f"Timing: {N_WARMUP} warmup, {N_TRIALS} trials (mean)")
    print()

    key = jax.random.PRNGKey(42)
    k_bm, k_rans = jax.random.split(key)

    # Prepare bitmap data
    packed_bitmap, dense_vals, perturbations, nnz = make_bitmap_data(
        k_bm, TOTAL_SYMBOLS, N_RUNS, SPARSITY)
    jax.block_until_ready(packed_bitmap)
    jax.block_until_ready(dense_vals)

    # Time bitmap once (constant across all S)
    t_bm = time_fn(bench_bitmap, (packed_bitmap, dense_vals, perturbations, N_RUNS))

    print(f"Bitmap: nnz={nnz}/{TOTAL_SYMBOLS} ({100*nnz/TOTAL_SYMBOLS:.1f}% nonzero), "
          f"{TOTAL_SYMBOLS * N_RUNS / t_bm / 1e6:.1f} Msym/s")
    print()

    # Table header
    print(f"{'S':>5}  {'Syms/Stream':>11}  "
          f"{'SSrANS':>10}  {'vRaNS':>10}  {'Bitmap':>10}  "
          f"{'BM/SS':>6}")
    print(f"{'':>5}  {'':>11}  "
          f"{'(Msym/s)':>10}  {'(Msym/s)':>10}  {'(Msym/s)':>10}  "
          f"{'':>6}")
    print("─" * 62)

    total_work = TOTAL_SYMBOLS * N_RUNS
    tp_bm = total_work / t_bm / 1e6

    for S in STREAM_COUNTS:
        syms_per_stream = TOTAL_SYMBOLS // S

        init_states = make_rans_states(k_rans, S)
        jax.block_until_ready(init_states)

        t_ss = time_fn(bench_ssrans, (init_states, syms_per_stream, N_RUNS))
        t_vr = time_fn(bench_vanilla_rans, (
            init_states, FREQ_TABLE, CFREQ_TABLE, INV_CFREQ_TABLE,
            syms_per_stream, N_RUNS))

        tp_ss = total_work / t_ss / 1e6
        tp_vr = total_work / t_vr / 1e6
        ratio = t_bm / t_ss

        print(f"{S:>5}  {syms_per_stream:>11,}  "
              f"{tp_ss:>10.1f}  {tp_vr:>10.1f}  {tp_bm:>10.1f}  "
              f"{ratio:>5.1f}×")

    print()
    print("BM/SS: bitmap time ÷ SSrANS time (>1× means SSrANS is faster)")


if __name__ == "__main__":
    main()