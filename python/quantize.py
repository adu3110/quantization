"""
Bit-Level Weight Quantization Engine — implemented from scratch.

What this demonstrates
----------------------
Quantization compresses model weights from float32/float16 into fewer bits
(8-bit, 4-bit, 2-bit) to reduce memory and speed up inference. Libraries like
bitsandbytes hide the math. This file performs the actual bit math in pure
NumPy so you can see exactly where precision is lost.

Schemes implemented
-------------------
1. Absmax (symmetric)   — used in LLM.int8() for 8-bit
2. Zero-point (affine)  — used in GPTQ-style 4-bit
3. Block quantization   — divide matrix into blocks; each block has its own
                          scale (reduces outlier sensitivity)
4. NF4 (Normal Float 4) — used in QLoRA; non-uniform grid matched to
                          normal distribution of weights

Run
---
    python quantize.py

Dependencies: numpy only.
"""

from __future__ import annotations

import struct
import numpy as np

RNG = np.random.default_rng(0)

# ──────────────────────────────────────────────────────────────────────────────
# Helper: float16 bit representation
# ──────────────────────────────────────────────────────────────────────────────

def float_to_bits(x: float, dtype: str = "f2") -> str:
    """Return binary string for a float16 (e) or float32 (f) value."""
    fmt    = "e" if dtype == "f2" else "f"
    size   = 2   if dtype == "f2" else 4
    packed = struct.pack(f">{fmt}", x)
    return " ".join(f"{b:08b}" for b in packed[:size])


# ──────────────────────────────────────────────────────────────────────────────
# 1. Absmax symmetric quantization  (8-bit or any n-bit)
# ──────────────────────────────────────────────────────────────────────────────

def absmax_quantize(
    weights: np.ndarray, bits: int = 8
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Symmetric quantization: maps [-max_abs, +max_abs] → [-2^(bits-1), 2^(bits-1)-1].

    scale  = max(|W|) / (2^(bits-1) - 1)
    W_int  = round(W / scale)
    W_deq  = W_int * scale
    """
    levels   = 2 ** (bits - 1) - 1           # e.g. 127 for 8-bit
    max_abs  = float(np.abs(weights).max())
    scale    = max_abs / levels if max_abs != 0 else 1.0

    quantized   = np.round(weights / scale).clip(-levels, levels).astype(np.int8)
    dequantized = quantized.astype(np.float32) * scale

    return quantized, scale, dequantized


# ──────────────────────────────────────────────────────────────────────────────
# 2. Zero-point affine quantization (asymmetric)
# ──────────────────────────────────────────────────────────────────────────────

def zeropoint_quantize(
    weights: np.ndarray, bits: int = 4
) -> tuple[np.ndarray, float, int, np.ndarray]:
    """
    Asymmetric quantization: maps [min(W), max(W)] → [0, 2^bits - 1].

    scale      = (max - min) / (2^bits - 1)
    zero_point = round(-min / scale)
    W_int      = round(W / scale) + zero_point
    W_deq      = (W_int - zero_point) * scale
    """
    levels     = 2 ** bits - 1                # e.g. 15 for 4-bit
    w_min, w_max = float(weights.min()), float(weights.max())
    scale      = (w_max - w_min) / levels if (w_max - w_min) != 0 else 1.0
    zero_point = int(round(-w_min / scale))

    quantized   = (np.round(weights / scale) + zero_point).clip(0, levels).astype(np.uint8)
    dequantized = (quantized.astype(np.float32) - zero_point) * scale

    return quantized, scale, zero_point, dequantized


# ──────────────────────────────────────────────────────────────────────────────
# 3. Block quantization
# ──────────────────────────────────────────────────────────────────────────────

def block_quantize(
    weights: np.ndarray, bits: int = 4, block_size: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide weight vector into blocks; each block gets its own scale.
    Reduces sensitivity to large outlier weights.

    Returns
    -------
    quantized   — integer representation (same shape)
    scales      — one scale per block
    dequantized — reconstructed float weights
    """
    flat     = weights.flatten()
    n        = len(flat)
    levels   = 2 ** (bits - 1) - 1
    n_blocks = (n + block_size - 1) // block_size

    quantized   = np.zeros(n, dtype=np.int8)
    dequantized = np.zeros(n, dtype=np.float32)
    scales      = np.zeros(n_blocks, dtype=np.float32)

    for b in range(n_blocks):
        start, end = b * block_size, min((b + 1) * block_size, n)
        block      = flat[start:end].astype(np.float32)
        max_abs    = float(np.abs(block).max())
        scale      = max_abs / levels if max_abs != 0 else 1.0
        scales[b]  = scale

        q = np.round(block / scale).clip(-levels, levels).astype(np.int8)
        quantized[start:end]   = q
        dequantized[start:end] = q.astype(np.float32) * scale

    return quantized.reshape(weights.shape), scales, dequantized.reshape(weights.shape)


# ──────────────────────────────────────────────────────────────────────────────
# 4. NF4 — Normal Float 4 (QLoRA)
# ──────────────────────────────────────────────────────────────────────────────

# 16 quantile levels derived from the standard normal CDF so the grid is
# denser where normal-distributed weights cluster (near zero).
_NF4_LEVELS = np.array([
    -1.0,       -0.6961,    -0.5251,    -0.3951,
    -0.2844,    -0.1848,    -0.0917,     0.0,
     0.0796,     0.1609,     0.2461,     0.3379,
     0.4407,     0.5626,     0.7230,     1.0,
], dtype=np.float32)


def nf4_quantize(
    weights: np.ndarray, absmax: float | None = None
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    NF4 as used in QLoRA (Dettmers et al., 2023).

    Normalise weights to [-1, 1] then map each value to the nearest level in
    the 16-point NF4 grid. Store as 4-bit indices (uint8 here for clarity).
    """
    if absmax is None:
        absmax = float(np.abs(weights).max()) or 1.0

    normalised  = weights.astype(np.float32) / absmax
    # Vectorised nearest-neighbour in the NF4 grid
    indices     = np.abs(normalised[..., None] - _NF4_LEVELS).argmin(axis=-1).astype(np.uint8)
    dequantized = _NF4_LEVELS[indices] * absmax

    return indices, absmax, dequantized


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2))


def snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Signal-to-noise ratio in dB (higher = better reconstruction)."""
    signal_power = float(np.mean(original.astype(np.float32) ** 2))
    noise_power  = mse(original, reconstructed)
    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def memory_bytes(bits: int, n_elements: int) -> float:
    return (bits * n_elements) / 8


# ──────────────────────────────────────────────────────────────────────────────
# Bit inspector
# ──────────────────────────────────────────────────────────────────────────────

def show_bit_comparison(original: float, quantized_int: int, reconstructed: float, bits: int) -> None:
    orig_bits  = float_to_bits(float(np.float16(original)), dtype="f2")
    recon_bits = float_to_bits(float(np.float16(reconstructed)), dtype="f2")
    q_bits     = f"{int(quantized_int) & ((1 << bits) - 1):0{bits}b}"
    print(f"    original  {original:+.6f}  → float16: {orig_bits}")
    print(f"    quantized                  → int{bits}:    {q_bits}")
    print(f"    recon'd   {reconstructed:+.6f}  → float16: {recon_bits}")
    print(f"    bit error: |{original:.6f} - {reconstructed:.6f}| = {abs(original - reconstructed):.6f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main demo
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    HR = "=" * 65
    print(HR)
    print("  Bit-Level Weight Quantization Engine — from scratch (NumPy)")
    print(HR)

    # Simulate a small weight matrix (e.g., one FFN layer row)
    weights = RNG.normal(0, 0.02, (4, 8)).astype(np.float32)
    n_el    = weights.size
    fp32_bytes = memory_bytes(32, n_el)

    print(f"\nWeight matrix shape: {weights.shape}  ({n_el} elements)")
    print(f"Original (float32) memory: {fp32_bytes:.0f} bytes")
    print(f"\nSample weights:\n{weights.round(5)}")

    # ── 8-bit absmax ──────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  [1] Absmax Symmetric — 8-bit")
    print(f"{'─'*65}")
    q8, scale8, deq8 = absmax_quantize(weights, bits=8)
    print(f"  Scale factor : {scale8:.8f}")
    print(f"  Quantized    :\n{q8}")
    print(f"  Dequantized  :\n{deq8.round(5)}")
    print(f"  MSE          : {mse(weights, deq8):.2e}")
    print(f"  SNR          : {snr_db(weights, deq8):.1f} dB")
    print(f"  Memory       : {memory_bytes(8, n_el):.0f} bytes  ({fp32_bytes/memory_bytes(8,n_el):.1f}x smaller)")
    print("\n  Bit-level inspection (first weight):")
    show_bit_comparison(float(weights.flat[0]), int(q8.flat[0]), float(deq8.flat[0]), bits=8)

    # ── 4-bit zero-point ──────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  [2] Zero-Point Affine — 4-bit")
    print(f"{'─'*65}")
    q4, scale4, zp4, deq4 = zeropoint_quantize(weights, bits=4)
    print(f"  Scale        : {scale4:.8f}   Zero-point: {zp4}")
    print(f"  Quantized    :\n{q4}")
    print(f"  Dequantized  :\n{deq4.round(5)}")
    print(f"  MSE          : {mse(weights, deq4):.2e}")
    print(f"  SNR          : {snr_db(weights, deq4):.1f} dB")
    print(f"  Memory       : {memory_bytes(4, n_el):.0f} bytes  ({fp32_bytes/memory_bytes(4,n_el):.1f}x smaller)")
    print("\n  Bit-level inspection (first weight):")
    show_bit_comparison(float(weights.flat[0]), int(q4.flat[0]), float(deq4.flat[0]), bits=4)

    # ── 4-bit block quantization ──────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  [3] Block Quantization — 4-bit, block_size=4")
    print(f"{'─'*65}")
    qb, scales_b, deqb = block_quantize(weights, bits=4, block_size=4)
    print(f"  Per-block scales: {scales_b.round(8)}")
    print(f"  Quantized    :\n{qb}")
    print(f"  Dequantized  :\n{deqb.round(5)}")
    print(f"  MSE          : {mse(weights, deqb):.2e}")
    print(f"  SNR          : {snr_db(weights, deqb):.1f} dB")
    overhead = scales_b.nbytes
    print(f"  Memory       : {memory_bytes(4, n_el) + overhead:.0f} bytes (incl. scale overhead)")

    # ── NF4 ───────────────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  [4] NF4 — Normal Float 4 (QLoRA)")
    print(f"{'─'*65}")
    qnf4, absmax_nf4, deq_nf4 = nf4_quantize(weights)
    print(f"  Absmax       : {absmax_nf4:.8f}")
    print(f"  NF4 grid (16 levels): {_NF4_LEVELS.round(4).tolist()}")
    print(f"  Indices      :\n{qnf4}")
    print(f"  Dequantized  :\n{deq_nf4.round(5)}")
    print(f"  MSE          : {mse(weights, deq_nf4):.2e}")
    print(f"  SNR          : {snr_db(weights, deq_nf4):.1f} dB")
    print(f"  Memory       : {memory_bytes(4, n_el):.0f} bytes  ({fp32_bytes/memory_bytes(4,n_el):.1f}x smaller)")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{HR}")
    print("  Summary: precision vs. compression")
    print(HR)
    print(f"  {'Method':<30} {'Bits':>4}  {'MSE':>10}  {'SNR (dB)':>9}  {'Memory':>7}")
    print(f"  {'──────':<30} {'────':>4}  {'───':>10}  {'────────':>9}  {'──────':>7}")
    rows = [
        ("float32 (original)",   32, 0.0,                    float("inf"),          fp32_bytes),
        ("Absmax 8-bit",          8, mse(weights, deq8),      snr_db(weights, deq8),  memory_bytes(8, n_el)),
        ("ZeroPoint 4-bit",       4, mse(weights, deq4),      snr_db(weights, deq4),  memory_bytes(4, n_el)),
        ("Block 4-bit (b=4)",     4, mse(weights, deqb),      snr_db(weights, deqb),  memory_bytes(4, n_el)),
        ("NF4 4-bit (QLoRA)",     4, mse(weights, deq_nf4),   snr_db(weights, deq_nf4), memory_bytes(4, n_el)),
    ]
    for name, bits, m, s, mem in rows:
        snr_str = f"{s:.1f}" if s != float("inf") else "∞"
        print(f"  {name:<30} {bits:>4}  {m:>10.2e}  {snr_str:>9}  {mem:>5.0f} B")
    print(HR)


if __name__ == "__main__":
    main()
