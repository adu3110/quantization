# quantization

> **8-bit: SNR = 45.1 dB (4× smaller). 4-bit: SNR = 20.6 dB (8× smaller).**  
> This is the arithmetic that lets a 70B model run on a laptop — absmax, zero-point, block quant in pure NumPy.

Bit-level weight quantization from scratch — no bitsandbytes, no GGUF. See exactly where the precision loss happens and how to measure it.

---

## Four schemes, side by side

### 1. Absmax symmetric (8-bit) — used in LLM.int8()

Map `[-max|W|, +max|W|]` → `[-127, 127]`:

```
scale     = max(|W|) / 127
W_int     = round(W / scale)          stored as int8
W_deq     = W_int × scale             dequantised for compute
```

**Precision:** MSE ~8×10⁻⁹, SNR ~45 dB. Compression: 4×.

### 2. Zero-point affine (4-bit) — used in GPTQ

Handle asymmetric weight distributions (e.g. all positive):

```
scale     = (max(W) - min(W)) / 15
zero_pt   = round(-min(W) / scale)
W_int     = round(W / scale) + zero_pt   stored as uint4
W_deq     = (W_int - zero_pt) × scale
```

**Precision:** MSE ~2×10⁻⁶, SNR ~21 dB. Compression: 8×.

### 3. Block quantization (4-bit) — used in GGUF / llama.cpp

Divide the weight matrix into blocks; give each its own scale. A single outlier weight can't distort the entire layer:

```
for each block of size B:
    scale_b = max(|block|) / 7
    W_int   = round(block / scale_b)
```

**Precision:** better SNR than global 4-bit. Adds small metadata overhead.

### 4. NF4 — Normal Float 4 (QLoRA, Dettmers et al. 2023)

Weights in trained models follow a roughly normal distribution. NF4 places its 16 quantisation levels at the quantiles of N(0,1) so more levels land where weights actually cluster (near zero):

```
levels = [-1.0, -0.6961, -0.5251, ..., 0.5626, 0.7230, 1.0]

W_norm  = W / max(|W|)                    normalise to [-1, 1]
W_int   = argmin |W_norm - levels[i]|     nearest quantile index
W_deq   = levels[W_int] × max(|W|)
```

**Precision:** best 4-bit quality on normal-distributed weights.

---

## Run

```bash
# Python (numpy only)
pip install numpy
python python/quantize.py

# TypeScript (zero dependencies)
npx ts-node typescript/quantize.ts
```

### Sample output

```
  Method                         Bits     MSE    SNR (dB)  Memory
  float32 (original)               32  0.00e+00         ∞   128 B
  Absmax 8-bit                      8  8.21e-09      45.1    32 B
  ZeroPoint 4-bit                   4  2.30e-06      20.6    16 B
  Block 4-bit (b=4)                 4  8.35e-07      25.0    16 B
  NF4 4-bit (QLoRA)                 4  1.59e-06      22.2    16 B

Bit-level inspection (absmax 8-bit, first weight):
  original   +0.002515  → float16: 00011001 00100110
  quantized               →   int8: 00000111
  recon'd    +0.002563  → float16: 00011001 01000000
  bit error: |0.002515 - 0.002563| = 0.000048
```

---

## Files

```
quantization/
├── python/quantize.py        4 schemes + bit inspector + summary table
└── typescript/quantize.ts    identical math, zero npm dependencies
```

---

## Related work

- **LLM.int8()** (Dettmers et al., 2022) — absmax per-row with mixed-precision outliers
- **GPTQ** (Frantar et al., 2022) — zero-point 4-bit with second-order weight update
- **QLoRA** (Dettmers et al., 2023) — NF4 + double quantization for fine-tuning
- **GGUF / llama.cpp** — block quantization making local LLM inference practical

---

## License

MIT
