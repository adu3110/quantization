# quantization

Bit-level weight quantization engine — four schemes implemented from scratch in NumPy / TypeScript.

Libraries like `bitsandbytes` hide the math. This repo performs the actual quantization arithmetic so you can see exactly where precision is lost.

## Schemes

| Scheme | Formula | Used in |
|--------|---------|---------|
| **Absmax symmetric** | `scale = max(|W|) / (2^(b-1) - 1)` | LLM.int8() |
| **Zero-point affine** | `scale = (max−min) / (2^b − 1)`,  `zp = round(−min/scale)` | GPTQ |
| **Block quantization** | Per-block scales, reduces outlier sensitivity | GGUF / llama.cpp |
| **NF4** | 16-point non-uniform grid matched to N(0,1) | QLoRA |

## Run

```bash
# Python (NumPy only)
python python/quantize.py

# TypeScript (zero deps)
npx ts-node typescript/quantize.ts
```

## What you'll see

- 8-bit and 4-bit quantized weight matrices
- Bit-level inspection of individual weights (binary representation before/after)
- MSE and SNR (dB) for each scheme
- Memory reduction ratio table

## Dependencies

- Python: `numpy` only
- TypeScript: none
