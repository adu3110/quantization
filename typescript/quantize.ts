/**
 * Bit-Level Weight Quantization Engine — implemented from scratch.
 *
 * Four quantization schemes, pure TypeScript, zero dependencies:
 *   1. Absmax symmetric  (8-bit)
 *   2. Zero-point affine (4-bit)
 *   3. Block quantization (4-bit)
 *   4. NF4 — Normal Float 4 (QLoRA)
 *
 * Run:  npx ts-node quantize.ts  |  tsc && node quantize.js  |  deno run quantize.ts
 */

// ──────────────────────────────────────────────────────────────────────────────
// Seeded LCG (reproducible without library)
// ──────────────────────────────────────────────────────────────────────────────
class LCG {
  private state: number;
  constructor(seed = 0) { this.state = seed; }

  next(): number {
    this.state = (1664525 * this.state + 1013904223) & 0xffffffff;
    return (this.state >>> 0) / 0x100000000;
  }

  /** Normal-distributed sample via Box-Muller */
  normal(mean = 0, std = 1): number {
    const u1 = this.next() || 1e-10;
    const u2 = this.next();
    const z  = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  }

  matrix(rows: number, cols: number, std = 0.02): number[][] {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => this.normal(0, std))
    );
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Flat view helpers
// ──────────────────────────────────────────────────────────────────────────────
function flatten(m: number[][]): number[] {
  return m.flat();
}

function reshape(flat: number[], rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, (_, r) => flat.slice(r * cols, (r + 1) * cols));
}

function maxAbs(arr: number[]): number {
  return arr.reduce((m, x) => Math.max(m, Math.abs(x)), 0);
}

function minOf(arr: number[]): number { return Math.min(...arr); }
function maxOf(arr: number[]): number { return Math.max(...arr); }

function mse(a: number[], b: number[]): number {
  return a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length;
}

function snrDb(original: number[], reconstructed: number[]): number {
  const sp = original.reduce((s, v) => s + v * v, 0) / original.length;
  const np_ = mse(original, reconstructed);
  return np_ === 0 ? Infinity : 10 * Math.log10(sp / np_);
}

function memBytes(bits: number, n: number): number { return (bits * n) / 8; }

// ──────────────────────────────────────────────────────────────────────────────
// 1. Absmax symmetric quantization
// ──────────────────────────────────────────────────────────────────────────────
interface AbsmaxResult {
  quantized:   Int8Array;
  scale:       number;
  dequantized: number[];
}

function absmaxQuantize(weights: number[], bits = 8): AbsmaxResult {
  const levels    = (1 << (bits - 1)) - 1;   // 127 for 8-bit
  const ma        = maxAbs(weights) || 1;
  const scale     = ma / levels;

  const quantized   = new Int8Array(weights.map(w => Math.round(w / scale)));
  const dequantized = Array.from(quantized).map(q => q * scale);
  return { quantized, scale, dequantized };
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. Zero-point affine quantization
// ──────────────────────────────────────────────────────────────────────────────
interface ZPResult {
  quantized:   Uint8Array;
  scale:       number;
  zeroPoint:   number;
  dequantized: number[];
}

function zeropointQuantize(weights: number[], bits = 4): ZPResult {
  const levels    = (1 << bits) - 1;           // 15 for 4-bit
  const wMin      = minOf(weights), wMax = maxOf(weights);
  const scale     = (wMax - wMin) / levels || 1;
  const zeroPoint = Math.round(-wMin / scale);

  const quantized   = new Uint8Array(weights.map(w =>
    Math.min(levels, Math.max(0, Math.round(w / scale) + zeroPoint))
  ));
  const dequantized = Array.from(quantized).map(q => (q - zeroPoint) * scale);
  return { quantized, scale, zeroPoint, dequantized };
}

// ──────────────────────────────────────────────────────────────────────────────
// 3. Block quantization
// ──────────────────────────────────────────────────────────────────────────────
interface BlockResult {
  quantized:   Int8Array;
  scales:      number[];
  dequantized: number[];
}

function blockQuantize(weights: number[], bits = 4, blockSize = 4): BlockResult {
  const levels    = (1 << (bits - 1)) - 1;
  const n         = weights.length;
  const nBlocks   = Math.ceil(n / blockSize);
  const quantized   = new Int8Array(n);
  const dequantized = new Array(n).fill(0);
  const scales: number[] = [];

  for (let b = 0; b < nBlocks; b++) {
    const start = b * blockSize, end = Math.min(start + blockSize, n);
    const block = weights.slice(start, end);
    const ma    = maxAbs(block) || 1;
    const scale = ma / levels;
    scales.push(scale);
    for (let i = start; i < end; i++) {
      const q = Math.round(weights[i] / scale);
      quantized[i]   = Math.max(-levels, Math.min(levels, q));
      dequantized[i] = quantized[i] * scale;
    }
  }
  return { quantized, scales, dequantized };
}

// ──────────────────────────────────────────────────────────────────────────────
// 4. NF4 — Normal Float 4 (QLoRA)
// ──────────────────────────────────────────────────────────────────────────────
const NF4_LEVELS: number[] = [
  -1.0, -0.6961, -0.5251, -0.3951,
  -0.2844, -0.1848, -0.0917,  0.0,
   0.0796,  0.1609,  0.2461,  0.3379,
   0.4407,  0.5626,  0.7230,  1.0,
];

interface NF4Result {
  indices:     Uint8Array;
  absmax:      number;
  dequantized: number[];
}

function nf4Quantize(weights: number[], absmax?: number): NF4Result {
  const am = absmax ?? maxAbs(weights) || 1;
  const indices = new Uint8Array(weights.map(w => {
    const norm  = w / am;
    let best = 0, bestDist = Infinity;
    for (let i = 0; i < NF4_LEVELS.length; i++) {
      const d = Math.abs(norm - NF4_LEVELS[i]);
      if (d < bestDist) { bestDist = d; best = i; }
    }
    return best;
  }));
  const dequantized = Array.from(indices).map(i => NF4_LEVELS[i] * am);
  return { indices, absmax: am, dequantized };
}

// ──────────────────────────────────────────────────────────────────────────────
// Bit inspector
// ──────────────────────────────────────────────────────────────────────────────
function toBin(n: number, bits: number): string {
  const masked = ((n | 0) + (1 << bits)) & ((1 << bits) - 1);
  return masked.toString(2).padStart(bits, "0");
}

function showBitComparison(
  original: number, quantizedInt: number, reconstructed: number, bits: number
): void {
  console.log(`    original   ${original >= 0 ? "+" : ""}${original.toFixed(6)}`);
  console.log(`    quantized  ${quantizedInt} → int${bits}: ${toBin(quantizedInt, bits)}`);
  console.log(`    recon'd    ${reconstructed >= 0 ? "+" : ""}${reconstructed.toFixed(6)}`);
  console.log(`    bit error: |${original.toFixed(6)} - ${reconstructed.toFixed(6)}| = ${Math.abs(original - reconstructed).toFixed(6)}`);
}

// ──────────────────────────────────────────────────────────────────────────────
// Main demo
// ──────────────────────────────────────────────────────────────────────────────
function main(): void {
  const HR = "=".repeat(65);
  console.log(HR);
  console.log("  Bit-Level Weight Quantization Engine — from scratch (TypeScript)");
  console.log(HR);

  const rng     = new LCG(0);
  const ROWS = 4, COLS = 8;
  const weightsMat = rng.matrix(ROWS, COLS, 0.02);
  const weights    = flatten(weightsMat);
  const n          = weights.length;
  const fp32Bytes  = memBytes(32, n);

  console.log(`\nWeight matrix shape: [${ROWS}, ${COLS}]  (${n} elements)`);
  console.log(`Original (float32) memory: ${fp32Bytes} bytes`);
  console.log("\nSample weights:");
  weightsMat.forEach(row => console.log(" ", row.map(v => v.toFixed(5)).join("  ")));

  // 1 ─ Absmax 8-bit
  console.log(`\n${"─".repeat(65)}`);
  console.log("  [1] Absmax Symmetric — 8-bit");
  console.log("─".repeat(65));
  const r8 = absmaxQuantize(weights, 8);
  console.log(`  Scale : ${r8.scale.toFixed(8)}`);
  console.log(`  MSE   : ${mse(weights, r8.dequantized).toExponential(2)}`);
  console.log(`  SNR   : ${snrDb(weights, r8.dequantized).toFixed(1)} dB`);
  console.log(`  Memory: ${memBytes(8, n)} bytes  (${(fp32Bytes / memBytes(8, n)).toFixed(1)}x smaller)`);
  console.log("\n  Bit-level inspection (first weight):");
  showBitComparison(weights[0], r8.quantized[0], r8.dequantized[0], 8);

  // 2 ─ ZeroPoint 4-bit
  console.log(`\n${"─".repeat(65)}`);
  console.log("  [2] Zero-Point Affine — 4-bit");
  console.log("─".repeat(65));
  const r4 = zeropointQuantize(weights, 4);
  console.log(`  Scale: ${r4.scale.toFixed(8)}   Zero-point: ${r4.zeroPoint}`);
  console.log(`  MSE  : ${mse(weights, r4.dequantized).toExponential(2)}`);
  console.log(`  SNR  : ${snrDb(weights, r4.dequantized).toFixed(1)} dB`);
  console.log(`  Memory: ${memBytes(4, n)} bytes  (${(fp32Bytes / memBytes(4, n)).toFixed(1)}x smaller)`);
  console.log("\n  Bit-level inspection (first weight):");
  showBitComparison(weights[0], r4.quantized[0], r4.dequantized[0], 4);

  // 3 ─ Block 4-bit
  console.log(`\n${"─".repeat(65)}`);
  console.log("  [3] Block Quantization — 4-bit, block_size=4");
  console.log("─".repeat(65));
  const rb = blockQuantize(weights, 4, 4);
  console.log(`  Per-block scales: ${rb.scales.map(s => s.toFixed(6)).join("  ")}`);
  console.log(`  MSE  : ${mse(weights, rb.dequantized).toExponential(2)}`);
  console.log(`  SNR  : ${snrDb(weights, rb.dequantized).toFixed(1)} dB`);

  // 4 ─ NF4
  console.log(`\n${"─".repeat(65)}`);
  console.log("  [4] NF4 — Normal Float 4 (QLoRA)");
  console.log("─".repeat(65));
  const rnf = nf4Quantize(weights);
  console.log(`  Absmax: ${rnf.absmax.toFixed(8)}`);
  console.log(`  MSE   : ${mse(weights, rnf.dequantized).toExponential(2)}`);
  console.log(`  SNR   : ${snrDb(weights, rnf.dequantized).toFixed(1)} dB`);
  console.log(`  Memory: ${memBytes(4, n)} bytes  (${(fp32Bytes / memBytes(4, n)).toFixed(1)}x smaller)`);

  // Summary
  console.log(`\n${HR}`);
  console.log("  Summary: precision vs. compression");
  console.log(HR);
  const rows: [string, number, number, number, number][] = [
    ["float32 (original)",  32, 0,                      Infinity,                   fp32Bytes],
    ["Absmax 8-bit",          8, mse(weights, r8.dequantized),  snrDb(weights, r8.dequantized),  memBytes(8, n)],
    ["ZeroPoint 4-bit",       4, mse(weights, r4.dequantized),  snrDb(weights, r4.dequantized),  memBytes(4, n)],
    ["Block 4-bit (b=4)",     4, mse(weights, rb.dequantized),  snrDb(weights, rb.dequantized),  memBytes(4, n)],
    ["NF4 4-bit (QLoRA)",     4, mse(weights, rnf.dequantized), snrDb(weights, rnf.dequantized), memBytes(4, n)],
  ];
  console.log(`  ${"Method".padEnd(30)} ${"Bits".padStart(4)}  ${"MSE".padStart(10)}  ${"SNR (dB)".padStart(9)}  ${"Memory".padStart(7)}`);
  console.log(`  ${"──────".padEnd(30)} ${"────".padStart(4)}  ${"───".padStart(10)}  ${"────────".padStart(9)}  ${"──────".padStart(7)}`);
  rows.forEach(([name, bits, m, s, mem]) => {
    const snrStr = isFinite(s) ? s.toFixed(1) : "∞";
    console.log(`  ${name.padEnd(30)} ${bits.toString().padStart(4)}  ${m.toExponential(2).padStart(10)}  ${snrStr.padStart(9)}  ${mem.toFixed(0).padStart(5)} B`);
  });
  console.log(HR);
}

main();
