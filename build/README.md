# Building vLLM from source for GB10 (sm_121)

Builds upstream vLLM on NVIDIA's signed base so the runtime has no unreproducible
third-party binary. **Status: builds and runs. Not yet benchmarked end-to-end** (needs Ray
or `external_launcher` for TP=2 — upstream has no `--nnodes/--node-rank/--master-addr`;
those are additions in the community image).

```bash
docker build -f Dockerfile.pinned -t vllm-gb10:pinned .
```

## What is established (measured, not assumed)

**The build works.** `vllm-0.1.dev1+gadc3e0351.cu133`, 24 min on 20 cores. Kernels execute
natively on GB10 — verified by running a real vLLM CUDA op on the device (`rms_norm`,
0.005 ms/call steady state, no JIT stall).

**sm_120 and sm_121 emit byte-identical machine code.** Same PTX → both targets:

```
ptxas -O3 -arch=sm_120  →  10024 bytes
ptxas -O3 -arch=sm_121  →  10024 bytes
SASS diff over 408 lines: only the `.target` label
```

This holds for **NVFP4 block-scaled MMA** too (`mma.sync...kind::mxf4nvf4.block_scale`,
5640 bytes both, label-only diff) — not just FP16 `wmma`. So targeting sm_121 buys nothing,
and an sm_121-only cubin would *not* run on sm_120. Hence `ARCH_LIST=12.0+PTX`.

**`cuobjdump` does distinguish the variants** (verified: `sm_120a`→`sm_120a`,
`sm_120f`→`sm_120 sm_120f`, `sm_121a`→`sm_121a`), so the reporting is trustworthy.

**Our output matches the working community image exactly:**

| module | ours | community image |
|---|---|---|
| `_qutlass_C.abi3.so` | `sm_120` | `sm_120` |
| `_moe_C_stable_libtorch.abi3.so` | `sm_120 sm_80` | `sm_120 sm_80` |

Neither ships `a`-variant cubins in the AOT modules. That is the status quo for vLLM, and it
is the configuration that demonstrably serves this model at ~41 tok/s.

**FlashInfer SM121 gaps do not manifest here.** [Issue #3170](https://github.com/flashinfer-ai/flashinfer/issues/3170)
documents a `minor == 0` heuristic excluding the b12x FP4 backend on SM121, plus SM90-only
modules that crash at load. On NVIDIA's FlashInfer 0.6.14 build, measured on the device:
`has_flashinfer_b12x_gemm() → True`, `mm_fp8` loads, MLA sparse imports, bf16 runs. vLLM
gates on `is_device_capability_family(120)`, which matches 121 by construction
(`121 // 10 == 120 // 10`), and there are no strict-equality checks excluding it.

## The open question

**Is NVFP4 GEMM running natively on the Tensor Cores, or silently dequantizing to BF16/FP8?**

GB10 Tensor Cores support NVFP4 in hardware. But the NVFP4 path in this stack is **not** in
the AOT `.so` files — it comes from FlashInfer/CuTeDSL **JIT at runtime**, which is why the
runtime sets `CUTE_DSL_ARCH=sm_121a` and why `W4A16FusedMoeKernel` compiles mid-inference.
Cubin inspection cannot answer this; it needs a runtime profile:

```bash
ncu --set full <workload>
# native  → block-scaled FP4 GEMM kernels, high Tensor Core utilization
# fallback→ dequantize + BF16/FP16 GEMM dominating
```

## Two traps that cost hours

**pip will silently replace NVIDIA's torch.** vLLM pins `torch == 2.13.0`; the NVIDIA base
ships `2.13.0a0+...nv26.07`. pip does not treat those as equal, uninstalls the tuned
aarch64/CUDA-13 build, and pulls a generic wheel — **the build still succeeds**, having
discarded the reason for using that base. Strip torch pins, use `--no-deps`, and assert
afterwards that NVIDIA's torch survived.

**`--no-deps` then breaks the Rust step.** vLLM has a Rust component; `setuptools-rust`
needs `semantic_version`, and the NVIDIA base has no `rustc`. Both must be added explicitly
or the build dies at packaging *after* the full 24-minute CUDA compile.

## Known gap vs the community image

`nvfp4_ds_mla` (NVFP4 KV cache) is **not upstream** — 0 references in vLLM v0.26.0 or main.
Falling back to `fp8_ds_mla` roughly halves the KV pool (~1.56M → ~800K tokens), which likely
puts full 1M single-request context out of reach. This is the one real regression from
building rather than using the community image.
