# The journey — what we tried, what broke, what fixed it

Written for the curious, and for anyone about to repeat our mistakes. The working
configuration is in the [README](../README.md); this is how we got there, including
the wrong turns. **Several conclusions in here were published and later proven wrong** —
they are kept, marked, because the corrections are the useful part.

---

## Start: does this even fit?

`DeepSeek-V4-Flash-0731` reports **304B parameters**. Two DGX Sparks have 256 GB of
unified memory between them. That looks impossible.

It isn't: `config.json` says `expert_dtype: "fp4"`, so the experts ship in 4-bit and the
checkpoint is **166.9 GB on disk** — 48 shards. It does not fit one 128 GB node; it fits
across two with room for a ~1.5M-token KV cache.

**Lesson:** for MoE checkpoints, parameter count tells you almost nothing about footprint.
Read the shard sizes.

## Wrong turn #1 — chasing NVFP4 weights that do not exist

We went looking for an NVFP4 build from a trustworthy vendor. Findings:

- NVIDIA publishes `DeepSeek-V4-Flash-NVFP4`, but it is the **April preview**, not 0731 —
  no DSpark module, and no smaller (~168 GB vs 166.9 GB).
- DeepSeek publishes **exactly one quantization per model**. Only the `-Base` repos differ
  (`expert_dtype: fp8`), which nearly doubles size — `Flash-Base` is **294.7 GB**.
- So **no NVFP4 weights for 0731 exist**, from anyone.

Then, reading the actual tensors rather than the config summary:

```
layers.4.ffn.experts.0.w1.weight  dtype=I8      shape=[2048, 2048]
layers.4.ffn.experts.0.w1.scale   dtype=F8_E8M0 shape=[2048, 128]
```

E8M0 scales at block size 32 — that is **MXFP4**, not NVFP4 (which is E4M3 at block 16).
The only NVFP4 in the stack is the KV cache (`--kv-cache-dtype nvfp4_ds_mla`).

**Lesson:** "FP4" is not one format. Check the scale dtype and block size before believing
a label — including our own earlier claim, which was wrong until we looked.

## The gotcha that would have invalidated every number

The upstream recipe's example sets `NCCL_IB_HCA=rocep1s0f1`. On our cluster the CX7 link is
`enp1s0f0np0` → **`rocep1s0f0`**. Had we copied the example, NCCL would have silently fallen
back to the 1 GbE management network — **no error, just quietly wrong TP numbers**.

We verified the fabric before trusting anything: `ibdev2netdev` to map HCA→netdev,
`ip route get <worker>` to confirm which interface actually carries traffic, then
`ib_write_bw` end-to-end — **106 Gb/s measured**.

**Lesson:** prove the interconnect with a bandwidth test before benchmarking anything
multi-node. A silent fallback looks exactly like "this hardware is slow".

## Wrong turn #2 — pipeline parallelism is impossible here

We wanted to compare TP vs PP. PP=2 fails **two independent ways**:

1. **With DSpark:** rejected in config validation — and the blocker is the *draft* model,
   not the target: `draft_model_config.verify_with_parallel_config(...)` →
   `NotImplementedError: Pipeline parallelism is not supported for this model.`
2. **Without DSpark:** runtime crash in DeepSeek-V4's own compressor —
   `compressor.py:372: Invalid state_cache.strides[0] ... expected to be divisible by 16`.
   43 layers split 22/21 across pipeline ranks produces a misaligned stride.

**TP=2 is the only working multi-node configuration.** No flag works around either failure.

## Wrong turn #3 — the one we got publicly wrong, twice

This is the embarrassing part, and the most instructive.

**Claim 1 (published, wrong):** "DSpark caps concurrency at 32 streams." Basis: one sweep
where 1/4/8/16/32 succeeded and 48 killed the engine. We turned a single data point into a
threshold.

**Claim 2 (published, also wrong):** "DSpark is inherently unstable; no safe concurrency
above 1." Basis: a second run died at **8** streams. We corrected the first error by
over-generalising in the other direction.

**Actual cause:** we were running the wrong sampling method. The upstream compose hardcodes

```json
{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}
```

but [DeepSeek's model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731#how-to-run-with-vllm)
specifies

```json
{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}
```

With DeepSeek's own parameters: **9 consecutive sweeps (3 rounds × 8/16/32 streams), zero
crashes**, including three passes through 8 streams — the exact point where `probabilistic`
died. Quality unchanged (4/4 on checkable probes). Cost: ~14% single-stream throughput.

**Lessons, in order of how much time they would have saved us:**

1. **Read the model author's own serving instructions first.** Hours lost to a crash the
   vendor's model card would have prevented.
2. **One data point is not a threshold.** Both wrong claims came from insufficient
   replication. The fix was a soak test — repeat until the result is boring.
3. **A downstream recipe's defaults are not the vendor's defaults**, even when the recipe
   is good and well documented.

## What the crash actually looked like

Useful if you hit it: the engine dies with `EngineDeadError`, and the scheduler dump shows
the drafter emitting **invalid draft token ids** on a step mixing prefill and decode:

```
TimeoutError: RPC call to sample_tokens timed out.
scheduled_spec_decode_tokens={...: [-1,-1,-1,-1,-1], ...}
SchedulerStats(num_running_reqs=8, kv_cache_usage=0.0150...)
```

Note `kv_cache_usage` — **1.5%**. It is not memory pressure, which is the natural first
guess. Look at the draft token ids.

## Things that quietly cost performance

- **Swap.** One node still had stock `vm.swappiness=60` and had pushed **1.8 GB of the vLLM
  stack to disk**. On unified memory a swapped page is a multi-second stall mid-generation.
  Set `vm.swappiness=0` on **both** nodes and reclaim with `swapoff -a && swapon -a`.
- **Client timeouts, misread as server failures.** Claude Code's `API_TIMEOUT_MS` defaults to
  600000 (10 min); a full 1M-token prefill takes **17–19 min**. The client errors while the
  server finishes anyway and caches the result — so the retry returns instantly and it looks
  like a flaky server that "fixes itself". It is neither flaky nor fixing itself.
  (Tested: `-1` hard-errors, `0` is undefined behaviour. Use a large finite value.)
- **Benchmarking a cold server.** Triton/CuTeDSL kernels JIT-compile *during* inference on
  first use of each batch shape. Our 16-stream TTFT read 10.61 s cold vs 0.91 s warm — a 12×
  artifact that looks exactly like a scaling wall. Warm up every shape you intend to measure.
- **Benchmarking under memory pressure.** One single-stream result read 27.8 tok/s; three
  re-measurements gave 41.8 / 40.9 / 40.2. A browser crashing on the host was enough.

## Provenance: can we avoid the third-party image?

The working stack depends on `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, a community image.
We audited it and evaluated alternatives.

**Audit result — no red flags, but not reproducible.** No phone-home, no cron, no obfuscated
payloads; the curl/AWS/Azure libraries are from NVIDIA's `nixl`. Its vLLM is
`0.25.2.dev0+g752a3a504`, and `752a3a504485` **is a genuine upstream commit** (2026-07-12).
But the image labels say `build.commit: unknown`, `build.pipeline: local` — **built by hand,
not reproducible from source.** "No evidence of malice" is the strongest available claim.

**NVIDIA's official image does not replace it.** `nvcr.io/nvidia/vllm:26.07-py3` is signed,
first-party, multi-arch, and explicitly documents DGX Spark — but:

| | NVIDIA `vllm:26.07` | Anemll `0.1.1` |
|---|---|---|
| `DeepseekV4ForCausalLM` | ✅ | ✅ |
| DSpark speculation | ❌ | ✅ |
| `nvfp4_ds_mla` KV | ❌ | ✅ |
| GB10 `sm_121` kernels | ❌ (`sm_120` + PTX) | ❌ (`sm_120/80/89/90` + runtime JIT) |
| Multi-node serve flags | ❌ (needs Ray) | ✅ (`--nnodes/--node-rank`) |

**Correction to our own earlier claim:** we said Anemll shipped "native sm_121a" kernels.
It does not — its prebuilts are `sm_120/80/89/90`, and `sm_121a` is reached by *runtime JIT*
(`CUTE_DSL_ARCH=sm_121a`). That is the same JIT that polluted our TTFT numbers. **Neither
existing image has native GB10 kernels.**

## Building it ourselves

That gap is what motivated a source build:

- **vLLM only lists `12.1` as a supported arch when CUDA ≥ 13.0** — their CMake comment:
  *"Family-conditional `12.0f` (one cubin for SM12x family) requires CUDA >= 13.0; fall back
  to architecture-specific `12.0a;12.1a` on CUDA < 13.0."* NVIDIA's base ships **CUDA 13.3**,
  and `nvcc -arch=sm_121a` compiles.
- **DSpark is upstream** — 23 references in v0.26.0, 37 on main, so building from source gets
  a version six weeks newer than the community image.
- **DSpark needs no custom kernels** — it is Python (`deepseek_v4` is 37 `.py` files, zero
  `.so`), so it rides the general vLLM kernels.
- **`nvfp4_ds_mla` is NOT upstream** (0 references). This is the one real regression: KV falls
  back to `fp8_ds_mla`, roughly halving the KV pool.

Base chosen: `nvcr.io/nvidia/vllm:26.07-py3` — NVIDIA's signed image already has aarch64
torch, CUDA 13.3 and the full toolchain, so we recompile only vLLM against a first-party base.

**Trap worth flagging:** vLLM pins `torch == 2.13.0`; NVIDIA ships `2.13.0a0+...nv26.07`.
pip does not treat those as equal and will **uninstall NVIDIA's tuned aarch64/CUDA-13 torch**
and pull a generic wheel — the build still "succeeds", having discarded the entire reason for
using that base. Strip torch pins from the build requirements, use `--no-deps`, and assert
afterwards that NVIDIA's torch survived.

### Result: the premise was wrong — sm_120 and sm_121 generate identical code

The build succeeded (`vllm-0.1.dev1+gadc3e0351.cu133`, 24 min, NVIDIA's torch intact), and
the kernels execute natively on GB10 — verified by running a real vLLM CUDA op on the device:

```
device: NVIDIA GB10 (12, 1)
first call:   3.7 ms      (context init, not a PTX JIT stall)
steady state: 0.005 ms/call
```

But `cuobjdump` reported the built kernels as **sm_120**, not sm_121 — the same as both
existing images. That looked like a failure. It is not. Two independent mechanisms make
`sm_120` code correct on GB10:

1. **Classic cubin compatibility** — a cubin for `X.y` runs on `X.z` where `z >= y`, so an
   `sm_120` cubin runs on `sm_121`. (The reverse is **not** true: `sm_121` cubins do not run
   on `sm_120`, so targeting 121 only *narrows* portability.)
2. **CUDA 13 family targets** — `compute_120f` produces one cubin covering the whole SM12x
   family. `cuobjdump` labels it by the family base, i.e. `sm_120`.

We then measured whether sm_121-native codegen differs at all. Same PTX, both targets,
representative kernel (wmma tensor cores + shared memory + block sync):

```
ptxas -O3 -arch=sm_120 k.ptx -o k120.cubin     10024 bytes
ptxas -O3 -arch=sm_121 k.ptx -o k121.cubin     10024 bytes

SASS diff, 408 lines:
  <   code for sm_120        >   code for sm_121
  <   .target sm_120         >   .target sm_121
```

**Byte-identical size; the only difference is the architecture label.** All 406 instruction
lines match — same selection, registers and scheduling.

**Conclusion: there is nothing to gain by building for sm_121.** sm_120 and sm_121 share one
12.x spec column (99 KB/block shared memory, 48 warps/SM, 64K registers/SM, identical
tensor-core dtypes, TMA, clusters). The only hardware difference is GB10's unified memory,
for which no kernel optimizations currently exist. Both prebuilt images were already emitting
optimal code for this GPU.

### The related concern that also turned out fine

[FlashInfer #3170](https://github.com/flashinfer-ai/flashinfer/issues/3170) documents real
SM121 gaps — notably the **b12x FP4 backend excluded from auto-selection because heuristics
check `minor == 0` only**, which is exactly the "sm_121 falls into a slow path" failure mode
worth worrying about, and b12x is the MoE backend this model uses.

On the versions here it does not apply. Measured on the device:

```
has_flashinfer_b12x_gemm            True
has_flashinfer_cutlass_fused_moe    True
is_device_capability_family(120)    True     # 121 // 10 == 120 // 10
```

vLLM gates on `is_device_capability_family(120)`, which matches sm_121 by construction, and we
found **no strict-equality checks** that would exclude it. NVIDIA's FlashInfer 0.6.14 build
handles SM121 explicitly (`compute_121a`).

### So what is the source build actually worth?

Honest answer, after all of the above: **provenance and freshness, not performance.**

| | gain |
|---|---|
| Reproducible from pinned upstream commit on NVIDIA's signed base | ✅ real |
| Six-weeks-newer DSpark | ✅ possibly |
| CUDA 13.3 vs the community image's 13.0 | ✅ marginal |
| "Native sm_121 kernels" | ❌ **no such thing — identical code** |
| `nvfp4_ds_mla` KV cache | ❌ **lost** — not upstream, halves the KV pool |

If you trust the community image, there is little performance reason to build. If you need
a reproducible supply chain, the build is worth it — and costs you the NVFP4 KV cache.

**Lesson:** we spent hours building to fix a problem that did not exist, on the assumption
that a version label implied different machine code. One `ptxas` invocation and a `diff`
would have answered it up front. Measure the premise before optimising against it.
