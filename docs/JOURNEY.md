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

*(Build results and benchmarks to follow — this section is updated as the work completes.)*
