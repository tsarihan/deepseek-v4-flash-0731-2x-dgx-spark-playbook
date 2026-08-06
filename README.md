# DeepSeek-V4-Flash-0731 on 2× DGX Spark (GB10)

Serving `deepseek-ai/DeepSeek-V4-Flash-0731` at its full **1M-token context** across two
DGX Sparks with tensor parallelism, DSpark speculative decoding, and an NVFP4 KV cache.

This is a **findings-and-delta** repo, not a fork. The launcher and compose stack come from
[MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
(MIT). What is added here: a small patch to make parallelism configurable, systemd units with
a watchdog, measurement harnesses, and a set of results that answer questions the upstream
recipe leaves open — most importantly **why pipeline parallelism cannot work for this model**
and **why DSpark speculative decoding is unsafe for unattended multi-user serving**.

Every number below was measured on the hardware described, not estimated.

---

## What to expect (performance at a glance)

> ⚠ **Read Finding #2 before planning around the multi-stream numbers.** With DSpark
> speculative decoding enabled, the engine can die at *any* concurrency above 1 — we saw it at
> 48 streams in one run and at 8 in another. The throughput below is real and reproducible,
> but it is **not a stability guarantee**. For unattended multi-user serving use
> `SPEC_DECODE=off`, which completed the full sweep without incident.

Measured at **TP=2, DSpark on, `max_num_seqs=48`, 1M context**. Long-answer prompt, 1024
output tokens per stream, distinct prompt per stream, warmup discarded.

| concurrent streams | TTFT (p50) | per-stream tok/s | **aggregate tok/s** | wall time |
|---:|---:|---:|---:|---:|
| 1 | 0.28 s | 47.3 | 46.8 | 22 s |
| 4 | 1.55 s | 26.0 | 98.4 | 42 s |
| 8 | 0.77 s | 18.3 | 140.3 | 58 s |
| 16 | 10.61 s ⚠ | 13.6 | 187.5 | 87 s |
| 32 | 1.04 s | 13.5 | **229.5** | 143 s |
| 48 | — | — | **engine died** (see Finding #2) | — |

Same box with speculation **off** — slower everywhere, but 48 streams work:

| streams | 1 | 4 | 8 | 16 | 32 | 48 |
|---|---:|---:|---:|---:|---:|---:|
| per-stream tok/s | 24.7 | 17.1 | 13.0 | 9.6 | 9.4 | 8.0 |
| aggregate tok/s | 24.5 | 64.0 | 94.2 | 145.6 | 159.0 | 189.8 |

**How to read this:**

- **One user gets ~47 tok/s.** That is the interactive experience on a single stream.
- **Throughput scales sublinearly, as expected** — 32× the load yields ~4.9× the aggregate
  (46.8 → 229.5 tok/s) while per-stream drops 47.3 → 13.5 tok/s as the batch shares two GPUs.
- **Throughput peaks around 32 streams** (229.5 tok/s) — spec-on @ 32 beats spec-off @ 48
  (189.8), so going wider does not help. This is a *throughput* observation, **not** a
  stability threshold; see Finding #2.
- **Speculation buys 1.4–1.9× at every level and never inverts** — but it is the component
  that crashes the engine. Turn it off for unattended or multi-user serving.
- ⚠ **The 16-stream TTFT is an artifact, not a scaling wall.** Triton/CuTeDSL kernels were
  still JIT-compiling *during* inference because one warmup pass did not cover every batch
  shape. Throughput columns are unaffected — they measure the decode window after the first
  token. Expect ~1 s TTFT there in steady state, in line with the 8- and 32-stream figures.

Long context, measured separately: a **130,029-token** prompt prefills in **71 s**
(1,831 tok/s) and correctly retrieves a needle buried at 50% depth. A full 1M-token prefill
takes roughly **17–19 minutes** — see Finding #6 before pointing a client at it.

---

## Hardware / software this was validated on

| | |
|---|---|
| Nodes | 2× DGX Spark, GB10, sm_121, 128 GB unified memory each |
| Interconnect | ConnectX-7, 200 GbE direct attach, RoCEv2 (**106 Gb/s** measured, `ib_write_bw`) |
| Image | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (vLLM `0.25.2.dev0`) |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731` @ `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| Weights on disk | **166.9 GB** — 48 shards, needed on **both** nodes |

**The 304B parameter count is misleading.** `expert_dtype: fp4` means the experts ship in FP4,
so the checkpoint is ~167 GB, not ~300 GB. It does not fit one 128 GB node; it does fit across
two with room for a 1.5M-token KV cache.

---

## Quantization: these are MXFP4 experts, not NVFP4

Widely muddled, so stated precisely. Verified by reading the tensors, not the config:

```
layers.4.ffn.experts.0.w1.weight  dtype=I8      shape=[2048, 2048]
layers.4.ffn.experts.0.w1.scale   dtype=F8_E8M0 shape=[2048, 128]
```

FP4 packed two-per-byte in int8 containers, **E8M0 scales at block size 32** — that is MXFP4.
NVFP4 is E4M3 scales at block size 16. The config's `"scale_fmt": "ue8m0"` agrees, and
DeepSeek's own SGLang recipe uses `--moe-runner-backend flashinfer_mxfp4`.

| component | format |
|---|---|
| MoE expert weights (bulk of the 167 GB) | **MXFP4** — E8M0 scales, block 32 |
| Attention, shared experts, MTP head | **FP8** e4m3, block 128×128 |
| KV cache | **NVFP4** (`--kv-cache-dtype nvfp4_ds_mla`) |

vLLM labels the whole scheme `quantization=deepseek_v4_fp8`.

**DeepSeek publishes exactly one quantization per model.** All V4 instruct/DSpark repos are
FP8 + MXFP4 experts; only the `-Base` repos differ (`expert_dtype: fp8`), which nearly doubles
size — `Flash-Base` is **294.7 GB** vs `Flash-0731` at 166.9 GB. Single `main` branch, no tags,
no alternate-quant revisions. There is **no BF16, NVFP4, or INT4/GGUF release from DeepSeek**.
NVIDIA's `DeepSeek-V4-Flash-NVFP4` covers the *April preview* only — it has no DSpark module
and is no smaller, so there is no NVFP4-weights option for 0731 today.

---

## Findings

### 1. Pipeline parallelism is impossible for this model — two independent failures

Upstream states "no pipeline parallelism" without saying why. Both paths were tested:

**With DSpark** — rejected at config validation. The *draft* model, not the target, is the
blocker:
```
File "vllm/config/speculative.py", line 1169, in _verify_args
    self.draft_model_config.verify_with_parallel_config(...)
NotImplementedError: Pipeline parallelism is not supported for this model.
                    Supported models implement the `SupportsPP` interface.
```

**Without DSpark** — runtime crash, deeper in the stack:
```
File "vllm/models/deepseek_v4/compressor.py", line 372, in forward
ValueError: Invalid state_cache.strides[0] ... expected to be divisible by 16
```
43 layers split across two pipeline ranks gives 22/21, producing a stride that violates the
compressor kernel's 16-element alignment requirement.

**Conclusion: TP=2 is the only working multi-node configuration.** No flag works around this.

### 2. DSpark speculative decoding is unstable under concurrency — there is no safe threshold

> **Corrected 2026-08-06.** An earlier version of this document claimed the ceiling was
> 32 streams. That was wrong — it was a single data point mistaken for a threshold. A second
> run killed the engine at **8** streams. Do not treat any concurrency above 1 as safe with
> DSpark enabled.

The engine dies with `EngineDeadError`. In every crash the scheduler dump shows the drafter
emitting invalid draft token ids — `[-1, -1, -1, -1, -1]` — on a step that mixes prefill and
decode work. `sample_tokens` then wedges until the RPC times out:

```
TimeoutError: RPC call to sample_tokens timed out.
scheduled_spec_decode_tokens={...: [-1,-1,-1,-1,-1], ...}
```

Two crashes, same signature, very different stream counts:

| run | `max_num_seqs` | survived | **died at** | KV in use | requests carrying `[-1,…]` |
|---|---:|---|---:|---:|---|
| A | 48 | 1, 4, 8, 16, 32 | 48 | 7.6% | all 24 running |
| B | 32 | 1, 4 | **8** | 1.5% | 1 of 8 |

In run B the failing step was 7 new prefills scheduled alongside 1 decoding request — a
**mixed prefill/decode batch**, which is also what the upstream "Keys concurrency patch"
notes identify as the fragile path in an older DSpark implementation.

**This is not memory pressure.** KV was 1.5% used in run B and 7.6% in run A; graph capture
fit in 2.89 GiB. It is not a clean function of stream count either — run A survived 32 streams
that run B never reached.

**Practical guidance:**

- **Speculation on** → excellent single-stream speed (~47 tok/s, 1.9× faster), but the engine
  can die at any concurrency above 1. Fine for a single interactive user; **not safe for
  unattended multi-user serving**. Run the watchdog in `systemd/`.
- **Speculation off** (`SPEC_DECODE=off`) → the only configuration that completed a full
  1→48-stream sweep without incident, at ~40% lower throughput.

Choose based on whether you need stability or speed. We have not found a setting that gives
both.

### 3. Speculation is a 1.4–1.9× win and never inverts

Headline numbers are in "What to expect" above; this is the full A/B with speedup ratios.

Expected wisdom is that speculation stops paying once the batch saturates the GPUs. It does
not here — measured at every level, same hardware, same prompts, only `SPEC_DECODE` differing:

**Per-stream decode tok/s**

| streams | spec ON | spec OFF | speedup |
|---:|---:|---:|---:|
| 1 | 47.31 | 24.67 | **1.92×** |
| 4 | 25.95 | 17.08 | 1.52× |
| 8 | 18.33 | 12.95 | 1.42× |
| 16 | 13.64 | 9.62 | 1.42× |
| 32 | 13.46 | 9.35 | 1.44× |
| 48 | *engine died* | 7.97 | — |

**Aggregate server tok/s**

| streams | spec ON | spec OFF |
|---:|---:|---:|
| 1 | 46.8 | 24.5 |
| 4 | 98.4 | 64.0 |
| 8 | 140.3 | 94.2 |
| 16 | 187.5 | 145.6 |
| 32 | **229.5** | 159.0 |
| 48 | — | 189.8 |

**Spec-on @ 32 streams (229.5 tok/s) beats spec-off @ 48 (189.8).** Going wider without
speculation is strictly worse, so 32 is the throughput optimum, not a compromise.

Cost of speculation is capacity, not quality: KV pool **1.56M vs 2.17M tokens** (−28%), graph
capture **58 s / 2.89 GiB vs 14 s / 1.14 GiB** (6× the capture set), and — more seriously — the instability in Finding #2.

### 4. Speculation is not an accuracy trade-off

Greedy decoding (`temperature=0`, fixed seed), spec-on vs spec-off:

- **4/4 checkable answers correct in both**
- **4/6 outputs byte-identical**; the two differences were cosmetic (`yes` vs `Yes`, different
  phrasing on an open-ended prompt) — floating-point nondeterminism from different kernel
  paths, not degradation

Consistent with theory: rejection sampling is distribution-preserving. Reproduce with
`scripts/quality_probe.py`.

### 5. The 1M context is real

130,029-token needle-retrieval probe returned the buried code correctly:
**TTFT 71 s, 1,831 tok/s prefill.** KV pool 1.56M tokens > 1M, so a full-length request fits
with ~1.4× headroom. Reproduce with `scripts/ctx_probe.py`.

### 6. Client timeouts on long contexts (this bites everyone)

Prefill runs ~1,831 tok/s at 130K and slows as context grows (~875 tok/s near 900K), so a full
1M-token prefill takes **~17–19 minutes**. Claude Code's `API_TIMEOUT_MS` defaults to
**600000 (10 min)**, so the client errors while the server finishes anyway and caches the
result — the retry then hits the warm prefix cache and returns instantly. That is the
"API error → retry → works" pattern.

```bash
export API_TIMEOUT_MS=72000000    # 20 h
```
Tested: **`-1` hard-errors** (`timeout must be a positive integer`) and **`0` is undefined**
(does not fail fast, but is not a documented "disable"). Use a large finite value; the
documented ceiling is `2147483647`, above which the timer overflows and requests fail
immediately.

### 7. Keep weights and KV resident

On unified memory, a swapped-out page is a multi-second stall mid-generation. One node had the
stock `vm.swappiness=60` and had silently pushed **1.8 GB of the vLLM stack to disk**. Set on
**both** nodes:

```bash
echo 'vm.swappiness = 0' | sudo tee /etc/sysctl.d/99-llm-serving.conf
sudo sysctl -p /etc/sysctl.d/99-llm-serving.conf
sudo swapoff -a && sudo swapon -a     # reclaim what is already out (needs free RAM)
```

---

## Gotcha that invalidates every TP benchmark

The upstream example names `NCCL_IB_HCA=rocep1s0f1`. **Verify yours** — if the HCA does not
match the interface actually carrying the inter-node subnet, NCCL silently falls back to the
management network (often 1 GbE) and every tensor-parallel number is garbage, with no error.

```bash
ibdev2netdev                      # map HCA -> netdev
ip route get <WORKER_ROCE_IP>     # which interface actually reaches the worker
show_gids | grep <your-hca>       # RoCEv2 GID index (differs per node; do not pin one value)
```

Prove the fabric before trusting any number:
```bash
# on worker:  ib_write_bw -d <hca> -x <gid> -F --report_gbits
ib_write_bw -d <hca> -x <gid> -F --report_gbits <WORKER_ROCE_IP>
```
Expect ~100 Gb/s+ on a 200 GbE link (single QP). A result in the single digits means you are
on the wrong interface.

---

## Setup

1. Clone the upstream recipe and apply the parallelism patch:
   ```bash
   git clone https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark /opt/dsv4/recipe
   cd /opt/dsv4/recipe && git apply /path/to/docker-compose.dspark.parallelism.patch
   ```
   The patch makes `--tensor-parallel-size` / `--pipeline-parallel-size` configurable (upstream
   hardcodes TP=2/PP=1) and adds a `SPEC_DECODE=on|off` toggle. Both are required to reproduce
   the comparisons above.

2. Copy `.env.dspark.example` to `/opt/dsv4/recipe/.env.dspark` on **both** nodes and fill in
   your IPs and HCA names.

3. Download the checkpoint to the **same path on both nodes** (~167 GB each):
   ```bash
   hf download deepseek-ai/DeepSeek-V4-Flash-0731 \
     --revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb \
     --local-dir /data/models/deepseek-v4-flash-0731 --max-workers 8
   ```
   Copying node→node over the high-speed link is much faster than downloading twice.

4. Pull the image on both nodes, then start:
   ```bash
   docker pull ghcr.io/anemll/dspark-vllm-gx10:0.1.1
   cd /opt/dsv4/recipe && ./start-deepseek-v4-flash-dspark.sh
   ```
   Startup is ~6–8 min (167 GB load + CUDA graph capture).

5. Optional — install the systemd units for boot survival and a watchdog (see `systemd/`).

---

## systemd

`systemd/` contains units that survive reboot, with three non-obvious details baked in:

- **Boot ordering.** The head node drives the worker over SSH; on a cold rack boot the head can
  be ready first. The unit waits up to 5 min for the worker before failing.
- **Cleanup before start.** The upstream launcher refuses to run if containers or the port are
  in use, so a plain `restart` fails without a preceding stop.
- **`TimeoutStartSec=1800`.** systemd's 90 s default kills the unit mid-weight-load every time.

The watchdog (`deepseek-healthcheck.{sh,service,timer}`) exists because the cluster unit is
`Type=oneshot` + `RemainAfterExit` — systemd cannot see an engine that died inside a container
that is still running. It probes `/v1/models` every 2 min and restarts after 3 consecutive
failures. Safe under load: `/v1/models` is served by the API process independently of the
engine and answered in **0.002 s** while the engine was mid-generation, so a busy server is
never mistaken for a wedged one.

---

## Reproducing the measurements

```bash
python3 scripts/sweep.py --concurrency 1,4,8,16,32,48 --max-tokens 1024   # throughput sweep
python3 scripts/ctx_probe.py --tokens 131072 --depth 0.5                  # long-context needle
python3 scripts/quality_probe.py --label spec-on --out spec-on.json       # quality A/B
./scripts/profile.sh set SPEC_DECODE=off && sudo systemctl restart deepseek-vllm
```

`sweep.py` gives each stream a distinct prompt so `--enable-prefix-caching` cannot make later
streams reuse earlier prefill work, and discards a warmup pass before measuring.

**Caveats on our numbers, so you can judge them:** TTFT at 8 and 16 streams is inflated by
Triton/CuTeDSL kernels JIT-compiling *during* inference (one warmup pass did not cover every
batch shape) — throughput columns are unaffected since they measure the post-first-token decode
window. Single-stream throughput is measured at `max_num_seqs=48`; a smaller value trades
concurrency for single-stream speed.

---

## Credits

- **[MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)**
  (MIT, © 2026 Tony Deangelo) — the launcher, compose stack, and original 2×Spark recipe this
  builds on.
- **[Anemll/dspark-vllm-gx10](https://github.com/Anemll/dspark-vllm-gx10)** — the GB10/sm_121
  vLLM port with native DSpark, NVFP4 DS-MLA, and b12x MoE.
- **[DeepSeek-AI](https://huggingface.co/deepseek-ai)** — the model (MIT).

See `NOTICE` for full attribution. MIT licensed; see `LICENSE`.
