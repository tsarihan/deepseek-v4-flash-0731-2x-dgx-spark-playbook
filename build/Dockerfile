# DeepSeek-V4-Flash-0731 runtime for DGX Spark (GB10 / sm_121), built from
# first-party sources only:
#   - base:   nvcr.io/nvidia/vllm:26.07-py3  (NVIDIA, signed, CUDA 13.3, aarch64 torch)
#   - engine: upstream vllm-project/vllm     (compiled here, not downloaded prebuilt)
#
# Why build at all: NVIDIA's prebuilt vLLM targets sm_120 + PTX and lacks DSpark;
# the community image targets sm_120/80/89/90 with runtime JIT for sm_121a and is
# not reproducible (build.commit=unknown). Compiling upstream with CUDA 13 gives
# native sm_121a cubins — vLLM only lists 12.1 as supported when CUDA >= 13.0.

FROM nvcr.io/nvidia/vllm:26.07-py3

ARG VLLM_REF=main
# sm_120 covers GB10 (sm_121) via minor-version cubin compatibility, and we
# MEASURED that sm_120 and sm_121 emit byte-identical SASS — for FP16 wmma and
# for NVFP4 block-scaled MMA alike. Adding 12.1 would duplicate a cubin for zero
# gain, and an sm_121-only build would NOT run on sm_120. +PTX gives a JIT path
# for future archs so nothing can report "no kernel image available".
ARG ARCH_LIST=12.0+PTX

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=${ARCH_LIST} \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=16 \
    NVCC_THREADS=2 \
    CCACHE_DIR=/ccache \
    VLLM_TARGET_DEVICE=cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ccache ninja-build cmake curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# vLLM now has a Rust component (setuptools-rust). The NVIDIA base has no rustc,
# and --no-deps (which protects NVIDIA's torch) also skips setuptools-rust's own
# dependency `semantic_version`. Both must be supplied explicitly or the build
# fails at the packaging step AFTER the full CUDA compile — a 24-minute waste.
ENV PATH=/root/.cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.90.0 && \
    rustc --version && cargo --version

# Remove the prebuilt 0.24 so the source build is unambiguous.
RUN pip uninstall -y vllm || true

WORKDIR /src
RUN git clone --depth 1 --branch ${VLLM_REF} https://github.com/vllm-project/vllm.git
WORKDIR /src/vllm

# Record exactly what we built, so the image is reproducible.
RUN git rev-parse HEAD > /BUILD_COMMIT && cat /BUILD_COMMIT

# CRITICAL: vLLM pins `torch == 2.13.0` but the NVIDIA base ships
# `2.13.0a0+...nv26.07` — an alpha of the same version, built for aarch64/CUDA 13.
# pip does not treat those as equal and will REPLACE NVIDIA's torch with a generic
# wheel, silently discarding the tuned build this image exists for. So strip every
# torch-family pin from the build requirements and never resolve deps.
RUN --mount=type=cache,target=/ccache \
    grep -vE '^\s*(torch|torchvision|torchaudio|pytorch-triton)\b' requirements/build/cuda.txt > /tmp/build-reqs.txt 2>/dev/null \
      || printf 'cmake>=3.26.1\nninja\npackaging>=24.2\nsetuptools>=77.0.3,<81.0.0\nsetuptools-scm>=8.0\nsetuptools-rust>=1.9.0\nwheel\njinja2\n' > /tmp/build-reqs.txt; \
    cat /tmp/build-reqs.txt && pip install --no-deps -r /tmp/build-reqs.txt && \
    pip install --no-deps semantic_version

# Verify NVIDIA's torch survived before spending an hour compiling against it.
RUN python3 -c "import torch; v=torch.__version__; print('torch:',v); assert 'nv' in v, 'NVIDIA torch was replaced — aborting'"

RUN --mount=type=cache,target=/ccache \
    pip install --no-build-isolation --no-deps -e .

# Fail the build loudly if the kernels did not actually target sm_121.
RUN python3 - <<'PYEOF'
import glob, os, subprocess, sys
# Editable installs scatter artifacts; search the source tree and site-packages.
cands = []
for root in ('/src/vllm', '/usr/local/lib/python3.12/dist-packages'):
    cands += glob.glob(os.path.join(root, '**', '*.so'), recursive=True)
cands = [c for c in cands if 'vllm' in c and os.path.getsize(c) > 2*1024*1024]
if not cands:
    print('NO vLLM .so FOUND'); sys.exit(1)
cands.sort(key=os.path.getsize, reverse=True)
found = set()
for so in cands[:5]:
    out = subprocess.run(['cuobjdump','--list-elf',so],capture_output=True,text=True).stdout
    a = {l.split('.sm_')[1].split('.')[0] for l in out.splitlines() if '.sm_' in l}
    if a: print(f"  {os.path.basename(so)}: sm_" + ", sm_".join(sorted(a)))
    found |= a
print('all archs:', 'sm_' + ', sm_'.join(sorted(found)) if found else '(none)')
if not any(x.startswith('121') for x in found):
    print('ERROR: no sm_121 kernels — the build did not target GB10'); sys.exit(1)
print('OK: native sm_121 kernels present')
PYEOF

ENV VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
    CUTE_DSL_ARCH=sm_121a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENTRYPOINT ["vllm"]
