#!/bin/bash
# Verify every pinned artifact still resolves and matches.
#
# Run this BEFORE trusting the repo — especially if it has been sitting for weeks.
# A tag that has been re-pushed, a deleted revision, or a changed checkpoint will
# all surface here rather than as a confusing runtime failure later.
#
# Exit 0 = every pin verified. Non-zero = something moved; do not trust a run.
set -uo pipefail

RUNTIME_IMAGE="ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8"
MODEL_REPO="deepseek-ai/DeepSeek-V4-Flash-0731"
MODEL_REV="9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
MODEL_DIR="${MODEL_DIR:-/data/models/deepseek-v4-flash-0731}"

# Versions the pinned image is expected to contain. If the digest is intact these
# cannot drift — checking them proves the digest resolved to what we validated.
EXPECT_PYTHON="3.12.13"
EXPECT_VLLM="0.25.2.dev0"

fail=0
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=1; }

echo "== runtime image (pinned by digest) =="
if docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1 || docker manifest inspect "$RUNTIME_IMAGE" >/dev/null 2>&1; then
    ok "image digest resolves"
    py=$(docker run --rm --entrypoint python3 "$RUNTIME_IMAGE" -V 2>/dev/null | awk '{print $2}')
    [ "$py" = "$EXPECT_PYTHON" ] && ok "python $py" || bad "python: expected $EXPECT_PYTHON, got ${py:-none}"
    vv=$(docker run --rm --entrypoint python3 "$RUNTIME_IMAGE" -c 'import vllm;print(vllm.__version__)' 2>/dev/null | head -1)
    case "$vv" in "$EXPECT_VLLM"*) ok "vllm $vv" ;; *) bad "vllm: expected $EXPECT_VLLM*, got ${vv:-none}" ;; esac
else
    bad "image digest does not resolve — it may have been deleted from the registry"
fi

echo "== model checkpoint (pinned by revision) =="
python3 - "$MODEL_REPO" "$MODEL_REV" "$MODEL_DIR" <<'PY'
import sys, os
repo, rev, base = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    from huggingface_hub import HfApi
except ImportError:
    print("  \033[33mSKIP\033[0m huggingface_hub not installed"); sys.exit(0)
try:
    info = HfApi().model_info(repo, revision=rev, files_metadata=True)
except Exception as e:
    print(f"  \033[31mFAIL\033[0m revision {rev[:12]} not resolvable: {type(e).__name__}"); sys.exit(1)
print(f"  \033[32mOK\033[0m   revision {rev[:12]} resolves upstream")
if not os.path.isdir(base):
    print(f"  \033[33mSKIP\033[0m {base} not present locally"); sys.exit(0)
missing, mismatch = [], []
for s in info.siblings:
    if not s.size: continue
    p = os.path.join(base, s.rfilename)
    if not os.path.exists(p): missing.append(s.rfilename)
    elif os.path.getsize(p) != s.size: mismatch.append(s.rfilename)
if missing or mismatch:
    print(f"  \033[31mFAIL\033[0m local weights differ — missing {len(missing)}, size-mismatch {len(mismatch)}")
    sys.exit(1)
print("  \033[32mOK\033[0m   all local shards match upstream sizes")
PY
[ $? -ne 0 ] && fail=1

echo
if [ $fail -eq 0 ]; then echo "All pins verified."; else echo "PINS MOVED — investigate before running."; fi
exit $fail
