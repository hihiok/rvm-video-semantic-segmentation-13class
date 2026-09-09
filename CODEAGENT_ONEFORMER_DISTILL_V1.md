# CodeAgent: continue RVM13 with guarded OneFormer distillation

## Mission and immutable inputs

Work only in this repository and branch:

```text
repo:   https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
branch: agent/rvm13-oneformer-distill-v1
```

The student must start from the already-trained residual-v1 RVM13 run:

```text
/data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class-rvm-residual-v1/output/rvm_vspw_rvm_residual_v1_13class_640x360
```

Teacher and datasets are fixed:

```text
teacher: shi-labs/oneformer_ade20k_swin_large
static:  /data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360
video:   /data/pub1/z00919662/segmentation/datasets/VSPW_13cls
```

Do not replace the RVM student, change class order, broaden the mapping, train
from random weights, or use a checkpoint from another run. Do not launch a full
job until every preflight below passes.

## Proxy and SSL/TLS (mandatory preflight)

Keep TLS certificate verification enabled. Never set `GIT_SSL_NO_VERIFY=true`,
`CURL_SSL_NO_VERIFY`, `verify=False`, or an empty CA bundle. Never write proxy
credentials into this repository, logs, shell history, or commits.

If this server needs a proxy, obtain it through the server's approved secret
mechanism and keep shell tracing disabled. The following uses a placeholder,
not a credential:

```bash
set +x
export CODEAGENT_PROXY_URL='http://USER:PASSWORD@PROXY_HOST:PORT'
export http_proxy="${CODEAGENT_PROXY_URL}"
export https_proxy="${CODEAGENT_PROXY_URL}"
export HTTP_PROXY="${CODEAGENT_PROXY_URL}"
export HTTPS_PROXY="${CODEAGENT_PROXY_URL}"
export GIT_SSL_NO_VERIFY=false

# For an organization CA, point both clients at the approved PEM file.
# export SSL_CERT_FILE=/approved/path/company-ca-bundle.pem
# export REQUESTS_CA_BUNDLE="${SSL_CERT_FILE}"
# export CURL_CA_BUNDLE="${SSL_CERT_FILE}"

python - <<'PY'
import ssl
print(ssl.get_default_verify_paths())
PY
git config --get http.sslVerify || true
```

Unset `CODEAGENT_PROXY_URL` after network downloads if policy requires it. When
proxy or certificate validation fails, stop with `HUMAN_ACTION_REQUIRED` and
the sanitized error; do not bypass TLS verification.

## Checkout and exact revision

```bash
set -euo pipefail
PROJECT=/data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class-rvm-residual-v1
cd "${PROJECT}"
git fetch origin agent/rvm13-oneformer-distill-v1
git switch agent/rvm13-oneformer-distill-v1 || \
  git switch -c agent/rvm13-oneformer-distill-v1 --track origin/agent/rvm13-oneformer-distill-v1
git pull --ff-only origin agent/rvm13-oneformer-distill-v1
git status --short --branch
```

If tracked local edits exist, stop and report them. Do not reset, discard, or
overwrite someone else's work.

## Environment and deterministic verification

Use the project's existing CUDA environment if compatible; otherwise create an
isolated environment. Install the distillation dependencies and run:

```bash
python -m pip install -r requirements-oneformer-distill.txt
python -m compileall -q distillation dataset tools train_vspw_mixed.py
bash -n scripts/cache_oneformer_teacher.sh scripts/train_oneformer_distill.sh
git diff --check
pytest -q
python tools/profile_student.py
```

Required profile result: MobileNetV3 RVM, 13 classes, exactly 3,747,961
parameters. Record both compute conventions printed by the tool; do not claim
that 90G and direct 2-FLOP-per-MAC numbers use the same convention.

## Validate the initialization checkpoint

```bash
CHECKPOINT_DIR=/data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class-rvm-residual-v1/output/rvm_vspw_rvm_residual_v1_13class_640x360
python tools/select_distillation_checkpoint.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --report-json /tmp/rvm13_distillation_checkpoint_report.json
```

The selector must validate the exact 13-class head and class order. It should
choose `best_spatial_preserved.pth` when valid, followed only by the documented
best-checkpoint order inside the same directory. It intentionally refuses to
select `last.pth`. Inspect `metrics.csv` and the JSON report; if the selected
file is obviously corrupt or inconsistent with the run's best epoch, stop with
`HUMAN_ACTION_REQUIRED` rather than guessing.

## Cache the soft teacher targets

The uploaded example uses Hugging Face's hard-label semantic post-processing.
Do not use that hard mask as KD supervision. This branch composes soft semantic
scores from OneFormer class-query softmax and mask-query sigmoid, undoes
processor padding, maps the 150 ADE20K classes to 13, and stores a resumable
uint8 `13x45x80` probability cache.

Before the full cache, run a small smoke test on copied/symlinked sample images
outside the datasets. Then generate and audit the complete train caches:

```bash
export VSPW_ROOT=/data/pub1/z00919662/segmentation/datasets/VSPW_13cls
export STATIC_ROOT=/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360
export TEACHER_CACHE_ROOT=/data/pub1/z00919662/segmentation/datasets/OneFormer_ADE20K_RVM13_cache_45x80
export DEVICE=cuda:0
export TEACHER_BATCH_SIZE=1
bash scripts/cache_oneformer_teacher.sh 2>&1 | tee /tmp/rvm13_oneformer_cache.log
```

Both audits must report `coverage: 1.0`, no invalid entries, finite normalized
probabilities, and non-collapsed class mass. A second cache command must skip
existing entries safely. Never write teacher outputs into the image or
annotation directories.

## End-to-end smoke training

Run one epoch in each stage on a small subset. Use a new temporary output
directory and the validated best checkpoint as `--init-checkpoint`; an existing
residual-v1 `last.pth` is initialization, not `--resume` for this new KD run.

```bash
INIT_CHECKPOINT="$(python tools/select_distillation_checkpoint.py \
  --checkpoint-dir "${CHECKPOINT_DIR}" --print-path-only)"
python -u train_vspw_mixed.py \
  --data-root "${VSPW_ROOT}" --static-root "${STATIC_ROOT}" \
  --teacher-cache-root "${TEACHER_CACHE_ROOT}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --output-dir /tmp/rvm13_oneformer_distill_smoke \
  --stage2-epochs 1 --stage3-epochs 1 \
  --stage2-clip-length 2 --stage3-clip-length 2 \
  --stage2-kd-weight 0.50 --stage3-kd-weight 0.20 \
  --stage2-temporal-weight 0.03 --stage3-temporal-weight 0.08 \
  --max-train-clips 2 --max-val-clips 2 \
  --max-static-train-images 2 --max-static-val-images 2 \
  --batch-size 1 --static-batch-size 1 --workers 0
```

Pass conditions:

- checkpoint compatibility is at least 80%, with a compatible 13-class head;
- both stages finish and write `last.pth` plus metrics;
- supervised, KD, and temporal losses are finite;
- KD selected-pixel counts are nonzero in both domains;
- validation runs on VSPW and static data;
- the output checkpoint records `oneformer_distillation_training: true`.

## Full training and monitoring

```bash
export CHECKPOINT_DIR=/data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class-rvm-residual-v1/output/rvm_vspw_rvm_residual_v1_13class_640x360
export OUTPUT_DIR=/data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class-rvm-residual-v1/output/rvm_oneformer_distill_v1_13class_640x360
export MAX_GPUS=2
bash scripts/train_oneformer_distill.sh 2>&1 | tee "${OUTPUT_DIR}.log"
```

The launcher uses low learning rates, supervised CE+Dice, guarded KD, static
replay, and causal temporal consistency. Monitor `metrics.csv` for VSPW mIoU,
static mIoU, stable-GT prediction flip rate, KD coverage/agreement, and finite
losses. Prefer `best_spatial_preserved.pth` for deployment. If static mIoU falls
more than the configured 0.02 tolerance or flip rate regresses materially,
reduce KD/LR and investigate; do not simply promote `last.pth`.

For an interrupted distillation run only, resume with:

```bash
RESUME="${OUTPUT_DIR}/last.pth" bash scripts/train_oneformer_distill.sh
```

## Stop conditions and report

Stop with `HUMAN_ACTION_REQUIRED` for missing datasets/cache/checkpoints,
taxonomy or tensor mismatch, dirty tracked files, failed TLS validation,
unavailable approved GPU capacity, NaN/Inf, cache coverage below 100%, or a
material validation regression. Report exact sanitized commands, selected
checkpoint, commit SHA, cache audits, profile numbers, smoke/full-run metrics,
and output paths. Do not push new changes unless you actually had to fix code;
if fixes are necessary, rerun all checks, commit only relevant files, and push
only `agent/rvm13-oneformer-distill-v1`.
