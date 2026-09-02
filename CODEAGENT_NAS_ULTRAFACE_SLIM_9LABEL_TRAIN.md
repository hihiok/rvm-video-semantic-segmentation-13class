# Task: prepare data and train NAS 9-label multi-label scene classifier with UltraFace slim backbone

## Goal

Use the code already prepared in this GitHub branch to build a 9-label multi-label scene classifier for NAS photo tagging.

Target labels:

1. indoor / 室内
2. outdoor / 户外
3. landscape / 风景
4. sports / 运动
5. food / 美食
6. animal / 动物
7. building / 建筑
8. sky / 蓝天（product name; semantic definition is any visibly present sky, not only blue sky）
9. office / 办公

The network is the UltraFace `Mb_Tiny` / slim convolutional backbone with RFB, SSD extras, bbox heads and face-confidence heads removed. It uses global average pooling and 9 independent sigmoid outputs.

Default model configuration:

```text
input: 224x224 RGB
base_channel: 16
parameters: about 0.173M
MACs: about 48.3M/image
output: 9 logits
```

This is a multi-label task. Never replace it with a 9-way softmax.

## Hard restrictions

- Do not write or modify Python training/data/model code locally.
- Do not change class definitions or mappings.
- Do not convert unknown label `-1` to negative `0`.
- Do not change the network into RFB or add SSD detection heads.
- Do not train RVM in this task.
- Do not modify the existing 13-class segmentation dataset or masks.
- Do not overwrite existing RVM checkpoints/output.
- If prepared code has a real bug, stop and report traceback + exact file/line. Do not patch locally.
- If a code change is necessary, report it to the user so ChatGPT can update GitHub first.
- Do not use more than 4 DataLoader workers. CPU usage must remain limited.
- Use one free GPU only. This model is too small for multi-GPU training to be useful.

## Proxy and SSL

The server is behind a corporate proxy / HTTPS inspection environment.

Use the proxy variables already exported in the shell/session. Do not echo proxy credentials and do not commit them to GitHub.

Before git/pip/wget operations:

```bash
set +x

if [[ -z "${http_proxy:-}" ]]; then
  echo "HUMAN_ACTION_REQUIRED: corporate http_proxy is not exported in this shell"
  exit 2
fi

export https_proxy="${https_proxy:-${http_proxy}}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"

git config --global http.proxy "${http_proxy}"
git config --global https.proxy "${https_proxy}"

git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=1
export HF_HUB_DISABLE_SSL_VERIFICATION=1

set -x
```

The prepared dataset downloader already uses `wget --no-check-certificate`.

Do not run `env | grep -i proxy` or otherwise print credentials into logs.

If the shell has no proxy variables, stop and report `HUMAN_ACTION_REQUIRED: YES`. The user must export the corporate proxy in the shell.

## Repository

```bash
export REPO_URL="https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git"
export BRANCH="agent/nas-ultraface-slim-9label-v1"
export PROJECT_ROOT="/data/pub1/z00919662/scene_multilabel/nas-ultraface-slim-9label"
```

If missing:

```bash
git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${PROJECT_ROOT}"
```

If it exists:

```bash
cd "${PROJECT_ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: local Git changes exist; do not overwrite"
  git status --short
  exit 2
fi

git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git merge --ff-only "origin/${BRANCH}"
```

Verify:

```bash
cd "${PROJECT_ROOT}"
git branch --show-current
git rev-parse HEAD

test -f nas_scene_multilabel/model.py
test -f nas_scene_multilabel/config.py
test -f nas_scene_multilabel/prepare_dataset.py
test -f nas_scene_multilabel/train.py
test -f nas_scene_multilabel/export_onnx.py
test -f nas_scene_multilabel/scripts/download_datasets.sh
test -f nas_scene_multilabel/scripts/run_train.sh
```

Expected branch:

```text
agent/nas-ultraface-slim-9label-v1
```

## Existing 13-class data

This dataset should already exist and must NOT be modified:

```bash
export SEG_ROOT="/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360"

test -d "${SEG_ROOT}/images/train"
test -d "${SEG_ROOT}/annotations/train"
test -d "${SEG_ROOT}/images/val"
test -d "${SEG_ROOT}/annotations/val"
```

Its semantic mapping used by the prepared code is:

```text
sky      = class 1
building = class 4
food     = class 6
ignore   = 255
```

For image-level classification supervision:

- sky positive if sky area >= 1%; zero sky pixels = negative; tiny nonzero sky below threshold = unknown
- building positive if building area >= 2%; zero building pixels = negative; tiny nonzero region = unknown
- food positive if food area >= 1%; zero food pixels = negative; tiny nonzero region = unknown

Do not modify these rules.

## Raw dataset paths

```bash
export RAW_ROOT="/data/pub1/z00919662/scene_multilabel/datasets_raw"
export PLACES_ROOT="${RAW_ROOT}/places365"
export COCO_ROOT="${RAW_ROOT}/coco2017"
export DATA_ROOT="/data/pub1/z00919662/scene_multilabel/nas_9label_partial_gt"
export OUTPUT_DIR="/data/pub1/z00919662/scene_multilabel/ultraface_slim_9label/output"
```

## Disk-space check

The task may need to download approximately:

```text
Places365 train_256: ~24 GB
Places365 val_256:   ~0.5 GB
COCO train2017:      ~18 GB
COCO val2017:        ~1 GB
COCO annotations:    ~0.24 GB
```

Archives plus extracted images temporarily require substantially more space.

Before downloading:

```bash
df -h /data/pub1
```

Require at least 80 GB free under the target filesystem.

If less than 80 GB is available:

```text
HUMAN_ACTION_REQUIRED: YES
Reason: insufficient free disk space for Places365 + COCO download/extraction
```

Stop. Do not delete unrelated user data.

## Python environment

Reuse an existing CUDA PyTorch environment. Do not replace or upgrade the current working torch/torchvision stack.

Inspect:

```bash
which python || true
python - <<'PY' || true
try:
    import torch, torchvision
    print("python ok")
    print("torch", torch.__version__)
    print("torchvision", torchvision.__version__)
    print("cuda", torch.cuda.is_available())
    print("gpu_count", torch.cuda.device_count())
except Exception as e:
    print("ENV_CHECK_FAILED", repr(e))
PY

command -v conda || true
conda env list || true
nvidia-smi
```

Prefer the existing environment already proven to train RVM or other PyTorch models.

Create an isolated venv inheriting system packages so torch/torchvision are reused:

```bash
export VENV_DIR="${PROJECT_ROOT}/.venv_nas_scene9"
BASE_PYTHON="$(command -v python)"

"${BASE_PYTHON}" -c 'import torch, torchvision; assert torch.cuda.is_available(); print(torch.__version__, torchvision.__version__)'

if [[ ! -d "${VENV_DIR}" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"

python -m pip install -r "${PROJECT_ROOT}/nas_scene_multilabel/requirements.txt" \
  --upgrade-strategy only-if-needed \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org
```

Verify torch was not replaced:

```bash
python - <<'PY'
import torch, torchvision, sklearn, PIL, numpy
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda", torch.cuda.is_available())
assert torch.cuda.is_available()
PY
```

If no working CUDA PyTorch environment exists, stop and report instead of constructing an arbitrary CUDA environment.

## Syntax checks

```bash
cd "${PROJECT_ROOT}/nas_scene_multilabel"
python -m compileall -q .
bash -n scripts/download_datasets.sh
bash -n scripts/run_train.sh
```

If any prepared file fails syntax checking, stop and report. Do not edit it locally.

## Model-size verification

```bash
cd "${PROJECT_ROOT}/nas_scene_multilabel"
python tools/model_info.py --input-size 224 --base-channel 16
```

Expected approximately:

```text
params = 172521 (~0.173M)
MACs   = 48.3M/image @224x224
```

Minor differences in printed formatting are fine. A large architecture difference is not fine.

## Download Places365 and COCO

The previous MobileCLIP test showed Places365 and COCO are not present locally, so download them now.

The prepared downloader is resumable:

```bash
mkdir -p "${RAW_ROOT}"
cd "${PROJECT_ROOT}/nas_scene_multilabel"

bash scripts/download_datasets.sh "${RAW_ROOT}" \
  2>&1 | tee "${RAW_ROOT}/download.log"
```

The script downloads official 256px Places365-Standard train/val and COCO 2017 train/val + instance annotations.

Do not download Places365 large/original-resolution 105 GB training archive. The 256px train archive is sufficient because the classifier input is 224x224.

After download verify:

```bash
test -d "${PLACES_ROOT}/data_256"
test -d "${PLACES_ROOT}/val_256"
test -f "${PLACES_ROOT}/categories_places365.txt"
test -f "${PLACES_ROOT}/IO_places365.txt"
find "${PLACES_ROOT}" -type f -name places365_val.txt -print -quit

test -d "${COCO_ROOT}/train2017"
test -d "${COCO_ROOT}/val2017"
test -f "${COCO_ROOT}/annotations/instances_train2017.json"
test -f "${COCO_ROOT}/annotations/instances_val2017.json"
```

If a download is interrupted, rerun the same script; `wget -c` resumes partial files.

## Prepare the partial-label dataset

Do not copy or resize the raw datasets. The prepared dataset is a set of JSONL manifests pointing at the original images.

```bash
rm -rf "${DATA_ROOT}"
mkdir -p "${DATA_ROOT}"

cd "${PROJECT_ROOT}/nas_scene_multilabel"
python -u prepare_dataset.py \
  --places-root "${PLACES_ROOT}" \
  --coco-root "${COCO_ROOT}" \
  --seg-root "${SEG_ROOT}" \
  --output-root "${DATA_ROOT}" \
  --places-train-cap-per-class 500 \
  --seed 20260902 \
  2>&1 | tee "${DATA_ROOT}/prepare.log"
```

Expected manifests:

```bash
test -f "${DATA_ROOT}/train.jsonl"
test -f "${DATA_ROOT}/val.jsonl"
test -f "${DATA_ROOT}/test.jsonl"
test -f "${DATA_ROOT}/dataset_summary.json"
```

Print:

```bash
cat "${DATA_ROOT}/dataset_summary.json"
wc -l "${DATA_ROOT}/train.jsonl" "${DATA_ROOT}/val.jsonl" "${DATA_ROOT}/test.jsonl"
```

The builder has a hard coverage guard. Every one of the 9 labels must have both positive and negative supervision in train, val and test.

Expected data responsibilities:

```text
Places365:
  indoor, outdoor, landscape, sports, office

COCO2017:
  food, animal

Existing SEG13:
  building, sky, food
```

Unknown labels remain `-1` and are excluded from the BCE loss and metrics.

If coverage fails, stop and return `dataset_summary.json`. Do not lower the coverage requirement and do not modify mappings.

## Model definition

The prepared model is:

```text
UltraFace Mb_Tiny/slim convolution stack
  -> no RFB
  -> no SSD extras
  -> no bbox regression
  -> no face confidence head
  -> AdaptiveAvgPool2d(1)
  -> Dropout(0.1)
  -> Linear(256, 9)
  -> 9 independent logits
```

Loss is masked multi-label BCEWithLogits with per-class positive weights computed from train supervision.

Input preprocessing:

```text
Train:
  RandomResizedCrop 224x224
  horizontal flip
  moderate color jitter
  light grayscale
  normalize x to roughly [-1, 1]
  light random erasing

Val/Test:
  direct Resize to 224x224
  normalize x to roughly [-1, 1]
```

## GPU selection

Use exactly one free GPU.

```bash
nvidia-smi
```

Pick a genuinely free GPU; example only:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Do not kill other processes.

## Smoke test

Run a short 1-epoch smoke in a separate output directory.

```bash
export SMOKE_OUTPUT="/data/pub1/z00919662/scene_multilabel/ultraface_slim_9label/smoke"
rm -rf "${SMOKE_OUTPUT}"
mkdir -p "${SMOKE_OUTPUT}"

cd "${PROJECT_ROOT}/nas_scene_multilabel"
python -u train.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${SMOKE_OUTPUT}" \
  --input-size 224 \
  --base-channel 16 \
  --epochs 1 \
  --batch-size 256 \
  --workers 2 \
  --lr 0.001 \
  --cpu-threads 4 \
  --print-every 100 \
  2>&1 | tee "${SMOKE_OUTPUT}/smoke.log"
```

Smoke must confirm:

```text
CUDA training works
model params ~0.173M
loss finite, no NaN
validation runs
best_macro_f1.pth generated
last.pth generated
best_deploy.pth generated after final validation/test
```

If batch 256 causes OOM, lower only runtime batch size to 128/64. Do not change network or input size.

## Formal training

Default formal configuration:

```text
input_size        = 224
base_channel      = 16
batch_size        = 256
workers           = 4 maximum
epochs            = 60
optimizer         = AdamW
lr                = 1e-3
weight_decay      = 1e-4
warmup            = 2 epochs
scheduler         = cosine
AMP               = FP16
CPU threads       = 4
```

Start foreground first long enough to confirm normal operation, then use nohup for the formal job.

```bash
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_ROOT}/nas_scene_multilabel"

export RAW_ROOT PLACES_ROOT COCO_ROOT SEG_ROOT DATA_ROOT OUTPUT_DIR
export INPUT_SIZE=224
export BASE_CHANNEL=16
export BATCH_SIZE=256
export WORKERS=4
export EPOCHS=60
export LR=0.001
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

nohup bash scripts/run_train.sh \
  > "${OUTPUT_DIR}/train.log" 2>&1 &

PID=$!
echo "${PID}" > "${OUTPUT_DIR}/train.pid"
echo "TRAIN_PID=${PID}"
```

Check after launch:

```bash
PID="$(cat "${OUTPUT_DIR}/train.pid")"
ps -p "${PID}" -f
nvidia-smi
tail -n 100 "${OUTPUT_DIR}/train.log"
```

Do not report `TRAINING_STARTED` unless batches are actually running and GPU memory/utilization correspond to this process.

## CPU usage guard

Because the server can become unstable with excessive CPU usage:

```text
WORKERS must stay <= 4
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
single GPU only
```

Check periodically:

```bash
ps -o pid,ppid,%cpu,%mem,cmd -p "$(cat "${OUTPUT_DIR}/train.pid")"
```

Do not increase workers simply to improve GPU utilization.

## Resume after interruption/reboot

If training is interrupted and `last.pth` exists:

```bash
cd "${PROJECT_ROOT}/nas_scene_multilabel"
export RESUME="${OUTPUT_DIR}/last.pth"

nohup bash scripts/run_train.sh \
  > "${OUTPUT_DIR}/train_resume.log" 2>&1 &

PID=$!
echo "${PID}" > "${OUTPUT_DIR}/train.pid"
```

Do not restart from epoch 0 when a valid `last.pth` exists.

## Final outputs

Expected:

```text
${OUTPUT_DIR}/last.pth
${OUTPUT_DIR}/best_macro_f1.pth
${OUTPUT_DIR}/best_deploy.pth
${OUTPUT_DIR}/metrics.jsonl
${OUTPUT_DIR}/best_val_per_class_0p5.csv
${OUTPUT_DIR}/thresholds.json
${OUTPUT_DIR}/test_per_class_calibrated.csv
${OUTPUT_DIR}/test_summary.json
${OUTPUT_DIR}/ultraface_slim_9label_224.onnx
${OUTPUT_DIR}/ultraface_slim_9label_224.json
```

`best_deploy.pth` contains the per-label validation-calibrated thresholds.

Threshold calibration is done on `val` only. Final reported metrics are from `test`.

## Accuracy interpretation

The product requirement says recognition accuracy >95%. For a multi-label task, do not use overall accuracy alone.

Report for every label:

```text
precision
recall
F1
accuracy
balanced_accuracy
average precision (AP)
calibrated threshold
known positive count
known negative count
```

Also report:

```text
macro-F1
macro-balanced-accuracy
macro-AP
```

A class should not be considered healthy merely because raw accuracy is high when negatives dominate.

## If base_channel=16 is not accurate enough

Do NOT automatically modify the model in this task.

First report the full 9-class test metrics for the exact UltraFace-slim-width model.

If accuracy/F1 is insufficient, the next controlled experiment can widen the same topology to `base_channel=24`. Do not start that experiment unless the user asks for it.

## Final report format

Return:

```text
STATUS: PASS / FAIL / TRAINING_STARTED

GITHUB_BRANCH:
GITHUB_COMMIT:

PYTHON_VERSION:
TORCH_VERSION:
TORCHVISION_VERSION:
GPU:
CUDA_VISIBLE_DEVICES:

PLACES_ROOT:
COCO_ROOT:
SEG_ROOT:
DATA_ROOT:
OUTPUT_DIR:

MODEL:
INPUT_SIZE:
BASE_CHANNEL:
PARAMS:
MACS_224:

DATASET_COUNTS:
train total:
val total:
test total:

PER_LABEL_SUPERVISION:
label | train pos/neg | val pos/neg | test pos/neg

SMOKE_STATUS:

FORMAL_TRAIN_STATUS:
TRAIN_PID:
CURRENT_EPOCH:

BEST_VAL_MACRO_F1_0P5:

FINAL_TEST_METRICS:
label | threshold | precision | recall | F1 | accuracy | balanced_accuracy | AP

TEST_MACRO_F1:
TEST_MACRO_BALANCED_ACCURACY:
TEST_MACRO_AP:

BEST_CHECKPOINT:
DEPLOY_CHECKPOINT:
ONNX_PATH:
TRAIN_LOG:

WARNINGS:
HUMAN_ACTION_REQUIRED: YES / NO
```

If all work can proceed automatically:

```text
HUMAN_ACTION_REQUIRED: NO
```

If proxy variables are missing, disk space is insufficient, existing segmentation data is missing, or prepared code has a bug:

```text
HUMAN_ACTION_REQUIRED: YES
```

State the exact action the user must perform.
