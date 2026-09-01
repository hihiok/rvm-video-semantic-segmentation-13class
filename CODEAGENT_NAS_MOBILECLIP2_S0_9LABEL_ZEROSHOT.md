# Task: run MobileCLIP2-S0 9-label zero-shot NAS photo tagging benchmark

## Goal

Run the code already prepared in this GitHub branch. Do not write or modify Python code.

Evaluate `MobileCLIP2-S0` zero-shot on these nine independent NAS photo tags:

1. indoor / 室内
2. outdoor / 户外
3. landscape / 风景
4. sports / 运动
5. food / 美食
6. animal / 动物
7. building / 建筑
8. sky / 蓝天（product name; semantic definition is visible sky of any color）
9. office / 办公

The test set must contain reliable known positives and negatives for all nine labels. Use existing datasets on the server, preferring Places365 + COCO + the existing 13-class semantic dataset.

## Hard restrictions

- DO NOT modify any `.py`, `.sh`, `.md`, config, prompt, label mapping, or model code.
- DO NOT create replacement scripts when a command fails.
- DO NOT change the nine class definitions.
- DO NOT convert unknown GT (`-1`) to negative (`0`).
- DO NOT use a 9-way softmax. This is multi-label tagging.
- DO NOT tune thresholds on the test GT and report the result as zero-shot.
- DO NOT alter existing RVM datasets, masks, checkpoints, or training outputs.
- DO NOT start any RVM training.
- If code has a real bug, stop and report the traceback plus the exact file/line. Do not patch it locally.
- If a code change becomes necessary, report it to the user so the change can be synchronized through GitHub first.

## Proxy and SSL

This server may require the corporate proxy and SSL verification bypass. Never print proxy credentials into logs or commit them to GitHub.

Reuse the proxy values already exported in the shell/session:

```bash
set +x

test -n "${http_proxy:-}" || {
  echo "HUMAN_ACTION_REQUIRED: http_proxy is not exported in this shell"
  exit 2
}

export https_proxy="${https_proxy:-${http_proxy}}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"

git config --global http.proxy "${http_proxy}"
git config --global https.proxy "${https_proxy}"

# Internal HTTPS inspection: skip certificate verification for git/HF downloads.
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=1
export HF_HUB_DISABLE_SSL_VERIFICATION=1

# Do not echo env or run `env | grep -i proxy` because that may expose credentials.
set -x
```

If the proxy variables are not already present, stop with `HUMAN_ACTION_REQUIRED: YES`; the user must export the corporate proxy in the CodeAgent shell. Do not ask the user to paste credentials into source files.

## Repository and paths

```bash
export REPO_URL="https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git"
export BRANCH="agent/nas-mobileclip2-s0-9label-zeroshot"
export PROJECT_ROOT="/data/pub1/z00919662/segmentation/nas-mobileclip2-s0-9label-zeroshot"
export BENCH_ROOT="/data/pub1/z00919662/segmentation/nas_scene_tagging/mobileclip2_s0_probe"
export RESULT_ROOT="/data/pub1/z00919662/segmentation/nas_scene_tagging/mobileclip2_s0_results"
export HF_HOME="/data/pub1/z00919662/.cache/huggingface"
```

Create cache/output directories if needed:

```bash
mkdir -p "${HF_HOME}" "$(dirname "${BENCH_ROOT}")"
```

## 1. Pull the prepared GitHub code

If the project directory does not exist:

```bash
git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${PROJECT_ROOT}"
```

If it already exists:

```bash
cd "${PROJECT_ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: repository has local modifications; do not overwrite them"
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

test -f nas_multilabel/labels.py
test -f nas_multilabel/build_probe_dataset.py
test -f nas_multilabel/run_mobileclip2_zeroshot.py
test -f nas_multilabel/requirements.txt
```

Expected branch:

```text
agent/nas-mobileclip2-s0-9label-zeroshot
```

## 2. Reuse an existing CUDA PyTorch environment without damaging it

First inspect the current environment:

```bash
which python || true
python - <<'PY' || true
try:
    import torch
    print("torch", torch.__version__)
    print("cuda", torch.cuda.is_available())
    print("gpu_count", torch.cuda.device_count())
except Exception as e:
    print("TORCH_CHECK_FAILED", repr(e))
PY

command -v conda || true
conda env list || true
nvidia-smi
```

Prefer the same working Python/Conda base environment previously used for the RVM project if it already imports CUDA PyTorch successfully.

Do NOT uninstall or upgrade the base environment's torch/torchvision.

Create an isolated venv that inherits the working torch installation:

```bash
export VENV_DIR="${PROJECT_ROOT}/.venv_mobileclip2_s0"

BASE_PYTHON="$(command -v python)"
"${BASE_PYTHON}" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__)'

if [[ ! -d "${VENV_DIR}" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org

python -m pip install -r "${PROJECT_ROOT}/nas_multilabel/requirements.txt" \
  --upgrade-strategy only-if-needed \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org
```

After installation verify that CUDA torch still works:

```bash
python - <<'PY'
import torch
import open_clip
import mobileclip
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("open_clip", getattr(open_clip, "__version__", "unknown"))
models = open_clip.list_models()
print("MobileCLIP2-S0_available", "MobileCLIP2-S0" in models)
assert torch.cuda.is_available()
assert "MobileCLIP2-S0" in models
PY
```

If the current Python is not a working CUDA torch environment, inspect existing Conda environments and activate one that is already known to run RVM. Do not create an arbitrary new CUDA stack unless there is no usable existing environment. If none exists, stop and report.

## 3. Syntax check prepared code

```bash
cd "${PROJECT_ROOT}/nas_multilabel"
python -m compileall -q .
```

Do not modify code if this fails. Report the exact error.

## 4. Locate existing datasets

### 4.1 Existing 13-class semantic dataset

Prefer the already prepared 640x360 static dataset:

```bash
export SEG_ROOT="/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360"

test -d "${SEG_ROOT}/images/val"
test -d "${SEG_ROOT}/annotations/val"
```

This source supplies reliable `sky` and `building` GT from segmentation masks. It may also supplement `food`.

If the exact path is absent, search only for an equivalent existing 13-class prepared dataset:

```bash
find /data/pub1/z00919662/dataset /data/pub1/z00919662/segmentation \
  -maxdepth 5 -type f -name PREPARED_16X9_MANIFEST.json -print 2>/dev/null
```

Use the corresponding dataset root only if it has `images/val` and `annotations/val` and the same 13-class mapping used by this project.

### 4.2 COCO

Find an existing COCO instances annotation file:

```bash
find /data/pub1/z00919662/dataset /data/pub1/z00919662/segmentation \
  -maxdepth 7 -type f \( -name instances_val2017.json -o -name instances_train2017.json \) \
  -print 2>/dev/null | head -20
```

Prefer `instances_val2017.json` and its matching `val2017` images. Set `COCO_ROOT` to the dataset root containing `annotations/` and `val2017/` or `images/val2017/`.

This source supplies reliable `food` and `animal` GT. Sports-object positives are supplemental only.

### 4.3 Places365

Search for the Places365 dataset that was previously downloaded under the user's dataset area:

```bash
find /data/pub1/z00919662/dataset /data/pub1/z00919662/segmentation/datasets \
  -maxdepth 5 -type d \( -iname '*places365*' -o -iname 'data_large' \) \
  -print 2>/dev/null | head -50
```

Also inspect for known class directories without enumerating image files globally:

```bash
find /data/pub1/z00919662/dataset \
  -maxdepth 6 -type d \( -name office -o -name office_cubicles -o -name mountain -o -name beach -o -name basketball_court \) \
  -print 2>/dev/null | head -50
```

Set `PLACES_ROOT` to the root above the Places365 class hierarchy. The prepared builder understands common layouts such as `data_large/a/airport_terminal/...`.

Places365 supplies `indoor`, `outdoor`, `landscape`, `sports`, and `office` GT.

Do not download another large scene dataset merely to hide a missing local dataset. If the existing Places365 data cannot be found or does not contain enough categories, continue to the dataset builder once to obtain its coverage report, then stop and report the missing coverage.

## 5. Build the nine-label probe test set

Before running, print paths only (never proxy values):

```bash
printf 'PLACES_ROOT=%s\nCOCO_ROOT=%s\nSEG_ROOT=%s\n' \
  "${PLACES_ROOT}" "${COCO_ROOT}" "${SEG_ROOT}"
```

Build:

```bash
rm -rf "${BENCH_ROOT}"
mkdir -p "${BENCH_ROOT}"

cd "${PROJECT_ROOT}/nas_multilabel"
python -u build_probe_dataset.py \
  --places-root "${PLACES_ROOT}" \
  --coco-root "${COCO_ROOT}" \
  --seg-root "${SEG_ROOT}" \
  --output-dir "${BENCH_ROOT}" \
  --n-pos 150 \
  --n-neg 150 \
  --min-pos 50 \
  --min-neg 50 \
  --seed 20260828 \
  --materialize symlink \
  2>&1 | tee "${BENCH_ROOT}/build.log"
```

The builder MUST pass coverage for all nine labels. Check:

```bash
cat "${BENCH_ROOT}/dataset_summary.json"
wc -l "${BENCH_ROOT}/manifest.jsonl"
```

Required conditions:

```text
indoor:   positive >= 50, negative >= 50
outdoor:  positive >= 50, negative >= 50
landscape:positive >= 50, negative >= 50
sports:   positive >= 50, negative >= 50
food:     positive >= 50, negative >= 50
animal:   positive >= 50, negative >= 50
building: positive >= 50, negative >= 50
sky:      positive >= 50, negative >= 50
office:   positive >= 50, negative >= 50
```

Important: labels not known for a source remain `-1` and are excluded from metrics. This is intentional.

If coverage fails, do not alter `labels.py` or lower the minimum. Report `candidate_summary.json` and stop.

## 6. Select one free GPU

```bash
nvidia-smi
```

Choose one genuinely free GPU. For example, if GPU 0 is free:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Do not kill or interfere with existing training/inference jobs.

## 7. Run MobileCLIP2-S0 9-label zero-shot

The official checkpoint is `MobileCLIP2-S0`, pretrained `dfndr2b`. Allow Hugging Face/OpenCLIP to cache it under `${HF_HOME}`.

```bash
rm -rf "${RESULT_ROOT}"
mkdir -p "${RESULT_ROOT}"

cd "${PROJECT_ROOT}/nas_multilabel"
python -u run_mobileclip2_zeroshot.py \
  --manifest "${BENCH_ROOT}/manifest.jsonl" \
  --output-dir "${RESULT_ROOT}" \
  --device cuda \
  --batch-size 32 \
  --workers 4 \
  --pretrained dfndr2b \
  --threshold 0.5 \
  --amp fp16 \
  --benchmark-warmup 50 \
  --benchmark-runs 200 \
  --visualize-errors 16 \
  2>&1 | tee "${RESULT_ROOT}/run.log"
```

If batch size 32 causes CUDA OOM, reducing `--batch-size` to 16, 8, 4, or 1 is allowed because it is a runtime parameter and does not modify model/code/accuracy logic. Record the final batch size.

Do not change prompts or the 0.5 primary threshold.

## 8. Verify outputs

```bash
test -f "${RESULT_ROOT}/REPORT.md"
test -f "${RESULT_ROOT}/summary.json"
test -f "${RESULT_ROOT}/per_class_metrics.csv"
test -f "${RESULT_ROOT}/predictions.csv"
test -f "${RESULT_ROOT}/prompts.json"
test -f "${RESULT_ROOT}/oracle_threshold_diagnostic.csv"
test -f "${RESULT_ROOT}/latency.json"

cat "${RESULT_ROOT}/REPORT.md"
```

List visualizations:

```bash
find "${RESULT_ROOT}/error_visualizations" -maxdepth 1 -type f -print | sort
```

## 9. Interpretation rules

Primary zero-shot result = fixed threshold 0.5 in `per_class_metrics.csv` / `summary.json`.

Report at minimum for each of the nine labels:

- Precision
- Recall
- F1
- Accuracy
- Average Precision
- number of known positives/negatives

Also report:

- macro-F1
- macro-AP
- micro-F1 over known label/image pairs

`oracle_threshold_diagnostic.csv` is diagnosis only. It is useful to answer:

- If oracle F1 is much better than fixed-0.5 F1: representation may be usable but zero-shot calibration/prompt threshold is poor.
- If oracle F1 is also poor: MobileCLIP2-S0 feature separation itself is insufficient for this label on these datasets.

Never call oracle-threshold numbers "zero-shot accuracy".

The latency benchmark is server GPU image-encoder latency only. It is NOT V516 latency and MUST NOT be used to claim 100 fps on V516.

## 10. Final report format

Return exactly enough detail to make the next decision:

```text
STATUS: PASS / FAIL

GITHUB_BRANCH:
GITHUB_COMMIT:

PYTHON:
TORCH_VERSION:
OPEN_CLIP_VERSION:
GPU:
CUDA_VISIBLE_DEVICES:

PLACES_ROOT:
COCO_ROOT:
SEG_ROOT:

BENCH_ROOT:
UNIQUE_TEST_IMAGES:

KNOWN_GT_COUNTS:
indoor: pos / neg
outdoor: pos / neg
landscape: pos / neg
sports: pos / neg
food: pos / neg
animal: pos / neg
building: pos / neg
sky: pos / neg
office: pos / neg

PRIMARY_ZERO_SHOT_THRESHOLD: 0.5

PER_CLASS_METRICS:
label | precision | recall | F1 | accuracy | AP

MACRO_F1:
MACRO_AP:
MICRO_F1_KNOWN_PAIRS:

ORACLE_DIAGNOSTIC:
label | oracle_threshold | oracle_best_F1

GPU_ENCODER_LATENCY_FP16_BATCH1:
mean_ms:
p95_ms:
encoder_only_fps:

REPORT_PATH:
PREDICTIONS_PATH:
ERROR_VIS_DIR:

WARNINGS:
HUMAN_ACTION_REQUIRED: YES / NO
```

If successful:

```text
HUMAN_ACTION_REQUIRED: NO
```

If an existing dataset needed for nine-label coverage is genuinely absent, or the pretrained checkpoint cannot be downloaded because proxy authentication is missing:

```text
HUMAN_ACTION_REQUIRED: YES
```

State exactly which dataset/path or environment action is needed. Do not modify code to work around it.
