# CodeAgent任务：RVM13严格空间旁路 + RVM式时域Loss V1

## 1. 任务目标

在全新目录下载指定GitHub分支，完成测试、CUDA smoke，并在资源允许时启动正式训练。

本实验实现：冻结完整Stage-1单帧网络；每帧独立运行冻结路径；新增低分辨率、零初始化的ConvGRU temporal residual adapter；第一帧或scene-cut状态清空后的首帧严格输出Stage-1结果；使用CE、Dice、13类Laplacian、stable-region causal KL和RVM式概率变化量loss；不使用class weights或蒸馏。

## 2. 绝对限制和失败规则

```text
禁止修改任何源代码或配置文件。
禁止让CodeAgent自行修复代码。
禁止commit或push。
禁止覆盖、删除旧训练目录、日志或checkpoint。
禁止停止、kill或影响正在运行的旧训练。
禁止从旧视频模型恢复本实验。
禁止启用class weights或蒸馏。
禁止在旧项目目录内切换分支。
```

任何测试、smoke或训练失败时立即停止当前阶段，保存完整命令、日志和traceback，不得修改代码重试，并报告：

```text
HUMAN_ACTION_REQUIRED: YES
```

如果认为代码必须修改，只报告文件、行号、原因和完整错误，让用户同步回ChatGPT。

## 3. 网络代理和SSL

代理密码禁止写入Git仓库、报告或日志。使用服务器安全注入的`CORPORATE_PROXY_URL`；如果变量未配置，停止并由用户在当前shell私下设置，不得向终端回显其值。

```bash
test -n "${CORPORATE_PROXY_URL:-}" || {
  echo "CORPORATE_PROXY_URL is not configured"
  exit 2
}
export http_proxy="${CORPORATE_PROXY_URL}"
export https_proxy="${CORPORATE_PROXY_URL}"
export HTTP_PROXY="${CORPORATE_PROXY_URL}"
export HTTPS_PROXY="${CORPORATE_PROXY_URL}"
git config --global http.proxy "${CORPORATE_PROXY_URL}"
git config --global https.proxy "${CORPORATE_PROXY_URL}"
git config --global http.sslVerify false
```

## 4. GitHub版本与独立目录

```bash
export REPOSITORY="https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git"
export BRANCH="agent/rvm13-rvm-loss-temporal-residual-v1"
export PROJECT_ROOT="/data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class-rvm-residual-v1"
```

不得使用或切换正在训练的旧目录：

```text
/data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class-temporal-preserve-guided
```

目录不存在时：

```bash
git clone --branch "${BRANCH}" --single-branch "${REPOSITORY}" "${PROJECT_ROOT}"
```

目录已经存在时，先确认工作区干净，否则停止：

```bash
test -z "$(git -C "${PROJECT_ROOT}" status --short)"
git -C "${PROJECT_ROOT}" fetch origin "${BRANCH}"
git -C "${PROJECT_ROOT}" checkout "${BRANCH}"
git -C "${PROJECT_ROOT}" merge --ff-only "origin/${BRANCH}"
```

验证：

```bash
cd "${PROJECT_ROOT}"
test "$(git branch --show-current)" = "${BRANCH}"
test -z "$(git status --short)"
git rev-parse HEAD
test -f CODEAGENT_RVM13_RVM_LOSS_RESIDUAL_V1.md
test -f scripts/train_vspw_rvm_residual.sh
test -f tools/verify_temporal_residual_checkpoint.py
rg -n "TemporalResidualAdapter|forward_spatial|rvm_temporal_derivative_loss|multiclass_laplacian_loss" model/segmentation.py semantic_utils.py
```

## 5. 固定数据、权重和输出

```bash
export VSPW_ROOT="/data/pub1/z00919662/segmentation/datasets/VSPW_13cls"
export STATIC_ROOT="/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360"
export PREPARED_STATIC_ROOT="${STATIC_ROOT}"
export INIT_CHECKPOINT="/data/pub1/z00919662/segmentation/RobustVideoMatting-semantic-13class-512/output/rvm_semantic_13class_512_2gpu/best_miou.pth"
export OLD_PROJECT="/data/pub1/z00919662/segmentation/rvm-video-semantic-segmentation-13class-temporal-preserve-guided"
export OLD_OUTPUT_DIR="${OLD_PROJECT}/output/rvm_vspw_temporal_preserve_13class_640x360"
export OUTPUT_DIR="${PROJECT_ROOT}/output/rvm_vspw_rvm_residual_v1_13class_640x360"
```

检查：

```bash
test -d "${VSPW_ROOT}/images/train"
test -d "${VSPW_ROOT}/annotations/train"
test -d "${VSPW_ROOT}/images/val"
test -d "${VSPW_ROOT}/annotations/val"
test -d "${STATIC_ROOT}/images/train"
test -d "${STATIC_ROOT}/annotations/train"
test -d "${STATIC_ROOT}/images/val"
test -d "${STATIC_ROOT}/annotations/val"
test -f "${STATIC_ROOT}/PREPARED_16X9_MANIFEST.json"
test -f "${INIT_CHECKPOINT}"
```

`INIT_CHECKPOINT`必须是Stage-1单帧13类checkpoint。禁止改成`${OLD_OUTPUT_DIR}`下的`last.pth`、`best_balanced.pth`或`best_video_miou.pth`。

## 6. 检查旧训练，绝不干扰

```bash
pgrep -af "train_vspw_mixed.py|train_vspw_temporal_preserve.sh" || true
nvidia-smi
```

如果旧实验仍在运行：不得停止或修改它；不得启动本实验正式训练；仅在存在一张完全空闲GPU时执行smoke，否则smoke也不运行；报告旧PID、工作目录和GPU占用；设置：

```text
FORMAL_START_STATUS: BLOCKED_BY_EXISTING_TRAINING
HUMAN_ACTION_REQUIRED: YES
```

人工操作只是等待旧训练结束后重新执行本指令，不需要修改代码。

## 7. Python环境

```bash
source /home/z00919662/anaconda3/etc/profile.d/conda.sh
conda activate ultraface
which python
python --version
python -c 'import torch,torchvision,cv2,numpy,PIL; print("torch:",torch.__version__); print("torchvision:",torchvision.__version__); print("CUDA:",torch.cuda.is_available()); print("GPU count:",torch.cuda.device_count())'
```

预期Python为`/home/z00919662/anaconda3/envs/ultraface/bin/python`，PyTorch为`2.4.1+cu121`且CUDA可用。若版本轻微差异但环境正确且CUDA可用，只记录；否则停止。

## 8. CPU占用限制

所有测试、smoke、正式训练和恢复前执行：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
export WORKERS=1
```

正式训练最多2张GPU，每个rank每个DataLoader最多1个worker。禁止自行增加workers。CPU load若持续超过机器逻辑核数的70%，停止本次新实验，但不得停止旧实验，并报告。

## 9. 回归测试

```bash
cd "${PROJECT_ROOT}"
python tests/test_vspw_tools_standalone.py -v
python -m pytest tests/test_vspw_mixed.py tests/test_label_mapping.py -q
python -m compileall -q dataset model tools tests semantic_utils.py train_video_semantic.py train_vspw_mixed.py inference_video_semantic.py test_video_semantic.py
bash -n scripts/train_vspw_mixed.sh
bash -n scripts/train_vspw_temporal_preserve.sh
bash -n scripts/train_vspw_rvm_residual.sh
bash -n scripts/benchmark_guided_upsample.sh
```

任何失败都停止，禁止修改代码。

## 10. CUDA Smoke Test

选择未被旧训练占用且满足`free memory >= 18000 MiB`、`utilization <= 10%`的一张GPU：

```bash
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits
export SMOKE_GPU_ID="0"
```

将`0`替换为实际空闲GPU。没有符合条件的GPU就停止smoke，不得抢占资源。

```bash
export SMOKE_OUTPUT_DIR="${PROJECT_ROOT}/output/rvm_vspw_rvm_residual_v1_smoke_640x360"
test ! -e "${SMOKE_OUTPUT_DIR}" || { echo "Smoke output exists; use a new explicit path"; exit 2; }
mkdir -p "${SMOKE_OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${SMOKE_GPU_ID}" python -u train_vspw_mixed.py \
  --data-root "${VSPW_ROOT}" \
  --static-root "${STATIC_ROOT}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --output-dir "${SMOKE_OUTPUT_DIR}" \
  --input-width 640 --input-height 360 --max-frame-gap 1 \
  --temporal-residual-adapter --temporal-hidden-channels 16 --temporal-adapter-scale 0.25 \
  --stage2-epochs 1 --stage3-epochs 1 \
  --stage2-clip-length 2 --stage3-clip-length 3 \
  --stage2-video-batches 1 --stage2-static-batches 0 \
  --stage3-video-batches 1 --stage3-static-batches 0 \
  --stage2-trainable-scope temporal_residual --stage3-trainable-scope temporal_residual \
  --stage2-laplacian-weight 0.05 --stage3-laplacian-weight 0.05 \
  --stage2-temporal-weight 0.02 --stage3-temporal-weight 0.05 \
  --stage2-rvm-temporal-weight 0.05 --stage3-rvm-temporal-weight 0.05 \
  --laplacian-levels 5 --rvm-temporal-beta 0.1 \
  --temporal-boundary-radius 2 --temporal-temperature 1.0 \
  --batch-size 1 --static-batch-size 2 --workers 0 \
  --max-train-clips 4 --max-val-clips 2 \
  --max-static-train-images 4 --max-static-val-images 2 \
  --learning-rate 1e-4 --static-retention-tolerance 0.0 \
  --prediction-flip-penalty 0.1 --save-every 1 --print-every 1 \
  2>&1 | tee "${SMOKE_OUTPUT_DIR}/smoke.log"
```

检查：

```bash
test -f "${SMOKE_OUTPUT_DIR}/baseline_stage1.pth"
test -f "${SMOKE_OUTPUT_DIR}/baseline_metrics.json"
test -f "${SMOKE_OUTPUT_DIR}/metrics.csv"
test -f "${SMOKE_OUTPUT_DIR}/last.pth"
rg -n "laplacian_weight|stable_temporal_weight|rvm_temporal_weight" "${SMOKE_OUTPUT_DIR}/smoke.log"
rg -n "video_laplacian_loss|video_stable_temporal_loss|video_rvm_temporal_loss|video_rvm_changed_pixels" "${SMOKE_OUTPUT_DIR}/metrics.csv"

CUDA_VISIBLE_DEVICES="${SMOKE_GPU_ID}" python tools/verify_temporal_residual_checkpoint.py \
  --checkpoint "${SMOKE_OUTPUT_DIR}/last.pth" \
  --stage1-checkpoint "${INIT_CHECKPOINT}" \
  --device cuda --require-trained-adapter \
  | tee "${SMOKE_OUTPUT_DIR}/integrity_check.json"
```

必须同时得到：`frozen_stage1_tensors_changed: 0`、`adapter_output_projection_nonzero: true`、`reset_frame_exact_spatial_bypass: true`。否则禁止正式训练。

## 11. 正式训练配置

```text
总计25 epochs
Stage 2: 10 epochs, T=5, Laplacian=0.05, stable KL=0.02, RVM derivative=0.05
Stage 3: 15 epochs, T=8, Laplacian=0.05, stable KL=0.05, RVM derivative=0.05
CE=1.0, Dice=1.0
trainable=temporal residual adapter only
class weights=disabled
distillation=disabled
static optimizer batches=0
static validation=enabled
input=640x360
workers=1, CPU thread limits=1
```

静态训练batch为0是设计要求：T=1严格旁路没有可训练梯度。CE、Dice和Laplacian都在有GT的视频帧上训练；静态集继续每epoch验证。

## 12. 正式启动

重新检查资源：

```bash
pgrep -af "train_vspw_mixed.py|train_vspw_temporal_preserve.sh" || true
nvidia-smi
uptime
test ! -e "${OUTPUT_DIR}/last.pth" || { echo "Formal checkpoint exists; do not overwrite"; exit 2; }
```

任何旧训练仍在运行时禁止启动本实验。最多使用2张满足至少18000 MiB空闲、利用率不超过25%的GPU。GPU 6、7都满足时可设置：

```bash
export CUDA_VISIBLE_DEVICES="6,7"
```

否则执行`unset CUDA_VISIBLE_DEVICES`，让脚本安全选择；若只有一张符合条件，可显式只设置那一张。

```bash
export INPUT_WIDTH=640
export INPUT_HEIGHT=360
export VIDEO_BATCH_SIZE=2
export STATIC_BATCH_SIZE=8
export WORKERS=1
export MAX_GPUS=2
export MIN_FREE_GPU_MIB=18000
export LEARNING_RATE=1e-4
export SAVE_EVERY=5
export STATIC_VALIDATION_WEIGHT=0.5
export STATIC_RETENTION_TOLERANCE=0.0
unset CLASS_WEIGHTS
unset RESUME

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"
nohup env PYTHONUNBUFFERED=1 bash scripts/train_vspw_rvm_residual.sh \
  > "${OUTPUT_DIR}/train.log" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "${TRAIN_PID}" > "${OUTPUT_DIR}/train.pid"
echo "TRAIN_PID=${TRAIN_PID}"
```

启动后：

```bash
sleep 10
ps -p "${TRAIN_PID}" -f
tail -n 200 "${OUTPUT_DIR}/train.log"
nvidia-smi
```

至少观察到20个training step：日志持续增长、loss有限、无OOM/NaN/worker异常、GPU正常、CPU无持续过载、只有adapter可训练，且未启用class weights或distillation。

## 13. 断点恢复

只有本实验意外中断后才能恢复；checkpoint只保存完整epoch，中间进度不可恢复：

```bash
export RESUME="${OUTPUT_DIR}/last.pth"
nohup env PYTHONUNBUFFERED=1 bash scripts/train_vspw_rvm_residual.sh \
  > "${OUTPUT_DIR}/train_resume.log" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "${TRAIN_PID}" > "${OUTPUT_DIR}/train_resume.pid"
```

禁止用旧实验checkpoint恢复。

## 14. 完成后验收

```bash
export CANDIDATE="${OUTPUT_DIR}/best_spatial_preserved.pth"
test -f "${CANDIDATE}"
python tools/verify_temporal_residual_checkpoint.py \
  --checkpoint "${CANDIDATE}" --stage1-checkpoint "${INIT_CHECKPOINT}" \
  --device cuda --require-trained-adapter \
  | tee "${OUTPUT_DIR}/best_spatial_preserved_integrity.json"
```

若没有`best_spatial_preserved.pth`，验收失败，不能用其他checkpoint冒充。验收目标：

```text
T=1/reset frame与Stage-1严格一致
冻结Stage-1参数和buffer零变化
Video mIoU > 本次实测Stage-1 video baseline
Static mIoU >= 本次实测Stage-1 static baseline
Prediction flip rate < 本次实测Stage-1 baseline
```

历史指标只作参考：Stage-1 Static mIoU 0.6413；旧视频模型Video mIoU 0.5747、Static mIoU 0.6116、prediction flip rate 0.023717。必须以本次同代码、同验证集重新测得的baseline为准。

## 15. Guided Upsample

只有生成合格候选后才执行：

```bash
for RADIUS in 1 2 4; do
  CHECKPOINT="${CANDIDATE}" OUTPUT_DIR="${OUTPUT_DIR}" \
  BENCHMARK_DIR="${OUTPUT_DIR}/guided_upsample_benchmark_r${RADIUS}" \
  GUIDED_RADIUS="${RADIUS}" BENCHMARK_MAX_SAMPLES=500 \
  BENCHMARK_BATCH_SIZE=1 BENCHMARK_WORKERS=1 BENCHMARK_DEVICE=cuda \
  bash scripts/benchmark_guided_upsample.sh
done
```

比较bilinear/guided的mIoU、Boundary F1、耗时和额外显存。业务输入若仍是raw YUV444 10-bit且fps/pixel format未确认，不得猜测参数，不执行MP4推理，并报告需要人工补充码流格式。

## 16. 最终报告

```text
STATUS:
GITHUB_REPOSITORY:
GITHUB_BRANCH:
GITHUB_COMMIT:
WORKTREE_CLEAN:
SOURCE_CODE_MODIFIED: NO

PYTHON_ENVIRONMENT:
PYTHON_EXECUTABLE:
TORCH_VERSION:
CUDA_AVAILABLE:
PROJECT_ROOT:
VSPW_ROOT:
STATIC_ROOT:
INIT_CHECKPOINT:
OLD_OUTPUT_UNTOUCHED:
OUTPUT_DIR:

EXISTING_TRAINING_FOUND:
EXISTING_TRAINING_PID:
FORMAL_START_STATUS:
REGRESSION_TESTS:
CUDA_SMOKE_STATUS:
FROZEN_STAGE1_TENSORS_CHANGED:
RESET_FRAME_EXACT_SPATIAL_BYPASS:
ADAPTER_OUTPUT_PROJECTION_NONZERO:

LOSS:
CLASS_WEIGHTS_ENABLED: NO
DISTILLATION_ENABLED: NO
TRAINABLE_SCOPE: temporal_residual
TRAINABLE_PARAMETERS:
TOTAL_PARAMETERS:
TRAINABLE_RATIO:

STAGE2_LAPLACIAN_WEIGHT: 0.05
STAGE2_STABLE_TEMPORAL_WEIGHT: 0.02
STAGE2_RVM_TEMPORAL_WEIGHT: 0.05
STAGE3_LAPLACIAN_WEIGHT: 0.05
STAGE3_STABLE_TEMPORAL_WEIGHT: 0.05
STAGE3_RVM_TEMPORAL_WEIGHT: 0.05
WORKERS: 1
CPU_THREAD_LIMIT: 1
CPU_STATUS:
CUDA_VISIBLE_DEVICES:
TRAIN_PID:
TRAIN_LOG:
CURRENT_PROGRESS:

BASELINE_VIDEO_MIOU:
BASELINE_STATIC_MIOU:
BASELINE_PREDICTION_FLIP_RATE:
BEST_SPATIAL_PRESERVED_CHECKPOINT:
BEST_EPOCH:
BEST_VIDEO_MIOU:
BEST_STATIC_MIOU:
BEST_PREDICTION_FLIP_RATE:
VIDEO_ACCURACY_IMPROVED:
STATIC_ACCURACY_PRESERVED:
TEMPORAL_STABILITY_IMPROVED:
EXPERIMENT_STATUS:
WARNINGS:
HUMAN_ACTION_REQUIRED:
```

若只启动成功但未完成：`STATUS: TRAINING_RUNNING`、`EXPERIMENT_STATUS: PENDING`。不能把训练启动写成目标达成。
