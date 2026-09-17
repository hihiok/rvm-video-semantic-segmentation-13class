# CodeAgent：八类固定confidence=0.99，至少一类输出，室内户外硬互斥

## 任务

使用已有NAS8 Rev10最佳权重重新执行全部test前向推理，生成GT和预测可视化。本次直接固定八类sigmoid confidence阈值0.99，**不选择validation阈值，不运行99% precision目标校准，不重新训练**。

最终决策顺序：
1. 八类各自 probability≥0.99 才进入候选。
2. indoor/outdoor同时进入候选，只保留概率较高者；完全相同保留indoor。分差≤0.05只标记歧义，不更改上述决定。
3. 没有候选时选择原始概率最高的一类兜底；并列按固定类别顺序。记录fallback_top1及其概率。**每张图至少一类，室内户外绝不同时输出。**
4. 不添加其他硬互斥，不改GT或raw probability。0.99是confidence阈值，不保证99%准确率。低于0.99的兜底单独标记并统计。

固定环境：`ultraface_new`，现有Python3.8.20、torch2.4.0+cu118、Pillow10.4.0。不重新安装环境。

- 工作root：`/mnt/ssd1/z00919662/NAS_scene_detection`
- checkout建议：`/mnt/ssd1/z00919662/NAS_scene_detection/nas8-rev10-training`
- 模型：`/mnt/ssd1/z00919662/NAS_scene_detection/runs/nas8_rev10_20260916_173205/train/best_ultraface_scene8.pth`
- 数据：`/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev10_20260916_152810`
- 顺序：night, indoor, rain_snow, office, outdoor, landscape, sports, objective_image。
- 原网络及预处理保持一致：320×180、RGB direct resize、(pixel−127)/128；本次FP32推理。

## 同步GitHub

clone/fetch之前配置代理和SSL。私有MD含实际代理配置；公共GitHub不保存凭据，禁止输出到日志或提交仓库。

```bash
set +x
: "${http_proxy:?请加载私有代理配置}"
export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$http_proxy"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=1
```

Repository：`https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git`
Branch：`agent/nas8-mir-nus-clean-gt100-v3`

已有checkout先核对origin及工作树；干净时fetch该分支，checkout同名分支，merge --ff-only对应origin。无checkout才clone，不覆盖同名目录。最低commit及ancestor校验见私有MD，记录实际完整HEAD。

禁止CodeAgent本地修改Python/config。发现本地改动须保留diff及文件，让用户上传ChatGPT同步；不reset/覆盖。不要执行旧九类训练文档，也不要执行上一版precision99校准脚本。

## 测试和执行

激活已有conda，在checkout根目录执行：

```bash
conda activate ultraface_new
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c 'import os,torch,cv2,numpy,PIL; assert os.environ.get("CONDA_DEFAULT_ENV")=="ultraface_new"; print(torch.__version__,cv2.__version__,numpy.__version__,PIL.__version__)'
(cd ultraface_scene_multilabel8 && python -m unittest test_fixed099 test_decision_policy test_visualize_test_rev10 -v)
```

14项测试必须通过，包含固定0.99边界、兜底、室内户外互斥、没有val文件时仍可完成全流程、确认没有调用阈值选择、真实小型CPU前向推理、GT关联、未知GT和可视化回归。

选择一张经nvidia-smi核实空闲的GPU，无需双卡、不干预其他进程。下方GPU0仅是确认空闲后的示例，CodeAgent填实际空闲编号。没有空闲GPU可用CPU：INFER_DEVICE=cpu，INFER_WORKERS=0，CUDA_VISIBLE_DEVICES为空。无需额外问用户。

```bash
set -euo pipefail
umask 077
INFER_GPU=0
export CUDA_VISIBLE_DEVICES="$INFER_GPU"
INFER_DEVICE=cuda:0
INFER_WORKERS=2
RUN_DIR=/mnt/ssd1/z00919662/NAS_scene_detection/runs/nas8_rev10_20260916_173205/train
DATA_ROOT=/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev10_20260916_152810
INFER_STAMP=$(date +%Y%m%d_%H%M%S)
INFER_ROOT="/mnt/ssd1/z00919662/NAS_scene_detection/runs/nas8_rev10_20260916_173205/fixed099_${INFER_STAMP}"
python -u ultraface_scene_multilabel8/infer_fixed099.py \
  --run-dir "$RUN_DIR" --data-root "$DATA_ROOT" --output-dir "$INFER_ROOT" \
  --device "$INFER_DEVICE" --batch-size 128 --workers "$INFER_WORKERS" \
  --per-group 10 --seed 20260917 2>&1 | tee "${INFER_ROOT}.log"
```

不要预建INFER_ROOT。所有产物使用新目录，不覆盖原模型、GT、thresholds.json或之前推理结果。OOM仅减batch到64/32或workers到0，用新时间戳目录重跑，保留原日志；其他代码问题报告ChatGPT，不本地patch。

脚本使用test manifest与旧test_predictions.npz核对逐图对应及原GT，并检查训练config哈希、checkpoint标签顺序和输入尺寸。本次分数来自**重新执行权重前向推理**，旧分数只用于数值差异诊断。不依赖val文件、不调阈值。

## 对照和可视化

在同一批新test分数上输出三组：

- `original_calibrated`：原校准阈值，仅对照。
- `fixed_099_only`：纯0.99阈值，允许空输出和室内户外共存，仅对照。
- **`fixed_099_context`：最终方案，0.99＋兜底＋室内户外互斥。**

返回各类Precision/Recall/F1/AP/FP/FN、macro指标、平均输出标签数、兜底数及新增FP/TP、互斥压制FP和误删TP变FN。最终方案 `empty_prediction_images=0`、`indoor_outdoor_overlap=0` 必须满足。

GT=-1不判对错，不改为0。规则抑制了GT正类仍计FN；兜底误报仍计FP。主清单含弱标签，单列strict指标，雨雪无可信负例仍标不支持。AP基于不变raw score，各策略应相同。此test已经用于诊断，不能称为未触碰的最终泛化评估。

按最终方案每类正确(TP+TN)随机10张、错误(FP+FN)随机10张，固定seed，每组不重复。某池不足10则全取并报告缺口，不凑数。不同类别可以重复展示同一图片，最多160条展示、16张contact图。完整八类GT、预测、概率、固定0.99阈值、弱证据、兜底/压制标记都须显示；HTML同图列三种方案预测。

## 结果交付

- `INFER_ROOT/index.html` 总览；`gallery/index.html` 每类GT/预测网页。
- `gallery/<class>/correct/`、`error/`：JPEG、contact.jpg、网页；`gallery_audit.json`抽样明细。
- `decision_policy.json`：八类阈值全为0.99，固定模式，规则明确。
- `test_inference.npz`、`predictions.jsonl`：路径、GT、raw scores、三个方案、规则记录。
- `comparison.csv`、三方案主/strict的per_class.csv和summary.json。
- `summary.json`：输入哈希、完整统计、兜底和互斥代价。
- `INFER_ROOT.zip`：完整离线网页和渲染图，可下载解压查看，无权重或完整数据集。

打开实际正确和错误卡片各一张，若存在再检查兜底或冲突卡片。核对GT、预测、路径及0.99阈值一致。保留原数据、权重、原阈值哈希不变。无需人工CSV审核，不开始训练。

报告完整commit、环境/GPU、三组指标和每类抽样数量、目录/ZIP/日志绝对路径：

```text
STATUS: INFERENCE_COMPLETE
HUMAN_ACTION_REQUIRED: NO
NEW_TRAINING_STARTED: NO
OLD_IMAGES_LABELS_CHANGED: NO
ORIGINAL_WEIGHTS_THRESHOLDS_CHANGED: NO
```

只有真实阻塞才返回BLOCKED及具体原因。用户只需把本MD交给CodeAgent，执行后查看新可视化。
