# CodeAgent：NAS8 Rev10 权重，99% precision 目标与有限规则重新推理

## 1. 当前任务与固定输入

执行本文件，使用已有八类权重重新跑 validation 与 test，生成新决策和GT可视化。本次不训练，不改数据、不改模型结构或输入分辨率，不执行历史九类指令。用户已授权直接完成，无需等待逐图审核。

- 环境：已有 `ultraface_new`，Python 3.8.20、torch 2.4.0+cu118、Pillow 10.4.0。无需重装环境。
- 工作 root：`/mnt/ssd1/z00919662/NAS_scene_detection`
- 模型目录：`/mnt/ssd1/z00919662/NAS_scene_detection/runs/nas8_rev10_20260916_173205/train`
- 数据目录：`/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev10_20260916_152810`
- 使用 `best_ultraface_scene8.pth`；原320×180、RGB direct resize、(pixel−127)/128。
- 标签顺序：night, indoor, rain_snow, office, outdoor, landscape, sports, objective_image。

## 2. 本次最终规则（以此取代此前讨论草案）

**至少输出一个标签，indoor/outdoor 绝不同时输出。99%是validation empirical precision选择目标，不是统一的概率阈值，也不代表每张图99%正确。**

1. 在validation重新前向推理，分别为八类选择阈值。仅GT=0/1进入选择；GT=-1忽略。候选阈值不低于原阈值及0.5；从validation分数的完整并列组边界寻找 empirical precision≥0.99 时召回最高的阈值。召回相同时优先precision，再优先高阈值。
2. 每类至少需30个已知正GT、30个已知负GT，并且选中至少30个已知GT样本，才可通过支持量检查。该下限只是防止极少样本决定阈值，不构成统计置信度保证。
3. 目标达不到或支持量不足，报告具体状态，该类阈值设为1.000001表示不通过常规阈值输出；**不降回97%，不放宽阈值凑数**。该类仍可能由下述单图兜底输出，必须标明。
4. 每张图各类独立通过新阈值，允许多个合格标签；不统一要求0.99，不限制输出类别总数。
5. indoor/outdoor均通过时只保留原始概率较高者；精确相同时按固定标签顺序保留indoor。概率差≤0.05时额外标记歧义，仍保留胜出者。0.05仅为诊断标记，不改变决策、不根据test调整。
6. 若没有任何类通过阈值，输出原始概率最高的一类，精确并列按固定标签顺序；记录 `fallback_top1=true`、概率和未达到的阈值。该兜底**不满足99%选择目标**，相关误报另行统计。不在已有合格标签时强行补top1。
7. 不禁止indoor+night、indoor+rain_snow、sports+其他场景；indoor+landscape、objective_image+场景同时出现只标记待查看，不强制删除。office不自动补indoor，landscape不自动补outdoor。
8. 所有后处理仅改变输出决定；不修改raw sigmoid probability、GT、原thresholds.json、权重、ONNX或原测试产物。输出0表示未输出，不表示人工确认不存在。

程序先保存并哈希冻结新 `decision_policy.json`，然后才执行test前向推理，test不用于数值阈值选择。用户已经看过test并据此提出方案，所以本轮结果属于诊断对照，不能称为全新未触碰测试。validation主标签含弱规则，尤其rain_snow没有逐图确认负例；99%仅相对于这些已知validation标签。

阈值应根据目标指标在验证数据上选择，而不是视0.5或0.99为通用最优值：[scikit-learn阈值选择说明](https://scikit-learn.org/stable/modules/classification_threshold.html)。本文件的具体99%目标、最低支持量和冲突策略是本项目试验配置。

## 3. 代理、SSL与同步GitHub

clone/fetch前加载私有代理配置。公共GitHub文件不含凭据；不要将凭据打印到日志或提交仓库。

```bash
set +x
: "${http_proxy:?请先加载私有代理配置}"
export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$http_proxy"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=1
```

Repository：`https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git`
Branch：`agent/nas8-mir-nus-clean-gt100-v3`
建议使用已有 checkout：`/mnt/ssd1/z00919662/NAS_scene_detection/nas8-rev10-training`。

核对origin和working tree；干净时fetch该分支、checkout同名分支、merge --ff-only origin分支。没有checkout才clone。不要覆盖同名非仓库目录。完整commit锁定见私有交付MD，同步后运行ancestor校验并记录完整HEAD。禁止reset/强制覆盖本地改动；发现local patch则保存diff和相关文件，让用户上传ChatGPT同步，不自行修改Python/config。不要启动训练脚本。

## 4. 环境与回归检查

进入checkout根目录，加载已有conda初始化脚本：

```bash
conda activate ultraface_new
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c 'import os,torch,cv2,numpy,PIL; assert os.environ.get("CONDA_DEFAULT_ENV")=="ultraface_new"; print(torch.__version__,cv2.__version__,numpy.__version__,PIL.__version__)'
(cd ultraface_scene_multilabel8 && python -m unittest test_decision_policy test_visualize_test_rev10 -v)
```

12项测试须通过，含真实小型CPU前向推理、validation/test路径关联和GT校验、99%目标与分数阈值区别、无达标类别时兜底、室内/户外互斥与并列、其他共存标签保留、metrics使用最终决策、GT未知忽略、输入哈希不变及完整图片网页输出。

## 5. 执行全量validation和test推理

选择一张实际空闲GPU，查询 `nvidia-smi` 并核对GPU进程归属；不要中断其他任务。只推理、不反传，一张V100即可，无需双卡DDP。下面的 `INFER_GPU` 填实际空闲卡编号（不是让用户手动执行）。若无空闲GPU可直接CPU，设置 `INFER_DEVICE=cpu`、`INFER_WORKERS=0`，报告CPU执行。

```bash
set -euo pipefail
umask 077
# CodeAgent根据GPU检查设置INFER_GPU，例如确认GPU0空闲时：
INFER_GPU=0
export CUDA_VISIBLE_DEVICES="$INFER_GPU"
INFER_DEVICE=cuda:0
INFER_WORKERS=2
RUN_DIR=/mnt/ssd1/z00919662/NAS_scene_detection/runs/nas8_rev10_20260916_173205/train
DATA_ROOT=/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev10_20260916_152810
INFER_STAMP=$(date +%Y%m%d_%H%M%S)
INFER_ROOT="/mnt/ssd1/z00919662/NAS_scene_detection/runs/nas8_rev10_20260916_173205/precision99_rules_${INFER_STAMP}"
python -u ultraface_scene_multilabel8/infer_precision_rules.py \
  --run-dir "$RUN_DIR" --data-root "$DATA_ROOT" --output-dir "$INFER_ROOT" \
  --device "$INFER_DEVICE" --batch-size 128 --workers "$INFER_WORKERS" \
  --target-precision 0.99 --min-predictions 30 --conflict-margin 0.05 \
  --per-group 10 --seed 20260917 2>&1 | tee "${INFER_ROOT}.log"
```

不要预建INFER_ROOT。保持原训练产物不变，所有结果输出到新同级目录。脚本先核对config中的val/test manifest哈希、八类顺序、预测NPZ覆盖及逐图GT、checkpoint metadata与原阈值，发现错配停止。NPZ只用于输入对应和旧新分数差异检查，**本次使用权重重新执行全部val/test的FP32前向推理**。旧结果可能曾用AMP，因此前后数值略有差异，本轮各方案都使用同一批新分数。

若OOM，仅允许减小batch到64/32或workers到0，用全新时间戳目录重跑；保留失败目录日志，不改模型/标签。其余代码错误保存BLOCKED.json并回报ChatGPT。无需审核CSV，也不自动扩展训练任务。

## 6. 比较方案与可视化

同一批新test分数比较五组：

| 名称 | 含义 |
|---|---|
| original_calibrated | 训练时原阈值 |
| all_099 | 所有类别概率阈值0.99，允许无输出，仅对照 |
| top1_plus_099 | 最高分必选，其他≥0.99，仅对照，不作为新默认 |
| precision_only | validation选出的99%目标独立阈值，未加规则和兜底，仅对照 |
| precision_context | 新默认：独立阈值＋室内户外互斥＋空结果top1兜底 |

必须区分“probability阈值0.99”和“validation precision目标99%”。报告每类Precision/Recall/F1/FP/FN/AP、macro指标、平均输出标签数、空输出图、兜底数量、目标未达类及其支持量、冲突数量。`precision_context` 的空输出图和室内户外同时输出图必须均为0。分别报告互斥规则移除的已知FP、移除TP变FN，以及兜底新增的已知FP/TP。

AP用未改动的raw scores计算，在五种策略间应相同；不要称阈值后处理提升了AP。GT=-1不判对错；最终不输出的GT正类仍计FN，不能将歧义/被压制结果从指标里排除。main和strict指标分开，严格天气负例缺失继续标记不支持。

按新默认的最终决定，每类从正确(TP+TN)、错误(FP+FN)分别均匀随机抽10张，seed固定；未知GT不抽。每组无重复，不同类别可以出现同一张图。数量不足展示全部并报告缺口，不凑数。最多160条展示记录、16张contact图。

每张卡片显示原图、GT完整八类、最终预测、raw probability、新阈值、FP/FN、人工/弱证据、兜底或压制标记；HTML在同一张图上并列显示五种方案的预测标签及规则明细。非空输出不等于该图一定有可信分类。

## 7. 产物与交付

- `INFER_ROOT/index.html`：总览及对比表；`gallery/index.html`：分类可视化。
- `gallery/<label>/correct/`、`error/`：JPEG、网页、contact；`gallery_audit.json`：抽样池/选中记录/缺口。
- `decision_policy.json`、`threshold_selection.json`：冻结新阈值和规则、每类目标与支持量。
- `val_inference.npz`、`test_inference.npz`：完整路径、GT、新分数、五种决策。
- `predictions.jsonl`：逐图GT/证据/raw score/五种标签输出/兜底/冲突明细。
- `comparison.csv`、`val_*`/`test_*` 主清单与strict的per_class.csv/summary.json。
- `summary.json`：输入哈希、规则影响、兜底误报、完整统计。
- `INFER_ROOT.zip`：含完整网页、渲染图、指标；下载解压打开index.html即可浏览，不需要原数据目录。无权重、无原始数据集打包。

至少打开一张正确、一张错误以及存在时一张兜底/冲突卡片，核查最终预测与JSON一致。检查原GT、权重和原阈值的输入哈希未变。不要用等量抽样图集的准确率替代全test结果，不根据本轮test结果再次调阈值。

部署时在原ONNX logits之后执行sigmoid，并使用 `decision_policy.py` 的 `decide(scores, thresholds, conflict_margin)`，参数读取新decision_policy.json；只换thresholds还不能实现互斥和兜底。原ONNX不需要重导出。

完成后返回完整HEAD、环境、GPU、五组指标表、每类新阈值/验证precision/召回/支持量/目标是否达成、规则及兜底代价、可视化和ZIP绝对路径：

```text
STATUS: INFERENCE_COMPLETE
HUMAN_ACTION_REQUIRED: NO
NEW_TRAINING_STARTED: NO
OLD_IMAGES_LABELS_CHANGED: NO
ORIGINAL_WEIGHTS_THRESHOLDS_CHANGED: NO
```

仅出现实际代码、输入、环境阻塞时返回BLOCKED及原因，禁止本地patch。99%目标未达仅如实报告并采用已规定的兜底，不作为本任务停止条件。最终实际效果由本轮输出决定，不能预先宣称减少了多少错label。
