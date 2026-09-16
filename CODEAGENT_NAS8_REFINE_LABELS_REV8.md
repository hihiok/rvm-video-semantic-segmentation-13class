# CodeAgent：NAS8 Rev8 标签配平与 GT100（只准备，不训练）

本轮执行八类任务，不执行历史九类训练文档。环境使用已有 ultraface_new。

## 1. 输入与保护规则

只读输入：
`/mnt/ssd1/z00919662/datasets/NAS8_multilabel_clean_v3_rev7_20260916_103040`

从该目录现有 train/val/test.jsonl 增量生成新版本，不重新跑 run_prepare.sh 或 prepare_all.py，不重新下载或解压。保留输入、原图、旧训练及 watchdog。不得本地 patch Python/config。遇到代码错误保留诊断并停止，交回 ChatGPT 修复 GitHub。

沿用现有 manifest 的绝对图片路径和 split，不重映射 AWB_10_scenes/10_scenes。逻辑 source=10_scenes 的 Night 行是用户确认的行车记录仪图，补 outdoor=1；其他类别不能由此推断。两张已确认 MIR 图 im14411.jpg、im1414.jpg 改 indoor=0、outdoor=1，其他标签不变。

## 2. 配平规则（必须遵守）

八类为 night、indoor、rain_snow、office、outdoor、landscape、sports、objective_image。

- 在 train、val、test 主清单内分别要求每类 negative <= positive。
- 所有含任意正标签的图片均保留，全部正标签不因配平被屏蔽。尤其雨雪、办公、自然风景、运动、客观图，逐图与逐类断言保护。
- 优先移除没有任何正标签的多余纯负样本；首先处理只有一个有效负标签的图片，包括其余七类均 -1、objective_image=0 的图片。
- 有其他正标签的图片不能删除。若仍有超额负例，仅把该类别的负监督 0 设为 -1，记录原值、依据及 sampling 原因。这表示本版不参与该类监督，不表示原始标注错误。
- 不把 -1 改 0，不把负例改正例凑数。某类正例为零时，不保留该类负监督，并报告质量缺口。
- 正例保护比较基准是明确人工纠错之后；两张 MIR 的 indoor 纠错单列报告，不与采样损失混淆。
- val/test 也按用户要求采样，结果不再代表自然类别分布。val_strict/test_strict 是可信证据子集，正负比可以不同，不能冒称这两个子集也满足 1:1。

## 3. 同步代码

Repository: https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
Branch: agent/nas8-mir-nus-clean-gt100-v3

使用现有仓库；如不存在，clone 到 `/mnt/ssd1/z00919662/NAS_scene_detection/nas8-mir-nus-clean-v3`。复用服务器已有网络/代理设置，不把凭据写入仓库或报告。核对 origin URL；工作树若不干净，保留并报告差异，不覆盖或 reset。

```bash
git status --short
git fetch origin agent/nas8-mir-nus-clean-gt100-v3
git checkout agent/nas8-mir-nus-clean-gt100-v3
git merge --ff-only origin/agent/nas8-mir-nus-clean-gt100-v3
git rev-parse HEAD
```

要求存在 `nas_data/scene8_v3/refine_labels.py` 和 `test_refine_labels.py`。记录实际 commit。

## 4. 环境与测试

加载本机已有 conda 初始化文件，激活 ultraface_new，不新建或升级环境。

```bash
conda activate ultraface_new
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
python -c 'import os,sys,PIL; assert os.environ.get("CONDA_DEFAULT_ENV")=="ultraface_new"; print(sys.version); print(PIL.__version__)'
(cd nas_data/scene8_v3 && python -m unittest test_pipeline test_nus21 test_legacy_roots test_relocation test_server_env test_refine_labels)
```

本版应通过 59 项测试，包含完整合成数据 GT100 流程及稀缺正例保护。失败即停止。

## 5. 执行

在仓库根目录使用 bash 执行：

```bash
set -euo pipefail
umask 077
INPUT_ROOT=/mnt/ssd1/z00919662/datasets/NAS8_multilabel_clean_v3_rev7_20260916_103040
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT=/mnt/ssd1/z00919662/datasets/NAS8_multilabel_clean_v3_rev8_${RUN_STAMP}
LOG=/mnt/ssd1/z00919662/datasets/nas8_refine_rev8_${RUN_STAMP}.log
test -d "$INPUT_ROOT"
test ! -e "$OUTPUT_ROOT"
python -u nas_data/scene8_v3/refine_labels.py \
  --input-root "$INPUT_ROOT" --output-root "$OUTPUT_ROOT" \
  --seed 20260910 2>&1 | tee "$LOG"
```

输出必须是输入的全新同级目录；不预建输出目录、不覆盖失败目录。脚本验证输入完成状态、文件哈希、图片路径、split/group 唯一性并保证旧标签文件未变。继承上一轮解码和去重结果，不宣称重新全量验图。

## 6. 验收和产物

检查 train/val/test JSONL+CSV、val_strict/test_strict、class_balance.csv、balance_audit.json、label_corrections.jsonl、negative_masks.jsonl、excluded.jsonl、source_audit.json、resolution_audit.json、summary.json、PREPARED.json。

source_records/excluded 仅描述本轮输入和新排除项；历史排除追溯输入目录，不重新启用历史排除图。

GT100 必须100张：COCO15、Places36520、SEG13 15、MIR20、NUS20、逻辑10_scenes10。含 gt100/index.html、100张图、10张contact图及根目录gt100.csv。某来源配平后样本被移除时，可作为明确标记“未进新manifest”的审核样本展示，不能为了配额重新加入训练。

每类分别输出输入、人工纠错后、配平后正/负/unknown，以及 negative/positive。断言所有正例逐图保留、负例上限通过、split不变、旧文件哈希不变。分开统计整行移除和单类别负监督屏蔽。

雨雪缺少真实照片负例的问题仍需审核 weather_review_template.csv；不允许用合成图负例冒充真实照片负例。不要为了消除 blocker 自动填 reviewed=1。查看 GT100、核查分辨率统计后再决定训练，本轮不启动训练。

最终报告：
STATUS: PREPARED_REVIEW_REQUIRED（仅完成且全部校验通过时）
GITHUB_COMMIT / ENVIRONMENT / INPUT_ROOT / NEW_LABEL_ROOT / LOG
各split八类数量、比例、纯负移除数、负监督屏蔽数
POSITIVE_RECORDS_LOST_BY_SAMPLING: 0
SPLITS_CHANGED: NO
OLD_IMAGES_LABELS_CHANGED: NO
OLD_TRAINING_WATCHDOG_CHANGED: NO
NEW_TRAINING_STARTED: NO
HUMAN_ACTION_REQUIRED: YES（GT100及真实照片雨雪负例审核）

失败时报告 BLOCKED、完整异常及诊断路径，不写准备完成结论。
