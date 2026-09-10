# NAS 八类新数据合并、旧 GT 修复、100 张 GT 可视化 — V3

Rev2 更新：已遇到 people_r1 / NUS 检索格式阻塞的服务器，先执行仓库根目录 CODEAGENT_NAS8_SOURCE_PREFLIGHT_REV2.md；不要直接重复全量准备。

本文件供新的 CodeAgent chat 执行，只做数据准备，不启动/停止训练，不改模型、阈值、watchdog。

## 1. 目标和范围

环境：使用服务器已有 `Ultraface`。代码仅需要 Python 标准库和 Pillow；不要新建环境，不升级 torch/torchvision，不用 GPU。

类别顺序：`night, indoor, rain_snow, office, outdoor, landscape, sports, objective_image`。

使用新准备好的 MIRFLICKR 与 NUS-WIDE 本地 ZIP；旧 COCO、Places365、COCO_ADE_13cls、10_scenes 用旧八类 JSONL 作为图片清单，**不信任旧的八维标签**，按新政策重新生成。只处理此前已纳入旧 manifest 的旧来源图片，不重新扫描/下载整套旧数据集。

源 ZIP、原图、原标签、mask、旧 manifest、checkpoint、正式训练与 watchdog 全部只读。不能删除、改名、移动或覆盖它们。允许将新 ZIP 解压一次到新的 cache，允许生成审核缩略图，不复制旧图片为第二份训练集。

## 2. GitHub、proxy、SSL

Repository：`https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git`

Branch：`agent/nas8-mir-nus-clean-gt100-v3`

Code：`nas_data/scene8_v3/`

新 checkout：`/data/pub1/z00919662/segmentation/nas8-mir-nus-clean-v3`。

GitHub 版不存凭据。先从现有私有 shell 的 http_proxy 或用户私有 proxy.md 载入企业代理；不要打印凭据、env 或 git config --list，不要把含凭据的 MD 提交 GitHub。

```bash
set +x
: "${http_proxy:?HUMAN_ACTION_REQUIRED: YES - load the private corporate proxy first}"
export https_proxy="${https_proxy:-$http_proxy}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$https_proxy"
# 用户指定的企业 HTTPS 检查环境，在 clone/fetch 前配置。
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=1
```

SSL bypass 降低证书身份验证保障，仅用于用户指定的企业代理环境。本次数据处理完全离线，**不下载任何数据集、图片、模型或标注**。

```bash
set -euo pipefail
umask 077
export REPO_URL="https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git"
export BRANCH="agent/nas8-mir-nus-clean-gt100-v3"
export PROJECT_ROOT="/data/pub1/z00919662/segmentation/nas8-mir-nus-clean-v3"
if [[ ! -e "$PROJECT_ROOT" ]]; then
  git clone --single-branch --branch "$BRANCH" "$REPO_URL" "$PROJECT_ROOT"
else
  test -d "$PROJECT_ROOT/.git"
  test -z "$(git -C "$PROJECT_ROOT" status --porcelain)" || {
    echo 'HUMAN_ACTION_REQUIRED: YES; local modifications; preserve and report'; exit 2;
  }
  git -C "$PROJECT_ROOT" fetch origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
  git -C "$PROJECT_ROOT" checkout "$BRANCH"
  git -C "$PROJECT_ROOT" merge --ff-only "origin/$BRANCH"
fi

test "$(git -C "$PROJECT_ROOT" branch --show-current)" = "$BRANCH"
test "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" = "$(git -C "$PROJECT_ROOT" rev-parse origin/$BRANCH)"
test -z "$(git -C "$PROJECT_ROOT" status --porcelain)"
git -C "$PROJECT_ROOT" rev-parse HEAD
```

本次记录具体 commit 后不要滚动更新。不在当前训练 checkout 切分支。禁止 stash、reset --hard、git clean、force push、CodeAgent 本地写/改 .py/.sh/.md/config。发现本地修改，停止并返回 diff --stat，让用户把修改源码或补丁同步给 ChatGPT。

## 3. 本地输入路径

```bash
export OLD_LABEL_ROOT="/data/pub1/z00919662/dataset/UltraFaceSlim_8scene_multilabel_manifests_640x360_v1"
export MIR_IMAGES="/data/pub1/z00919662/dataset/mirflickr25k.zip"
export MIR_ANN="/data/pub1/z00919662/dataset/mirflickr25k_annotations_v080.zip"
export NUS_ZIP="/data/pub1/z00919662/dataset/archive.zip"
export CACHE_ROOT="/data/pub1/z00919662/dataset/NAS8_new_sources_raw_v3"
export NEW_LABEL_ROOT="/data/pub1/z00919662/dataset/NAS8_multilabel_clean_v3_rev2"
```

旧来源只读根目录：

```text
/data/pub1/z00919662/segmentation/datasets/coco
/data/pub1/z00919662/segmentation/datasets/places365
/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360
/data/pub1/z00919662/dataset/10_scenes
```

用户只给了 MIR 图片 ZIP 的文件名。优先用上面的完整路径；若不存在，脚本也支持之前的 `/data/pub1/z00919662/segmentation/datasets/MIRFLICKR25K/downloads/mirflickr25k.zip`。都不存在时，仅在 `/data/pub1/z00919662/dataset` 下 maxdepth=3 找同名文件：

```bash
find /data/pub1/z00919662/dataset -maxdepth 3 -type f -name mirflickr25k.zip -print
```

找到唯一文件后，只改 `MIR_IMAGES` 运行变量；有多个文件不能猜，报告给用户。禁止重下载。

NUS 的 `archive.zip` 文件名不能证明内部布局。脚本先保存目录清单再解析，只支持可核验的官方 Groundtruth + ImageList：

- `TrainImagelist.txt / TestImagelist.txt` + `Labels_<concept>_Train.txt / ..._Test.txt`；
- 或 `Imagelist.txt` + `Labels_<concept>.txt`；
- 必须有 `Concepts81.txt` 及原始图片；可以有两层嵌套 ZIP。

逐行对应 night/sports/snow 的人工 GT，保持原始行号。**绝不能 sort 图片目录后与标签配对，也不能拿 AllTags81/用户 tags 代替 Groundtruth**。只有 features、未知 .mat/.csv、缺图片或清单时，返回 `archive_inventory/` 和 `BLOCKED.json`，等待 ChatGPT 增加与实际格式匹配的解析器，不允许自行造一套对应关系。

若已有正确解压的完整 NUS 数据，可用 `--nus-root 实际路径` 替代 ZIP 解压，源目录仍只读。不要为了节省空间删除用户的 ZIP。

## 4. 激活环境与本地测试

初始化服务器实际存在的 conda.sh，再：

```bash
conda activate ultraface
test "$CONDA_DEFAULT_ENV" = ultraface
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -c 'import sys,PIL; print(sys.version);print("Pillow",PIL.__version__)'
cd "$PROJECT_ROOT/nas_data/scene8_v3"
python test_pipeline.py
bash -n run_prepare.sh
```

28 项离线测试必须通过，网络请求/torch/GPU 不参与测试。不要把测试合成图当成真实 GT100 结果。

检查源 ZIP 和旧三份 JSONL 存在：

```bash
ls -lh "$MIR_ANN" "$NUS_ZIP"
for split in train val test; do test -s "$OLD_LABEL_ROOT/$split.jsonl"; done
df -h /data/pub1/z00919662/dataset
```

脚本按 ZIP 实际体积检查解压空间；不要对 archive.zip 猜固定大小。空间不足报告所需空间，不能擅自删旧数据。

## 5. 执行数据准备

默认输出目录必须是新目录。存在时不删除、不覆盖，改为新的带时间戳输出路径，后续汇报实际路径；cache 可以安全复用已有完整解压结果。

```bash
cd "$PROJECT_ROOT/nas_data/scene8_v3"
LOG="/data/pub1/z00919662/dataset/nas8_clean_v3_$(date +%Y%m%d_%H%M%S).log"
bash run_prepare.sh \
  --old-root "$OLD_LABEL_ROOT" \
  --mir-images "$MIR_IMAGES" \
  --mir-annotations "$MIR_ANN" \
  --nus-archive "$NUS_ZIP" \
  --cache-root "$CACHE_ROOT" \
  --output-root "$NEW_LABEL_ROOT" \
  --seed 20260910 \
  2>&1 | tee "$LOG"
```

日志应持续出现 EXTRACT、parse、SOURCE、QUALITY、render 等进度。解压仅选择图片/标注/README 等支持文件，不解压无关 .mat/.dat 特征包。图片 MD5 严格核对官方 MIR ZIP；NUS 记录本地 SHA256（不是官方认证值）。只对被提取 ZIP 成员检查 CRC，不宣称已验证跳过的特征文件。

遇到错误时保存已有诊断，不删除半成品；按说明换新的 `--output-root`。半完成的 cache 解压目录要报告，不能擅自清理它。

## 6. 本版标签规则（禁止擅自修改）

所有记录固定 8 维，1=正，0=有依据的负，-1=unknown。每类有 evidence 字段，`manual:` 是原始人工标注，`weak:` 是类别/照片来源推断，`user_review:` 是用户明确指出的单图，`human_review:` 是后来批准的审核。

### 夜景

MIR night 与 NUS nighttime 提供真实照片正负监督；MIR potential/r1 单独处理。旧 Places/COCO/SEG 不再自动给 night 正负。10_scenes 夜景 folder 可提供弱正样本，其他 folder 不自动当夜景负样本。真实照片负样本数量单独统计，不把 Computer_synthesized 的负样本算进去。

### 雨/雪

任一确认存在 =>1；两者都确认不存在 =>0；其余 =>-1。

NUS snow=1 ->1，snow=0 ->unknown。MIR 不提供雨雪。

**撤销 Places 整类推雨雪，也撤销 SEG13 的 snow=0 -> rain_snow=0。** SEG13 的原始类名是 ice_or_snow，无法区分纯冰与雪，本版不把这个混合类别直接当可靠雪 GT。雨雪优先来自 NUS snow 正样本与已有明确 rain/snow 的 10_scenes folder。

真实照片雨雪负样本很可能不足：不能拿大量合成图填充，也不能用“晴天/室内/没有雪”武断当双重否定。生成 `weather_review_template.csv`，待人确认；这不阻塞可视化，但阻塞后续宣称数据已适合可靠雨雪训练/验收。

### 自然风景

仅自然景观为主要内容。Places 中保留明确自然类别；downtown/skyline/street 等城市主场景作为弱负候选。harbor/village/wind_farm 等边界类别不自动标自然风景。**废除 SEG13 natural_area>=30% -> landscape=1。** 旧 10_scenes 的宽泛 landscape folder 暂列审核，不自动沿用。

### 运动、办公、室内/户外

NUS 原生 sports 0/1 可用；sports=0 与 soccer/running 等子类正样本冲突时设unknown。Places 体育场景/活动可提供弱正；不再将 river/beach/mountain 等一律设运动负。

office/home_office/office_cubicles/conference_room 为办公弱正；computer_room/reception/conference_center 不直接等于办公，保留unknown。它们仍可补客观图的真实困难负样本。

室内/户外使用原生 MIR indoor 和仓库内原版 Places IO 表；飞机舱窗外、车站站台、半开放环境等已知歧义类别设unknown。不能用 MIR indoor=0 自动推出outdoor=1。

### 客观图

沿用用户定义的 Computer_synthesized/pattern/chart 正样本，但不假设任意 CGI 都是测试图。增加真实办公室、显示器环境、会议室等困难负样本。普通照片来源推断显式标为 weak，需要抽查。

只在新训练标签中按比例抽样客观图负监督，优先保留办公/屏幕困难样本，不用十几万张普通负样本淹没正类。没有其他有效标签的未选记录可不参加训练。

### 用户报告的五个例子

`reported_examples.json` 会追踪它们是否出现在旧清单、旧 GT、新 GT 与证据，不靠可视化输出文件名的数字前缀定位。

- `Places365_val_00027091.jpg`：撤销冲突类别推断，列人工审核；不凭描述猜八个完整标签。
- `ADE_train_00001854.jpg`：按用户“草地打球”描述记 sports=1、landscape=0。
- `000000465180.jpg`：按用户“大象戏水为主体”描述记 landscape=0。
- `000000032334.jpg`、`000000469246.jpg`：展示 GT 0/1/? 全表，不能把没有 GT+ 误称没有监督。

若源清单找不到某例，报告 NOT FOUND，不伪造图片或标签。

## 7. 划分、排除和旧模型隔离

新 ZIP 自定义 split 或从官方 Train 留出 validation。旧数据保留 train/eval 属性，重复组冲突时 eval 优先。以完全相同解码像素、绝对路径、仅限 COCO/SEG 来源的 COCO ID 建立组；新三个 split 的已知组交集为零。

只对完全相同像素的记录合并标签，矛盾设unknown；不能把经过裁剪的 COCO 派生图与原图的场景标签无条件互传。每组只选一个有效记录用于新 manifest，全来源记录另存供审计。

dHash+长宽比仅做相似图候选检查：可能跨 train/eval 的候选训练组保守排除，输出报告。不是彻底查尽所有近重复，也不是将 dHash 相同认定为同一幅图。旧标注/文件保持不动。

**旧 checkpoint 已经看过旧数据，不能用这次重新分组的数据宣称旧模型是独立测试。** 后续需训练新模型。当前任务不停止旧训练，也不启动新训练。

## 8. GT100、尺寸审计与产物

抽样固定 100 个不重复的已知底层组，配额为：COCO 15、Places365 20、COCO_ADE/SEG13 15、MIRFLICKR 20、NUS-WIDE 20、10_scenes 10。

除了用户指定五个来源，额外覆盖 10_scenes，否则看不到客观图等关键来源。优先覆盖各类正例和用户报告问题图，再随机补齐。抽样包含少量被排除/待审核图，画面明确标注是否进入新 manifest，不能将它们当训练数据。

可视化是 **GT，不是预测**，无 score、无 threshold。每张图显示原图（不裁剪）、source、路径、split、八类完整 GT 和各标签证据；文字按实测宽度换行，表格列宽固定、高度动态，避免先前文字覆盖/裁切。

```text
NEW_LABEL_ROOT/
  train.jsonl / val.jsonl / test.jsonl
  train.csv / val.csv / test.csv
  val_strict.jsonl / test_strict.jsonl       只计原生或人工审核标签的评估子协议
  source_records.jsonl                     包含各来源观察与原标签/旧标签/证据
  excluded.jsonl
  legacy_label_changes.jsonl
  source_audit.json
  reported_examples.json
  near_duplicate_review.json
  resolution_audit.json
  summary.json
  archive_inventory/
  gt100/                                   恰好100张独立JPEG及index.html
  gt100.csv
  gt100_audit.json
  contact_01.jpg ... contact_10.jpg
  review100_template.csv
  weather_review_template.csv
```

strict 指标也可能缺某些类的监督，必须报告缺失，不能用全 -1 的列报0错误/100%准确。weak rule 派生的评估数字不能当人工金标准验收。

尺寸基于最终选入新 TRAIN 的图片，按来源和各类正例输出 P10/P50/P90、候选输入放大比例。SEG13 的640x360是派生尺寸，不能据此强制全模型640x360。本任务只给首个小分辨率建议，不做精度对照、不宣称已确定最佳训练尺寸。

## 9. 用户人工操作与重新生成

数据准备成功状态是 `PREPARED_REVIEW_REQUIRED`：这是处理完成、等待人工GT审核，不是代码失败；当前代码不会启动训练。须让用户查看 `gt100/index.html` 和100张图。

若 `training_quality_blockers` 有真实雨雪负例不足等项，明确输出缺哪个类别/哪个split/数量，用户在**新复制的**审核 CSV 填0/1/-1、`reviewed=1`、`reviewer` 和说明。空白不改标签，未批准行不生效；不要让 CodeAgent 根据亮度/文件夹或自己的猜测填标签。

填写后可通过 `--overrides /新路径/人工审核.csv --output-root /新路径/NAS8_multilabel_clean_v3_reviewed` 重新生成。旧结果和原图仍保留。不要覆盖当前模板或旧源标签。

因为图像包已在服务器，本任务不安排下载；只有本地ZIP缺失、坏包或NUS缺原图/清单时，明确告知用户需要提供什么文件，不能直接去网上下载不明版本替代。

## 10. 结束报告和停机规则

任何代码/解析/校验错误，保存 BLOCKED.json 与 archive_inventory，返回 traceback、命令、环境、branch/commit；禁止本地 patch，代码必须由 ChatGPT 更新 GitHub 后同步。

数据审核正常结束后返回：

```text
STATUS: PREPARED_REVIEW_REQUIRED / BLOCKED
GITHUB_BRANCH / COMMIT:
ENVIRONMENT: Ultraface
MIR_IMAGES_ACTUAL_PATH:
NUS_FORMAT_AND_IMAGE_COUNT:
OLD_LABEL_ROOT:
NEW_LABEL_ROOT:
PER_SOURCE_COUNTS_BEFORE_AFTER:
TRAIN_VAL_TEST_EACH_LABEL_POS_NEG_UNKNOWN:
REAL_PHOTO_NIGHT_NEGATIVE_COUNTS:
REAL_PHOTO_RAIN_SNOW_NEGATIVE_COUNTS:
EXCLUSIONS_BY_REASON:
KNOWN_GROUP_TRAIN_VAL_TEST_OVERLAP:
USER_REPORTED_5_EXAMPLES:
GT100_SOURCE_COUNTS:
GT100_DIR / HTML / CSV / CONTACT_SHEETS:
IMAGE_RESOLUTION_SUMMARY_AND_SUGGESTION:
TRAINING_QUALITY_BLOCKERS:
OLD_IMAGES_LABELS_CHANGED: NO
OLD_TRAINING_WATCHDOG_CHANGED: NO
NEW_TRAINING_STARTED: NO
HUMAN_ACTION_REQUIRED: YES (GT审核或缺失输入时，具体说明动作)
```

不能因为命令返回0就宣称“八类数据已完全可靠”；不能因为没有八类全标注而擅自补0。CodeAgent 必须完成解压、标签、清洗、审计、GT100和报告；失败时明确停在哪一步，不只输出“将继续等待”。
