# MIRFLICKR-25K：下载、校验、原始人工标签解析与 NAS 标签准备

本文件供全新 CodeAgent chat 完整执行。本任务只准备 MIRFLICKR-25K 这个数据源，不训练，不修改旧训练或 watchdog。其他历史 runbook 的训练指令不适用于本任务。

## 1. 目标和固定约束

GitHub repository: `https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git`

Branch: `agent/mirflickr25k-nas8-data-v1`

代码目录：`nas_data/mirflickr25k/`。

环境必须复用 `Ultraface`，不要创建新 conda/venv，不升级 torch/torchvision。本代码只用 Python 标准库和 Pillow，不需要 GPU，不需要旧 FSD 源码。

只下载官方 MIRFLICKR-25K 图片 ZIP 和人工 annotation v080 ZIP。不要下载 1M、缩略图版本、SIFT/其他预计算特征，也不要从 Flickr 用户 tags 造 GT。

严禁修改已有 COCO、ADE、Places365、10_scenes 的图片、mask、label、目录。严禁修改旧 manifest、训练输出、模型、watchdog 或结束现有训练进程。不在旧训练 checkout 上切分支；使用下文的新 checkout。

## 2. Proxy 和 SSL

GitHub 版本不存储代理凭据。在私有 shell 中沿用已经配置的 `http_proxy`，或从用户的私有 `proxy.md` 获取，不能把含密码的 MD 提交 GitHub。不要打印 proxy.md 内容、完整环境、带凭据的 git config 或 curl -v。

若未设置且无法找到用户私有配置，停止并说明用户需要将代理导入当前 CodeAgent shell；不要猜密码。

在 clone/fetch 之前执行：

```bash
set +x
: "${http_proxy:?HUMAN_ACTION_REQUIRED: load corporate proxy into this private shell first}"
export https_proxy="${https_proxy:-$http_proxy}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$https_proxy"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=1
```

上面的 SSL bypass 以及下载命令中的 `--insecure` 仅为用户指定的公司 HTTPS 检查环境使用；它降低证书认证保障。图片仍必须通过官方 MD5 和 ZIP CRC 校验。若服务器已正确安装企业 CA，可不使用 bypass。本任务禁止把密码写入源码、日志或公共报告。

## 3. 固定路径与严格仓库同步

```bash
set -euo pipefail
umask 077
export REPO_URL="https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git"
export BRANCH="agent/mirflickr25k-nas8-data-v1"
export PROJECT_ROOT="/data/pub1/z00919662/segmentation/nas-mirflickr25k-data-v1"
export MIR_ROOT="/data/pub1/z00919662/segmentation/datasets/MIRFLICKR25K"
export LABEL_ROOT="$MIR_ROOT/derived/mir25k_nas8_v1"

if [[ ! -e "$PROJECT_ROOT" ]]; then
  git clone --single-branch --branch "$BRANCH" "$REPO_URL" "$PROJECT_ROOT"
else
  test -d "$PROJECT_ROOT/.git"
  test -z "$(git -C "$PROJECT_ROOT" status --porcelain)" || {
    echo 'HUMAN_ACTION_REQUIRED: YES; local changes detected; preserve and report'; exit 2;
  }
  git -C "$PROJECT_ROOT" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
  git -C "$PROJECT_ROOT" checkout "$BRANCH"
  git -C "$PROJECT_ROOT" merge --ff-only "origin/$BRANCH"
fi

test "$(git -C "$PROJECT_ROOT" branch --show-current)" = "$BRANCH"
test "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" = "$(git -C "$PROJECT_ROOT" rev-parse origin/$BRANCH)"
test -z "$(git -C "$PROJECT_ROOT" status --porcelain)"
git -C "$PROJECT_ROOT" rev-parse HEAD
```

记录本次 commit，后续执行期间不再拉取滚动更新。不允许 stash、reset --hard、git clean、force push 或本地修补。出现本地修改时保留文件，输出 git diff --stat，并让用户把修改过的源码/patch 发给 ChatGPT 同步，不能擅自覆盖。

## 4. 激活环境并预检

使用服务器已有 conda 初始化脚本，然后：

```bash
conda activate Ultraface
test "$CONDA_DEFAULT_ENV" = Ultraface
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -c 'import sys, PIL; print(sys.version); print("Pillow",PIL.__version__)'
command -v curl
mkdir -p "$MIR_ROOT/downloads"
df -h "$MIR_ROOT"
cd "$PROJECT_ROOT/nas_data/mirflickr25k"
python test_prepare.py
bash -n run_prepare.sh
```

11 项离线测试必须通过。若 conda 命令未初始化，可只读查看 `command -v conda` 和常见的 `$HOME/anaconda3/etc/profile.d/conda.sh`、`/data/pub1/z00919662/anaconda3/etc/profile.d/conda.sh`，使用实际存在的脚本；不要创建环境。Pillow 缺失则停止报告，不擅自改环境。

从零开始通常应预留至少 8 GiB 空间；报告实测空闲量。脚本还会根据 ZIP 解压体积检查空间。空间不足不可清理别人的数据或删除旧实验。

## 5. 下载与准备

官方入口：`https://press.liacs.nl/mirflickr/mirdownload.html`

图片：`https://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip`

人工标注：`https://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k_annotations_v080.zip`

图片官方 MD5：`A23D0A8564EE84CDA5622A6C2F947785`。

优先复用 `$MIR_ROOT/downloads/` 下同名 ZIP，不重复下载。公司代理模式运行：

```bash
cd "$PROJECT_ROOT/nas_data/mirflickr25k"
LOG="$MIR_ROOT/prepare_$(date +%Y%m%d_%H%M%S).log"
bash run_prepare.sh --dataset-root "$MIR_ROOT" --insecure 2>&1 | tee "$LOG"
```

脚本将断点下载到 `.part`，显示字节进度，下载失败保留断点。图片包严格检查官方 MD5；两个包均做 ZIP CRC、安全路径检查，并记录 SHA256。annotation 包没有本任务已核实的官方 SHA256，不能把本地 SHA256 宣称为官方认证。

图片和原始标注只解压一次到新的 `raw/`。不会覆盖已存在但未完成的解压目录或新标签目录。重复成功执行直接复用；失败后需要新输出目录时，可传 `--output-dir "$MIR_ROOT/derived/mir25k_nas8_v1_retry_时间戳"`，不得删除旧结果。

生产准备要求完整 `im1.jpg` 至 `im25000.jpg`，每图完整解码。遇到坏图、标注 ID 不合法、格式变化或缺 README，要报告并停止，不得跳过后称 PASS。

## 6. 人工下载规则

无法联网、代理认证失败、官方服务器拒绝或下载重试失败时，不得无限等待。保留 `.part`，输出 `PREPARE_BLOCKED.json`，明确：

```text
STATUS: BLOCKED / MANUAL_DOWNLOAD_REQUIRED
HUMAN_ACTION_REQUIRED: YES
缺哪个文件：
官方下载链接：
需要用户保存到：
失败原因：
```

用户人工下载后上传到：

```text
/data/pub1/z00919662/segmentation/datasets/MIRFLICKR25K/downloads/mirflickr25k.zip
/data/pub1/z00919662/segmentation/datasets/MIRFLICKR25K/downloads/mirflickr25k_annotations_v080.zip
```

无需用户解压。重新执行以下命令，只从现成 ZIP 处理，不访问网络：

```bash
bash run_prepare.sh --dataset-root "$MIR_ROOT" --offline
```

MD5 错误要明确报错并保留文件，不能以 `--skip-checksum` 或换不明镜像规避。这里只支持官方 25K 图片包与 v080 标注布局。

## 7. 标签政策：严禁重现旧 GT 问题

固定八类：`night, indoor, rain_snow, office, outdoor, landscape, sports, objective_image`。

MIRFLICKR 本阶段只映射它确实提供的 `night` 和 `indoor`。全部原始人工概念，包括 potential 与 `_r1`，另存 `original_manual_labels.jsonl`。

若存在 `_r1`：相关正样本为 1；不在完整 potential 列表中的样本为 0；在 potential 但不在 `_r1` 的边界样本为 -1。若某概念没有 `_r1`，使用其原生 potential 人工二值标注并记录来源。不要把潜在相关与显著相关混淆。

这里的 night=0 是原始人工概念体系下的非夜景，不是根据亮度猜出来的，也不等于 Day 标签。indoor=0 不自动推出 outdoor=1。

其余六类全部 -1：不把河流/植物等于自然风景，不把全部照片当 objective=0，不根据 clouds/sky 推雨雪。

雨雪逻辑保持：rain=1 或 snow=1 -> 1；rain=0 且 snow=0 -> 0；其余 -> -1。MIRFLICKR 当前无这两个直接映射来源，所以 rain_snow=-1。

这不是八类完整训练集。六类缺失是本数据源的正常限制，只报告，不能伪造标签或触发下载无关大数据集。仅要求 night、indoor 在三个新 split 内均有正负监督。

## 8. 去重、划分和尺寸统计

脚本按完整解码 RGB 内容分组，再以 night/indoor 联合分层生成约 80/10/10 的自定义 train/val/test。不是官方划分。重复图的相关标签冲突只在新标签里设 unknown，保留原始标签和冲突审计。

三个新 split 的相同内容交集必须为零。代码不会声称已排除相似图、重新编码图、同拍摄序列，或与旧 COCO/Places/10_scenes 的交集；后续混合训练前仍须统一核验。

原图不缩放、不复制成训练图片集。只有独立审核目录包含缩略可视化。

`resolution_audit.json` 只基于新 TRAIN 统计宽高 P10/P50/P90、竖图比例，以及 256x144、320x180、384x216、512x288、640x360 在 letterbox 下的放大比例。输出原生尺寸上限建议，不固定 640x360，不代表已验证最佳精度。最终合并全部训练来源后还需重算。

## 9. 核查产物并汇报

```text
MIRFLICKR25K/
  downloads/                         官方 ZIP，失败时保留 .part
  raw/images/                        新下载的原图和原始 metadata
  raw/annotations_v080/              原始人工 annotation 和 README
  archive_audit.json
  derived/mir25k_nas8_v1/
    train.jsonl / val.jsonl / test.jsonl
    train.csv / val.csv / test.csv
    original_manual_labels.jsonl
    annotation_audit.json
    duplicate_audit.json
    dataset_summary.json
    resolution_audit.json
    review_samples.csv
    review/index.html                48 张独立审核图，每行 GT=1/0/?
    PREPARED.json
```

读取 annotation_audit.json 中记录的原始 README（不是 Flickr 用户 tags），确认处理文件来自人工 v080。若真实 README 与 parser 假设矛盾，停止，不擅自修改原标注或 parser。

最终返回：

```text
STATUS: PREPARED_MIRFLICKR25K / BLOCKED
GITHUB_BRANCH / COMMIT:
ENVIRONMENT: Ultraface
TOTAL_IMAGES:
IMAGE_OFFICIAL_MD5_PASS:
BOTH_ZIP_CRC_PASS:
ANNOTATION_CONCEPTS:
NIGHT_MAPPING: potential / relevance-aware
INDOOR_MAPPING: potential / relevance-aware
TRAIN / VAL / TEST COUNTS:
每 split 的 night、indoor positive / negative / unknown:
EXPECTED_UNKNOWN_LABELS: rain_snow, office, outdoor, landscape, sports, objective_image
IDENTICAL_RGB_CROSS_SPLIT_OVERLAP:
DUPLICATE_LABEL_CONFLICTS:
IMAGE_WIDTH_HEIGHT_P10_P50_P90:
NATIVE_RESOLUTION_UPPER_BOUND_WH:
FINAL_TRAINING_INPUT: NOT YET SELECTED FOR COMBINED DATASET
REVIEW_HTML:
MANIFEST_ROOT:
OLD_DATASETS_MODIFIED: NO
OLD_TRAINING_OR_WATCHDOG_MODIFIED: NO
TRAINING_STARTED: NO
READY_FOR_8LABEL_TRAINING: NO (only one source prepared)
HUMAN_ACTION_REQUIRED: YES / NO
```

正常完成不需要用户人工下载。若下载或数据错误需人工处理，说明具体动作。代码本地有 bug 时停止并返回完整 traceback、命令、commit，不能让 CodeAgent 自行修代码。视觉标签质量和产品用途许可审核是后续合并训练的事项，不得把准备成功写成已证明八类精度或可直接商用。
