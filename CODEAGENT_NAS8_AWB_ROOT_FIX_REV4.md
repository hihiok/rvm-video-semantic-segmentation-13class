# NAS 八类：AWB 根目录修复、完整清洗和 GT100 — Rev4 执行指令

本文件替代 Rev3 继续执行指令。上轮在 legacy 清洗阶段因默认源根目录遗漏
AWB_10_scenes 而 BLOCKED；这是代码默认配置缺漏，不是数据损坏。
MIR 和已核验 NUS21 解析保留，上轮输出及诊断保留，复用已完整解压的 cache。
当前目标是八类数据准备及可视化，不是历史九标签训练任务。

本次修复：
- 默认包含五个独立图片根目录，其中 10_scenes 和 AWB_10_scenes 分开列出。
- --source-roots 接受一个或多个显式根目录，不再固定四个槽位。
- 路径 resolve 后仍检查每张旧清单图片，输出/cache 重叠保护保留。
- 不允许把公共祖先 /data/pub1/z00919662/dataset 当作替代源根。
- 两个目录在旧 manifest 中的 source 均为 10_scenes，保持现有标签规则。
  AWB_10_scenes/Night 为 night 弱正，不补雨雪、室内、户外标签。
- source_audit.json 新增 legacy_root_counts，逐条 source_records 新增
  source_dataset_root，便于分别核对两个物理目录。

## 1. 已确认的结论和范围

固定类别顺序：night, indoor, rain_snow, office, outdoor, landscape, sports, objective_image。
每类取 1/0/-1，unknown 不参与该类监督，不能补零。

用户的 NUS-WIDE 镜像内有 21 类检索标签：

sky, clouds, person, water, animal, grass, buildings, window, plants, lake,
ocean, road, flowers, sunset, reflection, rocks, vehicle, snow, tree, beach, mountain。

- snow 是从 0 开始的第 17 列（第 18 列），正例共 5404：
  database 5227，query/test 177。这是图片质量检查前的标注行数。
- 没有 nighttime / sports，不能再声称该包可补这两类。
- snow=1 -> rain_snow=1；snow=0 -> rain_snow=-1。
- 其他七个 NAS 类均不由这个包自动生成标签；sunset 不能当 night。
- 水、草、山等物体标签不能 OR 成 landscape，person 不能当 sports。
- 不使用 tc10 重复扩充数据，不扫描无配对标注的剩余图片来制造 GT。

映射证据：21 列正例数量分别与官方81类人工GT唯一匹配；恢复的类别顺序恰好
是官方频次前21类；全部4580种非空21维标签组合及其计数完全一致。
完整组合匹配也意味着所有两类共现计数一致。两个索引文件与对应 multi-hot
文件逐行一致；database 和 query 路径无交集。

这是针对特定元数据的统计恢复，不能代替对原始图片的视觉审核。
图片/标签配对遵循包内显式清单；并未声称与未提供的官方 ImageList 逐图比对。
检索 query split 不是官方 NUS Test split。

证据配置已提交 nas_data/scene8_v3/nus21_verified_profile.json，包含六个文件
的字节数和 SHA256、官方 GT 校验值、类别顺序及组合分布校验值。
运行时六个文件必须完全匹配，不能换任意另一份21列文件套用这个顺序。
服务器不需要下载任何新的图片或标注。

## 2. proxy、SSL、GitHub 更新

GitHub：https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git

Branch：agent/nas8-mir-nus-clean-gt100-v3

本任务 checkout：/data/pub1/z00919662/segmentation/nas8-mir-nus-clean-v3

沿用此前成功的企业代理；GitHub 文档不保存密码，私有下载版包含配置。
禁止打印代理变量、git config --list 或开启 shell tracing。

```bash
set +x
set -euo pipefail
umask 077
: "${http_proxy:?先加载现有私有 proxy 配置}"
export https_proxy="${https_proxy:-$http_proxy}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$https_proxy"
# 用户指定：企业代理环境 clone/fetch 前跳过 SSL verification。
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=1
PROJECT_ROOT=/data/pub1/z00919662/segmentation/nas8-mir-nus-clean-v3
BRANCH=agent/nas8-mir-nus-clean-gt100-v3
test -d "$PROJECT_ROOT/.git"
test -z "$(git -C "$PROJECT_ROOT" status --porcelain)"
test "$(git -C "$PROJECT_ROOT" remote get-url origin)" = https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
git -C "$PROJECT_ROOT" fetch origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
git -C "$PROJECT_ROOT" checkout "$BRANCH"
git -C "$PROJECT_ROOT" merge --ff-only "origin/$BRANCH"
test "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" = "$(git -C "$PROJECT_ROOT" rev-parse "origin/$BRANCH")"
test -s "$PROJECT_ROOT/nas_data/scene8_v3/nus21_verified_profile.json"
test -s "$PROJECT_ROOT/nas_data/scene8_v3/nus21.py"
test -s "$PROJECT_ROOT/nas_data/scene8_v3/test_legacy_roots.py"
test -s "$PROJECT_ROOT/CODEAGENT_NAS8_AWB_ROOT_FIX_REV4.md"
git -C "$PROJECT_ROOT" rev-parse HEAD
```

记录实际 commit，执行期间不再滚动更新。不要切换其他训练 checkout。
本地有修改则保留并返回 diff --stat，请用户把源码/补丁同步给 ChatGPT。
禁止 CodeAgent 自行写/改 Python、shell、config、MD；禁止 stash/reset --hard/
git clean/force push。所有代码修改由 ChatGPT 提交 GitHub 后再同步执行。

## 3. 环境、输入、保护范围

初始化实际已有的 conda.sh，然后使用 ultraface（已确认 Python3.8.20、
Pillow10.4.0）。不新建环境、不升级依赖、不占 GPU。

```bash
conda activate ultraface
test "$CONDA_DEFAULT_ENV" = ultraface
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$PROJECT_ROOT/nas_data/scene8_v3"
python -c 'import sys,PIL; print(sys.version); print(PIL.__version__)'
python -m unittest discover -s . -p 'test*.py'
bash -n run_prepare.sh
```

必须42项测试通过，包括新增的五根默认配置、五来源旧清单解析、Night与合成图
标签及物理来源追踪、相邻目录/符号链接越界拒绝、CLI五根与输出重叠保护。
原有MIR、NUS指纹/行配对/雪负例unknown、GT100测试仍须通过。测试图不是用户真实GT100。

输入路径（全部只读）：

```text
旧八类标签：
/data/pub1/z00919662/dataset/UltraFaceSlim_8scene_multilabel_manifests_640x360_v1
MIR图片ZIP：
/data/pub1/z00919662/dataset/mirflickr25k.zip
MIR标注ZIP：
/data/pub1/z00919662/dataset/mirflickr25k_annotations_v080.zip
已解压cache：
/data/pub1/z00919662/dataset/NAS8_new_sources_raw_v3
已解压NUS根：
/data/pub1/z00919662/dataset/NAS8_new_sources_raw_v3/nus
旧图片根：
/data/pub1/z00919662/segmentation/datasets/coco
/data/pub1/z00919662/segmentation/datasets/places365
/data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360
/data/pub1/z00919662/dataset/10_scenes
/data/pub1/z00919662/dataset/AWB_10_scenes
```

旧JSONL只作为图片清单；不沿用不可信的旧labels。原图、mask、ZIP、旧标签、
checkpoint、旧训练、watchdog全部不改。cache复用完成标记；不删不重解压。
MIR仍复核ZIP和完成标记；NUS通过 --nus-root 复用，避免重复读取巨大archive.zip。

## 4. 直接继续完整数据准备

不再执行 Rev2 source_preflight.py 后停在等待映射。直接跑下面命令。
输出路径必须新建；已有路径则使用时间戳，不覆盖旧BLOCKED或诊断结果。

```bash
OLD_LABEL_ROOT=/data/pub1/z00919662/dataset/UltraFaceSlim_8scene_multilabel_manifests_640x360_v1
CACHE_ROOT=/data/pub1/z00919662/dataset/NAS8_new_sources_raw_v3
NEW_LABEL_ROOT=/data/pub1/z00919662/dataset/NAS8_multilabel_clean_v3_rev4
if [[ -e "$NEW_LABEL_ROOT" ]]; then
  NEW_LABEL_ROOT="${NEW_LABEL_ROOT}_$(date +%Y%m%d_%H%M%S)"
fi
for split in train val test; do test -s "$OLD_LABEL_ROOT/$split.jsonl"; done
test -d "$CACHE_ROOT/nus"
df -h /data/pub1/z00919662/dataset
LOG="/data/pub1/z00919662/dataset/nas8_clean_rev4_$(date +%Y%m%d_%H%M%S).log"
bash run_prepare.sh \
  --old-root "$OLD_LABEL_ROOT" \
  --mir-images /data/pub1/z00919662/dataset/mirflickr25k.zip \
  --mir-annotations /data/pub1/z00919662/dataset/mirflickr25k_annotations_v080.zip \
  --nus-root "$CACHE_ROOT/nus" \
  --cache-root "$CACHE_ROOT" \
  --output-root "$NEW_LABEL_ROOT" \
  --source-roots \
    /data/pub1/z00919662/segmentation/datasets/coco \
    /data/pub1/z00919662/segmentation/datasets/places365 \
    /data/pub1/z00919662/segmentation/datasets/COCO_ADE_13cls_16x9_640x360 \
    /data/pub1/z00919662/dataset/10_scenes \
    /data/pub1/z00919662/dataset/AWB_10_scenes \
  --seed 20260910 2>&1 | tee "$LOG"
```

不要传 --nus-metadata-root，不需要补 Concepts81。当前镜像将自动进入
retrieval_top21_verified 解析器，六文件指纹通过后按清单行号匹配图片。
找不到图片则记录缺失并保留原行号；不得删除缺失行后重新按顺序配标签。

MIR上轮真实25K已PASS；people_r1四个差集继续作为审计，不阻断。
night/indoor显式relevant正例优先，potential-only unknown，原始标注不改。

## 5. 清洗与划分规则

- 夜景：MIR人工night负责真实照片正负；旧COCO/SEG/Places的无依据夜景标签撤销。
  NUS21夜景全部unknown，AWB_10_scenes/Night只提供明确标记的弱正。
- 雨雪：雨或雪任一确认存在为正；只有两者都确认不存在为负。NUS只提供雪正。
  不用SEG13的ice_or_snow或“无雪”推出“无雨雪”，不整类从Places猜天气。
- 风景：Places明确自然场景可提供弱标签；城市/主体不明确的区别处理。
  撤销分割自然区域面积大就算风景，宽泛10_scenes landscape留待审核。
- 运动/办公：使用明确场景类别和用户已确认单图；球/人/电脑不等同对应场景。
- 室内/户外：MIR indoor与仓库官方Places IO表；边界场景unknown；
  indoor=0不自动等于outdoor=1。
- 客观图：沿用用户定义的Computer_synthesized等正例，保留真实办公/屏幕困难负例，
  控制普通照片负监督数量，避免淹没正类。
- 保留已确认单图修正：草地打球ADE_train_00001854.jpg为sports=1、landscape=0；
  大象戏水000000465180.jpg为landscape=0；其余报告问题图按原V3逐项追踪。
- 损坏/过小/无有效监督/重复等记录排除原因。NUS非雪样本八类全unknown，
  不会凭空当负例；可保留部分供审计，默认不作为有监督训练样本。
- 检索query固定test，database确定性分train/val；旧数据保留eval属性。
  路径/完全相同像素/限定COCO来源ID分组，eval优先；只在完全相同像素间合并标签，
  冲突unknown；可能跨split近重复保守排除并报告。
- 新split不能让看过旧数据的checkpoint变成独立测试模型，本轮不训练。

## 6. 必须完成的产物

继续到全部产物完成，不能只输出“解析通过”：

- train/val/test.jsonl与CSV、val_strict/test_strict；
- source_audit.json（包括NUS指纹、列语义恢复证据、实际雪正例和缺图数）；
- source_records.jsonl、excluded.jsonl、legacy_label_changes.jsonl；
- reported_examples.json、near_duplicate_review.json、summary.json；
- GT100：gt100/index.html、100张独立JPEG、gt100.csv、10张contact图；
- review100_template.csv、weather_review_template.csv、resolution_audit.json。

核对 source_audit.json 的 legacy_root_counts（清洗前清单行数，非最终训练数）：
COCO 7000、Places365 219000、SEG13 115795、10_scenes 3077、AWB_10_scenes 31907。
总数为376779。若与上轮不同，先核对 old_manifest_files 中的哈希及源清单变化，
如实报告，不通过删行、改源名或改标签凑数。
两个10_scenes物理目录继续归入同一逻辑来源，不增加或改变GT100配额。

GT100固定配额：COCO15、Places36520、SEG13 15、MIR20、NUS20、10_scenes10，
共100个不重复底层组。每张显示八类完整1/0/?、路径、来源、证据和是否进manifest。
显示GT，不跑模型预测。包含少量unknown/排除图时必须明确标识，不冒充训练样本。

分辨率基于最终TRAIN原图统计；候选256x144、320x180、384x216、512x288、
640x360。不预先固定640x360，SEG13派生尺寸不能当原生细节依据。
当前只给小输入起点建议，不宣称已通过精度对照确定最佳分辨率。

## 7. 报告和人工动作

成功状态应为 PREPARED_REVIEW_REQUIRED。缺某类可靠正/负监督属于
training_quality_blockers：不阻止GT100生成，但必须如实报告，不能补0消除缺口。
用户需查看GT100；雨雪真实负例不足时需在审核CSV明确确认既无雨也无雪。
未经人工批准不要填reviewed=1，也不要启动训练。

报告：STATUS、branch/commit、环境、NEW_LABEL_ROOT、完整日志路径、各来源清洗
前后数量、五个物理根的 legacy_root_counts、各split每类正/负/unknown、真实照片night/rain_snow负例数、
NUS_FORMAT、NUS列顺序、雪正例5227/177与实际可用差异、缺图数、
排除原因、split泄漏检查、GT100来源配额和路径、分辨率建议及训练质量缺口。

```text
OLD_IMAGES_LABELS_CHANGED: NO
OLD_TRAINING_WATCHDOG_CHANGED: NO
NEW_TRAINING_STARTED: NO
HUMAN_ACTION_REQUIRED: YES（说明需要查看GT100及具体标签审核事项）
```

真正代码/文件损坏/六文件指纹不一致时保存BLOCKED.json、source_audit.json、
traceback和诊断；停止并同步给ChatGPT，禁止本地patch。
当前已核验的这份包不再以“缺Concepts81/类别名文件”作为阻塞理由。
