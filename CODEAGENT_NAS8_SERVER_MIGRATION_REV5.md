# NAS 八类：新服务器迁移、标签清洗及 GT100 — Rev5 完整执行指令

本任务继续八类数据准备，不执行历史九标签训练文档。用户已停止旧服务器任务。
数据已迁往 /mnt/ssd1/z00919662/datasets；其中 NAS8_multilabel_clean_v3_rev4
是未完成的派生输出，保留原样，不当作已完成标签，也不作为旧原始清单输入。
本文件替代 Rev4。不要重启旧训练、watchdog 或占 GPU。

本次代码新增：
- --relocated-datasets-root 设置新服务器路径，并按五个精确数据集前缀转换旧图片路径。
- 保留每条 legacy_image_original 和稳定 sample_id；新 image 使用新服务器真实路径。
- 不按文件名搜索配图，不直接修改旧 JSONL，不建立伪造旧路径的软链接。
- legacy 清单的路径校验与标签重建提前执行；迁移缺图汇总到 source_audit.json。
- 已完整解压的 MIR 可通过两个 root 参数直接复用，无需为了路径变化重新解压。
  此模式不声称重新校验 ZIP MD5：检查完整25K编号、人工标注ID/哈希和选入图片解码。
- NUS 仍要求已核验六文件 SHA256，映射和监督范围与 Rev3/4 相同。
- 完整流程从原始清单重建新版本；不拼接未完成的中间 manifest。

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

## 2. proxy、SSL、GitHub 与环境

Repository: https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
Branch: agent/nas8-mir-nus-clean-gt100-v3
执行文件: CODEAGENT_NAS8_SERVER_MIGRATION_REV5.md
代码默认目录: /mnt/ssd1/z00919662/NAS_scene_detection/nas8-mir-nus-clean-v3

使用用户既有私有 proxy 配置。公开文档不保存密码，私有下载版提供完整设置。
按用户指定在 clone/fetch 前跳过 SSL verification。禁止输出代理变量、开启 set -x。

```bash
set +x
set -euo pipefail
umask 077
: "${http_proxy:?先加载用户私有代理配置}"
export https_proxy="${https_proxy:-$http_proxy}"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$https_proxy"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=1
PROJECT_ROOT=/mnt/ssd1/z00919662/NAS_scene_detection/nas8-mir-nus-clean-v3
BRANCH=agent/nas8-mir-nus-clean-gt100-v3
REPO=https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
if [[ ! -e "$PROJECT_ROOT" ]]; then
  mkdir -p "$(dirname "$PROJECT_ROOT")"
  git clone --branch "$BRANCH" --single-branch "$REPO" "$PROJECT_ROOT"
fi
test -d "$PROJECT_ROOT/.git"
test -z "$(git -C "$PROJECT_ROOT" status --porcelain)"
test "$(git -C "$PROJECT_ROOT" remote get-url origin)" = "$REPO"
git -C "$PROJECT_ROOT" fetch origin "refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
git -C "$PROJECT_ROOT" checkout "$BRANCH"
git -C "$PROJECT_ROOT" merge --ff-only "origin/$BRANCH"
test "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" = "$(git -C "$PROJECT_ROOT" rev-parse "origin/$BRANCH")"
test -s "$PROJECT_ROOT/nas_data/scene8_v3/relocation.py"
test -s "$PROJECT_ROOT/CODEAGENT_NAS8_SERVER_MIGRATION_REV5.md"
git -C "$PROJECT_ROOT" rev-parse HEAD
```

若目标目录已存在且不是本仓库，保留该目录，选择同父目录一个新的 checkout 名。
若 checkout 有本地修改，先停止，返回 diff --stat；需把源码/补丁同步给 ChatGPT，
不要覆盖、stash、reset --hard 或 git clean。记录本轮实际 commit，不中途滚动更新。

只读查找本机现有 conda.sh，初始化后激活已有 ultraface 环境（Ultraface 大小写
不同但确为同一用途的已迁移环境也可）。不新建环境、不升级依赖。
若本机未迁移环境，明确报告需要迁移/安装的环境，不能声称已有环境可用。

```bash
conda activate ultraface
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$PROJECT_ROOT/nas_data/scene8_v3"
python -c 'import sys,PIL; print(sys.version); print(PIL.__version__)'
python -m unittest discover -s . -p 'test*.py'
bash -n run_prepare.sh
```

必须49项测试全部通过；包含迁移路径、越界/缺图阻断、只读旧清单、合成100图完整
流程测试。合成测试会替换原生MIR/NUS解析器；真实解析由原有独立测试和本机全量校验覆盖。

## 3. 核对新服务器文件（只读）

数据父目录：/mnt/ssd1/z00919662/datasets

预期已迁移输入：

| 用途 | 父目录下的相对路径 |
|---|---|
| 旧八类原始清单 | UltraFaceSlim_8scene_multilabel_manifests_640x360_v1/train.jsonl、val.jsonl、test.jsonl |
| COCO | coco/ |
| Places365 | places365/（内部 versions/1/train、val 等层次保持原样） |
| SEG13 | COCO_ADE_13cls_16x9_640x360/ |
| 合成客观图等 | 10_scenes/ |
| Night弱正 | AWB_10_scenes/ |
| 已解压原始数据cache | NAS8_new_sources_raw_v3/mir_images、mir_annotations、nus |
| 若未迁移解压数据，可用本地ZIP | mirflickr25k.zip、mirflickr25k_annotations_v080.zip、archive.zip |
| 上轮未完成输出（保留） | NAS8_multilabel_clean_v3_rev4/ |

优先读取目录列表、旧任务日志、_EXTRACTED.json 和少量旧清单行确认实际布局。
CodeAgent 可以只读查找文件并调整下面已有 CLI 参数的路径值，不得改源码或生成新解析器。
不得把不完整 Rev4 的 train/val/test 或 source_records 冒充原始旧 manifest。

默认自动转换两种历史前缀下的上述五个数据集：
/data/pub1/z00919662/segmentation/datasets/<dataset>/...
/data/pub1/z00919662/dataset/<dataset>/...
到 /mnt/ssd1/z00919662/datasets/<dataset>/...，内部相对路径完整保留。
已经指向新服务器的路径也接受。两个10_scenes目录不能合并。

若实际目录名不同，可在命令同时显式传 --source-roots（列齐五个真实根）及
--legacy-path-map OLD_DATASET_ROOT NEW_DATASET_ROOT（每个不同旧前缀重复一次）；
只依据真实迁移对应关系，不按 basename 猜测，不传 datasets 公共祖先作为源根。
--old-root、--cache-root、--mir-images、--mir-annotations、--nus-archive 等也可显式覆盖。

缺少文件时先只读确认是否位于新父目录的其他位置；找不到再报告需要从旧服务器
补迁移的准确文件/目录和新目标路径，不自动重新下载、不手工造对应关系。
旧manifest、原图、原始标签和旧Rev4输出均保持原样。

## 4. 执行到完整结果

默认命令复用已解压cache。若实际MIR或NUS独立放在 MIRFLICKR-25K/、NUS-WIDE/
等目录，只读确认内容后用 --mir-image-root/--mir-annotation-root/--nus-root 指定
各自准确的根目录；MIR两个root必须成对提供。不要把整个datasets父目录传作图片根。

```bash
DATASETS_ROOT=/mnt/ssd1/z00919662/datasets
OLD_LABEL_ROOT="$DATASETS_ROOT/UltraFaceSlim_8scene_multilabel_manifests_640x360_v1"
CACHE_ROOT="$DATASETS_ROOT/NAS8_new_sources_raw_v3"
NEW_LABEL_ROOT="$DATASETS_ROOT/NAS8_multilabel_clean_v3_rev5_$(date +%Y%m%d_%H%M%S)"
LOG="$DATASETS_ROOT/nas8_clean_rev5_$(date +%Y%m%d_%H%M%S).log"
for split in train val test; do test -s "$OLD_LABEL_ROOT/$split.jsonl"; done
for src in coco places365 COCO_ADE_13cls_16x9_640x360 10_scenes AWB_10_scenes; do
  test -d "$DATASETS_ROOT/$src"
done
MIR_ARGS=(--mir-images "$DATASETS_ROOT/mirflickr25k.zip" --mir-annotations "$DATASETS_ROOT/mirflickr25k_annotations_v080.zip")
if [[ -d "$CACHE_ROOT/mir_images" && -d "$CACHE_ROOT/mir_annotations" ]]; then
  MIR_ARGS=(--mir-image-root "$CACHE_ROOT/mir_images" --mir-annotation-root "$CACHE_ROOT/mir_annotations")
fi
NUS_ARGS=(--nus-archive "$DATASETS_ROOT/archive.zip")
if [[ -d "$CACHE_ROOT/nus" ]]; then
  NUS_ARGS=(--nus-root "$CACHE_ROOT/nus")
fi
df -h "$DATASETS_ROOT"
bash run_prepare.sh \
  --relocated-datasets-root "$DATASETS_ROOT" \
  --old-root "$OLD_LABEL_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --output-root "$NEW_LABEL_ROOT" \
  "${MIR_ARGS[@]}" "${NUS_ARGS[@]}" \
  --seed 20260910 2>&1 | tee "$LOG"
```

不复用 .extracting 目录。已有不完整cache不能删除或覆盖；若仅剩ZIP，可显式选择
新的独立 --cache-root 解压，并先确认磁盘空间。MIR ZIP模式仍校验官方MD5；
直接root模式将如实记录未重验ZIP哈希。NUS指纹、逐行图标配对规则不放宽。

缺图会在 legacy 阶段汇总并写 BLOCKED.json/source_audit.json；图片解码失败或
过小则由后续质量扫描排除并记录。处理日志持续输出进度，不占用GPU。
任务尚未结束时不要把存在 train.jsonl 当作成功；只有 PREPARED.json +
summary.json 的 PREPARED_REVIEW_REQUIRED 且 GT100完整才算本轮准备完成。

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
总数为376779，这是上轮旧清单的参考值，不是新训练数。若与上轮不同，先核对 old_manifest_files 中的哈希及源清单变化，
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
前后数量、五个物理根的 legacy_root_counts、path_relocation 的转换/缺图数、各split每类正/负/unknown、真实照片night/rain_snow负例数、
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
