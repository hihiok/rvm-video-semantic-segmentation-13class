# NAS8 MIR 修复与 NUS 源数据预检 — Rev2

本轮承接 c418fd557f0c7560379403096e4c14f7ac3eabce 的 BLOCKED 报告。
ChatGPT 已修复 MIR relevance 子集假设；NUS 数值索引与类别名称尚无已核验映射，
本版不声称已支持这个检索包。先从现有 cache 导出实际元数据，不能凭列数造 GT。
本任务是八类数据准备，不是历史九标签训练任务。

## 1. GitHub、proxy 与 SSL

仓库：https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git

分支：agent/nas8-mir-nus-clean-gt100-v3

checkout：/data/pub1/z00919662/segmentation/nas8-mir-nus-clean-v3

沿用上轮已配置成功的企业代理。GitHub 文档不保存代理密码；
私有下载版提供代理配置。不能打印环境变量、代理凭据或 git config --list。

```bash
set +x
set -euo pipefail
umask 077
: "${http_proxy:?请载入上轮私有代理配置}"
export https_proxy="${https_proxy:-$http_proxy}"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$https_proxy"
# 用户指定：企业代理环境 clone/fetch 前跳过 SSL verify。
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
test -s "$PROJECT_ROOT/nas_data/scene8_v3/source_preflight.py"
git -C "$PROJECT_ROOT" rev-parse HEAD
```

发现本地改动或非 fast-forward，不要 stash/reset/clean/覆盖：返回 diff --stat，
请用户把源码或补丁传给 ChatGPT。禁止 CodeAgent 自行修改 Python、shell、配置或 MD。

## 2. 环境与测试

初始化实际已有的 conda.sh，使用已确认的环境名 ultraface，Python 3.8.20。
不新建环境，不升级 torch/Pillow，不用 GPU。

```bash
conda activate ultraface
test "$CONDA_DEFAULT_ENV" = ultraface
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$PROJECT_ROOT/nas_data/scene8_v3"
python -c 'import sys,PIL; print(sys.version); print(PIL.__version__)'
python test_pipeline.py
bash -n run_prepare.sh
```

必须 28 项离线测试通过。run_prepare.sh 现在同时接受 ultraface / Ultraface，
不要为满足字符串校验伪造 CONDA_DEFAULT_ENV。

## 3. 复用 cache，执行真实源文件预检

无需重新下载或解压任何图片/标签 ZIP。以下目录来自上一轮成功解压：

- MIR 图片：/data/pub1/z00919662/dataset/NAS8_new_sources_raw_v3/mir_images
- MIR 标注：/data/pub1/z00919662/dataset/NAS8_new_sources_raw_v3/mir_annotations
- NUS：/data/pub1/z00919662/dataset/NAS8_new_sources_raw_v3/nus

如目录缺失，返回实际 cache 清单；不能自行重新生成标注或清理 cache。

```bash
CACHE_ROOT=/data/pub1/z00919662/dataset/NAS8_new_sources_raw_v3
DIAG_ROOT="/data/pub1/z00919662/dataset/NAS8_source_preflight_rev2_$(date +%Y%m%d_%H%M%S)"
test -d "$CACHE_ROOT/mir_images"
test -d "$CACHE_ROOT/mir_annotations"
test -d "$CACHE_ROOT/nus"
set +e
python -u source_preflight.py \
  --nus-root "$CACHE_ROOT/nus" \
  --mir-image-root "$CACHE_ROOT/mir_images" \
  --mir-annotation-root "$CACHE_ROOT/mir_annotations" \
  --output-root "$DIAG_ROOT" 2>&1 | tee "${DIAG_ROOT}.log"
PREFLIGHT_EXIT=${PIPESTATUS[0]}
set -e
printf 'PREFLIGHT_EXIT=%s\nDIAG_ROOT=%s\n' "$PREFLIGHT_EXIT" "$DIAG_ROOT"
```

预期 MIR_STATUS=PASS、MIR_IMAGES=25000；people_r1 差集四个 ID 留在 source_audit.json。
原始列表保持不动。实际参与八类的 night/indoor 如有 relevant 超出 potential，
显式 relevant 正例优先，potential-only 保持 unknown；不把显式正例变成负例。
非法 ID、重复 ID、缺少必要文件仍然报错。

NUS 诊断包包含原始 README、受限的类别/图片清单/标签文件、行数、列数、
整数值统计、前几行样例和 SHA256；不包含图片、模型、代理或其他用户文件。
它还包含 source_preflight.json 和 MIR source_audit.json。

若返回 NUS_RETRIEVAL_MAPPING_UNVERIFIED / exit=2：

1. 保留并提供 `$DIAG_ROOT/nus_diagnostics.zip` 的完整路径。
2. 明确要求用户将这个 ZIP 上传给 ChatGPT；无需上传巨大 archive.zip 或 JPG。
3. 不重复跑 prepare_all.py，不因诊断包生成成功而声称数据处理成功。
4. 不跳过 NUS、不用 10/21 类索引冒充 81 类、不把 unknown 改 0。

这是尚缺实际标注证据的阻塞，并非 MIR 修复失败。完成诊断和真实 MIR 校验后，
本轮到此返回结果；按旧文档第 10 节，不得自行猜测、patch 后继续。

## 4. 仅当源预检通过时继续

如服务器已另行备齐官方 Concepts81、ImageList 和对应 Groundtruth，
允许通过 --nus-metadata-root 指向那个只读目录；必须明确报告实际路径和来源，
不能让 CodeAgent 临时手写类别表。照片仍从现有 cache/nus 读取。

只有 SOURCE_PREFLIGHT_PASS 才继续执行
`CODEAGENT_NAS8_MIR_NUS_CLEAN_GT100_V3.md`：

- 新输出：/data/pub1/z00919662/dataset/NAS8_multilabel_clean_v3_rev2；
  已存在则使用带时间戳的新路径，不覆盖上轮 BLOCKED 目录。
- 复用 CACHE_ROOT，必要时向 run_prepare.sh 传相同 --nus-metadata-root。
- 完成旧标签清洗、去重划分、各类正/负/unknown 统计、GT100 和尺寸审计。
- 旧图片/标签、训练/watchdog 全部不动，不启动训练。
- 雨或雪任一确认存在为正；缺雪不能推出无雨雪。
- GT100 配额：COCO 15、Places365 20、SEG13 15、MIR 20、NUS 20、10_scenes 10。

## 5. 最终报告

输出 STATUS、branch/commit、环境、MIR_STATUS/MIR_IMAGES、MIR 差集 ID、
NUS_FORMAT、DIAG_ROOT、nus_diagnostics.zip 绝对路径、退出码及错误。

遇到本轮预期的 NUS 未核验格式：

```text
STATUS: BLOCKED_NUS_MAPPING_EVIDENCE_REQUIRED
MIR_STATUS: PASS (以实际校验为准)
HUMAN_ACTION_REQUIRED: YES
ACTION: 将 DIAG_ROOT/nus_diagnostics.zip 上传给 ChatGPT
MANIFESTS_WRITTEN: NO
OLD_IMAGES_LABELS_CHANGED: NO
OLD_TRAINING_WATCHDOG_CHANGED: NO
NEW_TRAINING_STARTED: NO
```

不要要求人工重新下载照片。只有核实实际元数据后确认缺原始标注时，
才应由 ChatGPT 给出具体的小标注包/清单补充方案。
