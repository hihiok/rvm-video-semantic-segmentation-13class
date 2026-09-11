# CodeAgent：先完成 NAS8 Rev3 数据，再续跑 MobileCLIP2-S0（2026-09-11）

本文件完整替代上一版“数据未准备好就结束整个任务”的执行流程。
本次任务包括：运行已有 Rev3 数据准备代码 → 检查完整输出 → GPU smoke → 全量 test 诊断评测。
这些都是本任务步骤；不要再要求用户手动运行 run_prepare.sh、重新授权数据准备、手动传回本轮输出路径或另开会话。
真正需要用户的是 GT100/天气标签审核，或无法自动解决的文件、权限、资源与代码问题；逐项说明。
本文件不授权训练、自动审核 GT、改源码、改原始数据或停止其他进程。

GitHub 仓库：https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
数据准备分支：agent/nas8-mir-nus-clean-gt100-v3
数据准备固定 commit：c8b3bd4b7585040950d4257046ca8d8a0958c521
benchmark 分支：agent/nas8-mobileclip2-rev3-resume-20260911
benchmark 修复及测试 commit：1609335dec728b8c62364fc7fbe4b0b4ecfa3bee
GitHub 完整指令：CODEAGENT_NAS8_REV3_PREP_THEN_MOBILECLIP2_20260911.md

已核对两个版本的实际源码：Rev3 summary.schema 是 nas8_source_curation_v3_rev3，
之前的 benchmark 只接受 nas8_source_curation_v3，这是继数据未生成之后的另一个兼容问题。
新分支已修复，19 项离线测试通过；未在目标服务器运行数据准备或 GPU，必须现场执行以下检查。
不修改数据 schema 来迁就旧 benchmark。

固定八类：night, indoor, rain_snow, office, outdoor, landscape, sports, objective_image。
1=正、0=有证据负、-1=unknown；unknown 不补零。
NUS21 仅提供 snow 正例，对应 rain_snow=1；无雪仍为雨雪 unknown。
该包没有 nighttime/sports；不能用 sunset 代替 night、物体区域代替自然风景主体。
更多八类政策见阶段 B。

## 先配置 proxy 和 SSL

公开版使用现有授权代理变量；下载的私有版预置代理。不要打印凭据，不开启 set -x；
不输出完整 env、git config --list 或 pip config debug；不要提交 proxy.md/私有 MD。

~~~bash
set +x
# PRIVATE_PROXY_PRESET
test -n "$http_proxy"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$https_proxy"
# 在任何 clone/fetch 之前，按用户指定跳过 SSL verification。
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=1
export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org"
export HF_HUB_DISABLE_XET=1
export PYTHONDONTWRITEBYTECODE=1
set -euo pipefail
umask 077
~~~

Git 与 pip/HF 的 TLS 配置分别生效；阶段 B 的 HF 下载使用 --insecure-downloads。
不sudo、不修改系统驱动/cuDNN、不绕过明确的403/组织访问拦截。
源码、下载或导入错误要保留脱敏 traceback；不要统一归咎于系统cuDNN版本。

## 阶段 A：完整执行已有 Rev3 数据准备

先确认没有另一个 prepare_all.py 正在使用同一 cache；若正在运行，跟踪现有进程和日志，
等其完成后检查它的实际输出。不得同时启动第二份准备，不停止既有任务。
如果已存在完整 Rev3 输出且能核实其提交、输入、完成状态，就复用它，不重复生成。
目前用户报告“Rev3尚未执行”，按以下流程完成一次。数据准备只用CPU，不下载模型。

1. 复用用户报告的专用 Rev3 checkout；先验证 GitHub 来源和固定提交。
只检查 tracked 改动及数据准备目录的未跟踪文件；其他位置原有 proxy.md、codeagent_doc、
.venv 等未跟踪文件不需要删除，也不应单独触发整个任务停止。

~~~bash
export PREP_REPO=/data/pub1/z00919662/segmentation/nas8-mir-nus-clean-v3
export PREP_PIN=c8b3bd4b7585040950d4257046ca8d8a0958c521
export PREP_BRANCH=agent/nas8-mir-nus-clean-gt100-v3
test -d "$PREP_REPO/.git"
test "$(git -C "$PREP_REPO" remote get-url origin)" = https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
git -C "$PREP_REPO" diff --exit-code
git -C "$PREP_REPO" diff --cached --exit-code
test -z "$(git -C "$PREP_REPO" status --porcelain --untracked-files=all -- nas_data/scene8_v3)"
git -C "$PREP_REPO" fetch origin "refs/heads/$PREP_BRANCH:refs/remotes/origin/$PREP_BRANCH"
git -C "$PREP_REPO" merge-base --is-ancestor "$PREP_PIN" "refs/remotes/origin/$PREP_BRANCH"
test "$(git -C "$PREP_REPO" rev-parse HEAD)" = "$PREP_PIN"
git -C "$PREP_REPO" rev-parse HEAD
~~~

不在该 checkout 滚动升级到未经本指令核验的后续数据版本。
若实际源码有修改，保留并报告 diff --stat，要求用户上传改动源码/patch给ChatGPT同步。
不要stash/reset/git clean。若只是 checkout 位置或版本不符，可在新空目录从上述GitHub分支
clone --no-checkout，再 checkout --detach "$PREP_PIN"，修改PREP_REPO运行变量后继续；
不改旧工作区，不要求用户代写代码。

2. 使用已有 ultraface 环境做数据准备；它不安装 OpenCLIP。
服务器已报告 /home/z00919662/anaconda3 下有解释器；先验证 conda.sh，路径若不符仅查找已有安装。

~~~bash
test -f /home/z00919662/anaconda3/etc/profile.d/conda.sh
source /home/z00919662/anaconda3/etc/profile.d/conda.sh
conda activate ultraface
test "$CONDA_DEFAULT_ENV" = ultraface
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "$PREP_REPO/nas_data/scene8_v3"
python -c 'import sys,PIL; print(sys.version); print("Pillow",PIL.__version__)'
python -m unittest test_pipeline test_nus21
bash -n run_prepare.sh
~~~

38 项准备测试必须通过；不创建/更新 ultraface 环境，不使用 GPU。

3. 执行完整准备。所有原图、ZIP、旧标签、旧输出和训练任务保留。
复用已有 MIR 解压缓存与 NUS 根；不再次解压 archive.zip，不下载新数据。
旧 BLOCKED.json 不能删除，不向旧输出目录补写伪造的 PREPARED.json。

~~~bash
export OLD_LABEL_ROOT=/data/pub1/z00919662/dataset/UltraFaceSlim_8scene_multilabel_manifests_640x360_v1
export CACHE_ROOT=/data/pub1/z00919662/dataset/NAS8_new_sources_raw_v3
export MIR_IMAGES=/data/pub1/z00919662/dataset/mirflickr25k.zip
export MIR_ANN=/data/pub1/z00919662/dataset/mirflickr25k_annotations_v080.zip
if [[ ! -f "$MIR_IMAGES" ]]; then
  export MIR_IMAGES=/data/pub1/z00919662/segmentation/datasets/MIRFLICKR25K/downloads/mirflickr25k.zip
fi
test -f "$MIR_IMAGES"
test -f "$MIR_ANN"
test -d "$CACHE_ROOT/nus"
for split in train val test; do test -s "$OLD_LABEL_ROOT/$split.jsonl"; done
export NEW_LABEL_ROOT=/data/pub1/z00919662/dataset/NAS8_multilabel_clean_v3_rev3
if [[ -e "$NEW_LABEL_ROOT" ]]; then
  export NEW_LABEL_ROOT="$NEW_LABEL_ROOT"_$(date +%Y%m%d_%H%M%S)
fi
test ! -e "$NEW_LABEL_ROOT"
export PREP_LOG=/data/pub1/z00919662/dataset/nas8_clean_rev3_$(date +%Y%m%d_%H%M%S).log
df -h /data/pub1/z00919662/dataset
cd "$PREP_REPO/nas_data/scene8_v3"
bash run_prepare.sh \
  --old-root "$OLD_LABEL_ROOT" \
  --mir-images "$MIR_IMAGES" \
  --mir-annotations "$MIR_ANN" \
  --nus-root "$CACHE_ROOT/nus" \
  --cache-root "$CACHE_ROOT" \
  --output-root "$NEW_LABEL_ROOT" \
  --seed 20260910 2>&1 | tee "$PREP_LOG"
~~~

输出目录由脚本创建，不提前 mkdir NEW_LABEL_ROOT。任何失败保留新输出与日志，禁止自动清理后重试。
如果固定输入位置缺失，允许只读查找已有唯一文件并更改运行变量；多份无法区分则说明需用户提供哪一份。
六文件 NUS 指纹不一致或真实代码错误才停止同步。people_r1 relevant/potential 差集按Rev3审计，不恢复旧阻断。
不要执行旧 Rev2“诊断后等待类别映射”流程，不传 --nus-metadata-root，不补 Concepts81。

4. 核验完整产物，并直接传给 benchmark。

~~~bash
test ! -e "$NEW_LABEL_ROOT/BLOCKED.json"
for name in summary.json PREPARED.json train.jsonl val.jsonl test.jsonl val_strict.jsonl test_strict.jsonl source_audit.json resolution_audit.json review100_template.csv weather_review_template.csv gt100/index.html; do
  test -f "$NEW_LABEL_ROOT/$name"
done
python -c 'import json,os; from pathlib import Path; p=Path(os.environ["NEW_LABEL_ROOT"]); s=json.loads((p/"summary.json").read_text()); assert s["schema"]=="nas8_source_curation_v3_rev3",s.get("schema"); assert s["labels"]==["night","indoor","rain_snow","office","outdoor","landscape","sports","objective_image"]; assert s["status"]=="PREPARED_REVIEW_REQUIRED",s.get("status"); print("LABEL_ROOT_ACTUAL:",str(p)); print("DATA_STATUS:",s["status"]); print("GT_QUALITY_BLOCKERS:",s.get("training_quality_blockers",[]))'
export NAS8_LABEL_ROOT="$NEW_LABEL_ROOT"
git -C "$PREP_REPO" diff --exit-code
git -C "$PREP_REPO" diff --cached --exit-code
conda deactivate
~~~

同时核对 GT100 为100张独立图、10张contact、CSV/index与来源配额：
COCO15、Places36520、SEG13 15、MIR20、NUS20、10_scenes10。
记录 source_audit 中 NUS 六文件指纹、格式 retrieval_top21_verified、
雪正标注 database5227/query177 与实际可用图的差异、缺图数、split泄漏检查及分辨率建议。
旧manifest SHA由准备脚本复查；基于TRAIN的分辨率建议不改变 MobileCLIP2 的预训练256×256输入。

PREPARED_REVIEW_REQUIRED 是成功生成、待人工审核，不是 BLOCKED。
不要因 HUMAN_ACTION_REQUIRED=true / READY_FOR_TRAINING=false / 某类缺正负GT 而结束诊断任务。
保持 reviewed 字段原样；禁止自动填 reviewed=1、依据模型预测改GT，或因此启动训练。
将 GT100 和天气审核列为最终人工待办后，继续阶段 B。
NAS8_LABEL_ROOT 必须是本轮 NEW_LABEL_ROOT，不让用户手动告知已经由你生成的路径。

## 阶段 B：以下为完整 benchmark 执行步骤

本阶段使用新的 benchmark 分支和19项测试。进入新venv后恢复CPU线程上限2，
GPU重新通过nvidia-smi选取；数据阶段设置过 CUDA_VISIBLE_DEVICES=""，不能沿用空值。
先前GPU1空闲只是报告时状态，必须现场再次检查。
下面章节从1重新编号，所有命令保持同一shell或恢复记录的变量。
阶段 A 成功并通过上述门槛后，不再请求“是否继续”。



## 1. 任务与版本

必须采用最新八类方案，禁止运行旧九类 benchmark。

Repository: https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
Branch: agent/nas8-mobileclip2-rev3-resume-20260911
兼容代码 commit: 1609335dec728b8c62364fc7fbe4b0b4ecfa3bee
入口：nas_benchmark/scene8_mobileclip2/run.py
本 benchmark 分支用于评测；数据准备固定使用阶段 A 指定的 Rev3 提交，不能调用 benchmark 分支中较早的数据准备副本。
ChatGPT 已准备代码并推送，19 项离线单元/CLI 测试通过；尚未在目标 GPU 实测，必须先 smoke。

CodeAgent 只下载、检查、执行、回报，不自行写改源码、shell、配置、prompt，不训练、不蒸馏、不提交代码、不停止旧训练/watchdog。
发现代码问题由 ChatGPT 更新 GitHub 后再拉取。发现既有本地改动，保留并报告 diff，让用户同步给 ChatGPT。
不使用旧 nas_multilabel/ 脚本/requirements，不使用先前 nas9 兼容分支。

八类固定顺序：
night, indoor, rain_snow, office, outdoor, landscape, sports, objective_image
夜景、室内、雨/雪、办公场景、户外、自然风景、运动、客观图。

语义：
- 夜景不等于暗图，禁止根据亮度生成 GT。
- 雨/雪：任一确认存在为1；两者都确认不存在才为0；否则-1。NUS snow=0不等于无雨雪，SEG ice_or_snow不自动当可靠雪GT。
- 自然风景必须自然景观为主体，不含以城市、人物、动物或特写物体为主体的照片。
- 办公：office/home_office/office_cubicles/conference_room。显示器不自动等于办公。
- 客观图：用户定义的测试图/pattern/chart，不泛指所有CGI或带屏幕的照片。
- 室内/户外不通过互相取反补GT；体育活动/场馆沿用V3。
- 1正、0有依据负、-1未知；不把unknown变成0。

进入阶段 B 后，原图、标注、本轮 Rev3 manifest、旧标签和训练输出全部只读。不重下数据、不再重建标签或重分split、不另建旧九类probe。阶段 A 由现成 Rev3 脚本生成新的派生输出，是本任务必要步骤。
GT100含排除/审核样本，不用作正式test。

## 2. Proxy、SSL

公开版继承已有代理变量；私有下载版预置用户提供的代理。
不要打印密码，不set -x，不输出env/pip config debug/git remote -v，不把私有文件提交GitHub。

~~~bash
set +x
# 先配置用户授权的 http_proxy、https_proxy；不得打印凭据。
test -n "$http_proxy"
test -n "$https_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$https_proxy"
# 按用户要求，clone/fetch之前配置SSL bypass：
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=1
export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org"
export HF_HUB_DISABLE_XET=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
~~~

TLS bypass降低证书验证保护，本任务按用户要求使用。Git/pip/HF分别配置；HF在脚本中以 --insecure-downloads 开启当前进程requests TLS bypass。
Access Denied/403/组织策略/权限拦截时停止并报告，不更换未授权代理、镜像或凭据绕过限制。
不sudo、不替换系统驱动/CUDA/cuDNN、不创建跨cuDNN主版本so软链接。

## 3. 新建独立checkout

仍使用NAS原服务器路径，不从其他项目的服务器迁移信息推断本项目路径。根目录缺失/无权限时报告。

~~~bash
set -euo pipefail
umask 077
export NAS8_BASE=/data/pub1/z00919662/segmentation
test -d "$NAS8_BASE"
df -h "$NAS8_BASE"
export NAS8_RUN="$(mktemp -d "$NAS8_BASE/nas8_mobileclip2_zs_20260911_XXXXXX")"
export NAS8_REPO="$NAS8_RUN/repo"
export HF_HOME="$NAS8_RUN/hf_cache"
export TMPDIR="$NAS8_RUN/tmp"
mkdir -p "$HF_HOME" "$TMPDIR"
git clone --single-branch \
  --branch agent/nas8-mobileclip2-rev3-resume-20260911 \
  https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git "$NAS8_REPO"
cd "$NAS8_REPO"
git merge-base --is-ancestor 1609335dec728b8c62364fc7fbe4b0b4ecfa3bee HEAD
git rev-parse HEAD
git status --short
~~~

记录NAS8_RUN实际绝对路径。各步骤在同一shell；换shell须恢复变量，不要重新mktemp。
复用本任务专用checkout时核对origin/分支/干净状态后才能fetch + pull --ff-only。不在训练checkout切分支，不reset/stash/git clean/force push。
如检测到CodeAgent以前改过源码，保留并返回diff，要求用户同步给ChatGPT。

## 4. 使用本轮已完成的 Rev3 标签

必须沿用阶段 A 的实际输出路径，不再使用旧 BLOCKED 根目录。
若换了 shell，从阶段 A 的完成报告恢复准确的 NEW_LABEL_ROOT；不能按修改时间挑选目录。

~~~bash
test -n "$NEW_LABEL_ROOT"
export NAS8_LABEL_ROOT="$NEW_LABEL_ROOT"
test -f "$NAS8_LABEL_ROOT/summary.json"
test -f "$NAS8_LABEL_ROOT/PREPARED.json"
test ! -e "$NAS8_LABEL_ROOT/BLOCKED.json"
for name in train.jsonl val.jsonl test.jsonl val_strict.jsonl test_strict.jsonl; do
  test -f "$NAS8_LABEL_ROOT/$name"
done
~~~

Rev3 实际 schema 为 nas8_source_curation_v3_rev3。新 benchmark 明确兼容该版本与原 nas8_source_curation_v3；不修改 summary.json 伪装旧 schema。
原生/人工审核 strict 生成规则和八类顺序仍要逐项校验。
PREPARED_REVIEW_REQUIRED / HUMAN_ACTION_REQUIRED=true / training_quality_blockers 只表示审核待办，不阻止诊断评测。
实际 BLOCKED、缺文件、数据仍在写入、schema不支持、GT冲突、跨split重叠等才停止。

## 5. 独立Python环境

推荐Python3.10–3.12 + torch2.4.1 + torchvision0.19.1。
不能在Python3.8 Ultraface环境安装OpenCLIP3.2；不要修改其他任务在用环境。
已有专用于本任务、版本均满足且CUDA可用的环境可直接复用，否则找现有解释器创建新venv。
磁盘不足、无兼容Python或venv不可用则报告，不sudo、不反复conda create、不自动清理旧文件。

~~~bash
command -v python3.12 || true
command -v python3.11 || true
command -v python3.10 || true
# 用上述实际存在的绝对路径替换这一运行变量：
export NAS8_PYTHON=/home/z00919662/anaconda3/bin/python3.12
"$NAS8_PYTHON" -V
"$NAS8_PYTHON" -m venv "$NAS8_RUN/venv"
source "$NAS8_RUN/venv/bin/activate"
python -m pip install --no-cache-dir --index-url https://pypi.org/simple pip==25.2
python -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1
python -m pip install --no-cache-dir --index-url https://pypi.org/simple \
  torch==2.4.1 torchvision==0.19.1 \
  -r "$NAS8_REPO/nas_benchmark/scene8_mobileclip2/requirements.txt"
python -m pip check
~~~

固定OpenCLIP3.2.0、timm1.0.20、HFHub0.34.4，不安装Apple mobileclip，不升级torch2.8。
OpenCLIP3.2.0确实存在，内部镜像只列2.32不能据此说不存在。
torch2.4.1+cu121不应描述为“cuDNN8版”。wheel依赖装在用户环境，系统只有cuDNN8不能据此要求管理员升级。
不要--no-deps安装torch，不卸载wheel依赖的nvidia/cudnn包。失败输出脱敏原始错误，区分下载问题与真实运行兼容问题。

~~~bash
cd "$NAS8_REPO/nas_benchmark/scene8_mobileclip2"
python -m unittest discover -s . -p 'test_*.py' -v
python run.py --help
python run.py --label-root "$NAS8_LABEL_ROOT" \
  --output-dir "$NAS8_RUN/data_preflight" --split test --preflight-only
~~~

19项测试必须通过。preflight仅数据检查、不加载模型，不能当GPU smoke。
coverage.json显示strict/mapped各类正负未知数量，不含虚拟预测成绩。

## 6. GPU smoke

nvidia-smi选空闲卡，不固定占用0，不杀别人任务，没有可用卡则报告等待资源。
设置NAS8_GPU为现场确定的一张物理编号。
顺序解码、torch线程2，不启动大量worker。

~~~bash
nvidia-smi
export NAS8_GPU='<空闲GPU物理编号>'
export CUDA_VISIBLE_DEVICES="$NAS8_GPU"
python -c "import torch; print(torch.__version__,torch.version.cuda,torch.backends.cudnn.version()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
cd "$NAS8_REPO/nas_benchmark/scene8_mobileclip2"
python run.py --label-root "$NAS8_LABEL_ROOT" \
  --output-dir "$NAS8_RUN/smoke" --split test --limit 16 \
  --batch-size 4 --precision fp32 --warmup 2 --runs 5 \
  --insecure-downloads 2>&1 | tee "$NAS8_RUN/smoke.log"
~~~

模型唯一来源 https://huggingface.co/timm/MobileCLIP2-S0-OpenCLIP
文件open_clip_model.safetensors约300MB；脚本只下载此文件并记录snapshot revision及SHA256。
这是Apple MobileCLIP2-S0的OpenCLIP命名适配权重，不是V1-S0，不用Apple原始命名权重混充。
重参数化来自timm，smoke检查重参数化前后FP32输出、CUDA、256输入和有限八类得分。
smoke前16图只是执行检查，不保证八类覆盖，不当正式精度。

保持预训练默认256×256 preprocess，不改640×360。将来与UltraFace比较时使用同一test/strict清单，各模型预处理分别记录，不能声称输入分辨率相同。
八类是固定正负prompt独立sigmoid得分，不是八类softmax/argmax。

HF下载被阻止时说明人工操作：
下载页 https://huggingface.co/timm/MobileCLIP2-S0-OpenCLIP/blob/main/open_clip_model.safetensors
上传 NAS8_RUN/manual_weights/open_clip_model.safetensors（回报必须展开实际绝对路径）。
然后smoke/full都可增加 --checkpoint /实际路径/open_clip_model.safetensors 跳过模型下载。
手工文件记录SHA256并确认来源，不strict=False掩盖不匹配。

STATUS.json=SMOKE_PASS且无FAILED.json才继续。
CUDA/动态库/导入错误停止回报，不自动CPU fallback。OOM仅允许batch4→1并改新输出目录重跑，不改模型/源码。

## 7. 完整test评测

固定FP32、阈值0.5；不在val/test调阈值或prompt，不训练/蒸馏/伪标。
正式命令不得带--limit：

~~~bash
cd "$NAS8_REPO/nas_benchmark/scene8_mobileclip2"
python run.py --label-root "$NAS8_LABEL_ROOT" \
  --output-dir "$NAS8_RUN/test_fp32" --split test \
  --batch-size 16 --precision fp32 --warmup 50 --runs 200 \
  --insecure-downloads 2>&1 | tee "$NAS8_RUN/test_fp32.log"
~~~

全量指本版本test全部图片，不重新采样或遍历所有原始源图片。
同一得分同时评估strict和mapped。OOM可batch16→8→4→1，新输出目录重跑。
任何错误保留日志/FAILED.json，禁止自行patch。所有重跑使用新目录，不覆盖失败结果。

输出：
- REPORT.md、STATUS.json、metrics.json
- strict_per_class.csv、mapped_per_class.csv、各来源mapped CSV
- predictions.csv：image/source/group_id/split以及八类GT、strict_GT、score、pred
- data_audit.json、coverage.json、prompts.json
- environment.json：环境/GPU/输入/preprocess/commit/权重revision/SHA256
- latency.json：GPU encoder batch1时间、图像评分循环吞吐
- error_visualizations：strict/mapped每类最多8张FP与8张FN，完整原图+侧栏八类得分，index.json有路径/证据。

重点检查night误报、自然风景的人/动物主体误报、办公/客观图的显示器困难负样本、雨雪漏检。
区分弱GT争议与模型误判，不根据预测自行修改GT。

## 8. 指标与人工审核

strict优先（原生/人工/用户审核）；mapped含弱规则，仅诊断，不当人工金标准。
缺正例或负例的类标NO_GT/SINGLE_CLASS_GT：F1/AP/AUC/accuracy=N/A；有正例仍可单报recall，有负例可单报specificity。
covered宏平均必须列出覆盖类别，不能叫八类总体；八类全有正负GT才输出macro_f1_all8/macro_ap_all8。
数量覆盖也不等于样本量足够/GT完全可靠。绝不能把全unknown列报0错误或100%准确。

雨雪真实照片负例不足时继续完成可算指标，明确缺口，不能用客观图负例充数。
保留V3 training_quality_blockers，必要时请用户看原GT100和weather_review_template.csv，生成新的reviewed标签版本后再验收；本任务不填审核CSV。
旧UltraFace模型看过旧数据时不能把新split直接包装为其独立测试。预训练重叠不能排除。
GPU encoder FPS不是V516 NPU FPS，不宣称达到100fps。

~~~bash
cd "$NAS8_REPO"
git diff --exit-code
git diff --cached --exit-code
git status --short
git rev-parse HEAD
~~~

源码必须未修改。BENCHMARK_COMPLETE_DIAGNOSTIC表示执行完成，不代表业务精度达标。
人工GT审核、缺文件/解释器/权限/资源、下载受阻或需ChatGPT修代码 => HUMAN_ACTION_REQUIRED=YES，写清具体操作。
执行成功且无待办才NO，不要求用户安装系统cuDNN。

最终直接给用户以下字段，所有路径展开为真实绝对路径，八类表直接贴出：

STATUS:
HUMAN_ACTION_REQUIRED / EXACT_USER_ACTION:
REPO_URL / BRANCH / COMMIT:
SOURCE_CODE_MODIFIED / WORKTREE_CLEAN:
PYTHON_ENV / TORCH / TORCHVISION / OPEN_CLIP / TIMM:
GPU / CUDA_RUNTIME / CUDNN_LOADED:
LABEL_ROOT_ACTUAL / SCHEMA / TEST_MANIFEST_SHA256:
LABEL_ORDER / TEST_IMAGE_COUNT / EVALUATED_IMAGE_COUNT:
SMOKE / FULL_RUN:
STRICT_EACH_CLASS_POS_NEG_UNKNOWN_P_R_F1_AP:
MAPPED_EACH_CLASS_POS_NEG_UNKNOWN_P_R_F1_AP:
STRICT_COVERED_LABELS / MACRO_F1_COVERED / MACRO_F1_ALL8:
REAL_PHOTO_NIGHT_AND_WEATHER_COVERAGE:
GT_QUALITY_BLOCKERS:
PREPROCESS / PRECISION:
WEIGHTS_SOURCE / REVISION / SHA256:
GPU_ENCODER_MEAN_MS / P95 / FPS / SCOPE:
REPORT_PATH / PREDICTIONS_PATH / VISUALIZATIONS_PATH:
FAILURE_COMMAND_AND_TRACEBACK_IF_ANY:


最终报告还必须加入阶段 A：
PREP_STATUS / PREP_COMMIT / PREP_LOG / NEW_LABEL_ROOT / NUS_FORMAT；
GT100_INDEX_PATH / CONTACT_SHEETS_PATH / REVIEW100_TEMPLATE_PATH / WEATHER_REVIEW_TEMPLATE_PATH；
原图/旧标签/旧BLOCKED/旧训练是否未修改；数据准备38项测试及benchmark19项测试结果。
将需要用户看GT100、确认真实照片是否既无雨也无雪列为 EXACT_USER_ACTION，给出实际绝对路径。
诊断benchmark完成与人工审核待办可以同时成立；不要写成“全部失败”，也不要把诊断结果称为八类最终验收。
