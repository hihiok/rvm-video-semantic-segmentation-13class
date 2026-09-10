# CodeAgent：MobileCLIP2-S0 最新 NAS 八类 zero-shot V3（完整执行版）

## 1. 任务与版本

必须采用最新八类方案，禁止运行旧九类 benchmark。

Repository: https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
Branch: agent/nas8-mobileclip2-zeroshot-v3-20260910
兼容代码 commit: 6d0b5f28e25a15741e8d1ac221e4068074ea2443
入口：nas_benchmark/scene8_mobileclip2/run.py
本分支基于 agent/nas8-mir-nus-clean-gt100-v3，保留 V3 数据政策不变。
ChatGPT 已准备代码并推送，16 项离线单元/CLI 测试通过；尚未在目标 GPU 实测，必须先 smoke。

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

原图、标注、V3 manifest、旧标签和训练输出全部只读。不重下数据、不重建标签、不重分split、不另建旧九类probe。
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
export NAS8_RUN="$(mktemp -d "$NAS8_BASE/nas8_mobileclip2_zs_20260910_XXXXXX")"
export NAS8_REPO="$NAS8_RUN/repo"
export HF_HOME="$NAS8_RUN/hf_cache"
export TMPDIR="$NAS8_RUN/tmp"
mkdir -p "$HF_HOME" "$TMPDIR"
git clone --single-branch \
  --branch agent/nas8-mobileclip2-zeroshot-v3-20260910 \
  https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git "$NAS8_REPO"
cd "$NAS8_REPO"
git merge-base --is-ancestor 6d0b5f28e25a15741e8d1ac221e4068074ea2443 HEAD
git rev-parse HEAD
git status --short
~~~

记录NAS8_RUN实际绝对路径。各步骤在同一shell；换shell须恢复变量，不要重新mktemp。
复用本任务专用checkout时核对origin/分支/干净状态后才能fetch + pull --ff-only。不在训练checkout切分支，不reset/stash/git clean/force push。
如检测到CodeAgent以前改过源码，保留并返回diff，要求用户同步给ChatGPT。

## 4. 使用最新V3标签

默认根目录：
/data/pub1/z00919662/dataset/NAS8_multilabel_clean_v3

如果人工审核后已生成用户确认的新版 NAS8_multilabel_clean_v3_reviewed 或其他实际输出目录，只更改NAS8_LABEL_ROOT为那个完整版本。
多个候选无法判定时请求确认，不仅按修改时间猜；不能混用两个版本文件。

~~~bash
export NAS8_LABEL_ROOT=/data/pub1/z00919662/dataset/NAS8_multilabel_clean_v3
test -f "$NAS8_LABEL_ROOT/summary.json"
test -f "$NAS8_LABEL_ROOT/PREPARED.json"
test ! -f "$NAS8_LABEL_ROOT/BLOCKED.json"
for name in train.jsonl val.jsonl test.jsonl test_strict.jsonl; do
  test -f "$NAS8_LABEL_ROOT/$name"
done
~~~

summary.schema必须nas8_source_curation_v3，labels顺序必须一致。
script检查train/val/test的group_id和绝对路径交集，但正式只推理test。
test_strict可为空/缺类；不能补GT。缺文件、数据仍在生成、有BLOCKED则停止并报告，先完成八类数据准备，不回退旧方案。
V3 PREPARED_REVIEW_REQUIRED允许做诊断评测，不代表数据已人工验收。

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
export NAS8_PYTHON='<实际已有Python3.10–3.12绝对路径>'
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

16项测试必须通过。preflight仅数据检查、不加载模型，不能当GPU smoke。
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
