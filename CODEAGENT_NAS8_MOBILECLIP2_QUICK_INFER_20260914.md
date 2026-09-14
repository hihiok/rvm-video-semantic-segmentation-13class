# CodeAgent：MobileCLIP2-S0 八类快速看图，新服务器版（2026-09-14）

## 本轮目标

立即从现有原图抽取32张，跑通 MobileCLIP2-S0，输出八类得分与可视化。
不等NAS8 V3数据，不运行数据准备，不读取summary/PREPARED/BLOCKED或GT清单，不跑正式benchmark。
不要执行仓库内旧 PREP_THEN、Rev3数据准备或旧九类指令；以本文件为准。

新服务器任务根：/mnt/ssd1/z00919662/NAS_scene_detection
现有图片数据根：/mnt/ssd1/z00919662/datasets/COCO_ADE_13cls_16x9_640x360
不沿用旧服务器 /data/pub1 的代码、标签或环境路径，不假定旧GPU编号仍空闲。

仓库：https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
分支：agent/nas8-mobileclip2-quick-infer-20260914
最低代码提交：78293facf1d5827c5bea5c5437bdd6906c816904
入口：nas_benchmark/scene8_mobileclip2/quick_infer.py
本指令：CODEAGENT_NAS8_MOBILECLIP2_QUICK_INFER_20260914.md

八类固定为：night、indoor、rain_snow、office、outdoor、landscape、sports、objective_image。
对应：夜景、室内、雨/雪、办公场景、户外、自然风景、运动、客观图。
雨或雪任一存在为正；自然风景要求自然景观为主体；夜景不等于暗图；
显示器不等于办公/客观图，客观图沿用测试图/pattern/chart的定义。
每类独立得分，允许多个类别同时命中，没有others类或八类softmax。
0.5阈值只供预览，得分不是校准后的概率，不根据这32图调整prompt或阈值。

ChatGPT已经编写并提交代码，3项离线测试通过；未在新服务器验证GPU。
CodeAgent只拉取、检查、执行、回报，不自行改源码/依赖文件/prompt。
如果发现自己此前改过代码，保留，报告diff摘要，并请用户把改动文件或patch同步给ChatGPT。
旧代码和数据不覆盖，不训练，不停止其他任务，不sudo。

## 1. Proxy 和 SSL（任何 clone/fetch 之前）

私有附件预置用户提供的代理；GitHub公开文件继承已有代理配置。
不要打印代理密码，不set -x，不输出完整env、git config --list或pip config debug。
私有MD和proxy.md不得提交GitHub。

~~~bash
set +x
# PRIVATE_PROXY_PRESET
test -n "$http_proxy"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$https_proxy"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=1
export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org"
export HF_HUB_DISABLE_XET=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
set -euo pipefail
umask 077
~~~

HF下载由命令中的 --insecure-downloads 对当前进程启用TLS bypass。
遇到明确403/组织访问阻断，报告实际资源与错误，不更换未授权代理绕过。
不修改系统CUDA/cuDNN，不做跨主版本so软链接，不要求先装系统cuDNN9。

## 2. 拉取本轮代码

任务根可已有其他checkout；本轮在其子目录中新建独立checkout，不在旧工作区切分支。
保持同一shell；恢复会话时使用记录的NAS8_RUN等实际变量，不重复mktemp。

~~~bash
export NAS8_ROOT=/mnt/ssd1/z00919662/NAS_scene_detection
export NAS8_DATA=/mnt/ssd1/z00919662/datasets/COCO_ADE_13cls_16x9_640x360
test -d "$NAS8_DATA"
mkdir -p "$NAS8_ROOT"
df -h "$NAS8_ROOT" "$NAS8_DATA"
export NAS8_RUN="$(mktemp -d "$NAS8_ROOT/mobileclip2_quick_20260914_XXXXXX")"
export NAS8_REPO="$NAS8_RUN/repo"
export HF_HOME="$NAS8_ROOT/hf_cache"
export TMPDIR="$NAS8_RUN/tmp"
mkdir -p "$HF_HOME" "$TMPDIR"
git clone --single-branch --branch agent/nas8-mobileclip2-quick-infer-20260914 \
  https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git "$NAS8_REPO"
git -C "$NAS8_REPO" merge-base --is-ancestor 78293facf1d5827c5bea5c5437bdd6906c816904 HEAD
git -C "$NAS8_REPO" rev-parse HEAD
git -C "$NAS8_REPO" status --short
~~~

若复用本轮专用且无源码修改的checkout，可fetch并pull --ff-only该分支。
不得reset/stash/git clean，也不因其他旧checkout有未跟踪proxy/环境文件而清理或阻塞。
记录NAS8_RUN、NAS8_REPO的实际绝对路径。

## 3. 确认原图目录，选择空闲GPU

只读检查目录结构，优先选择原图 images 子目录，或其 train/val 子目录。
不选 masks、annotations、labels；13类语义分割mask不用于本任务，也不转换成八类GT。

~~~bash
find "$NAS8_DATA" -maxdepth 3 -type d | head -60
export NAS8_IMAGES="$NAS8_DATA"
if [[ -d "$NAS8_DATA/images" ]]; then
  export NAS8_IMAGES="$NAS8_DATA/images"
fi
nvidia-smi
~~~

若原图实际在其他名字的目录，只把NAS8_IMAGES设为查到的真实目录；
这是运行参数，不算修改源码。脚本递归抽样并排除常见mask目录/文件名，
但文件名不能保证是原图，后面仍要查看抽样预览。

现场选一张空闲GPU，由CodeAgent填写运行变量，不让用户代填。
下面占位符必须换成nvidia-smi实际空闲编号；没有空闲卡则报告等待资源，不杀进程。

~~~bash
export NAS8_GPU='<CodeAgent现场选定的空闲GPU物理编号>'
export CUDA_VISIBLE_DEVICES="$NAS8_GPU"
~~~

## 4. 环境：优先复用兼容环境，否则独立venv

先只读检查当前/已有环境。可直接复用满足Python3.10–3.12、
OpenCLIP3.2.0、timm1.0.20、HFHub0.34.4且CUDA可用的专用环境；
已有torch2.4.1/torchvision0.19.1组合优先。不要修改其他任务在用环境。
不要再安装Apple mobileclip或旧 nas_multilabel/requirements。

若无兼容环境，用新服务器已有Python3.10–3.12建立本轮venv。
先找解释器，不能写死旧服务器anaconda位置：

~~~bash
command -v python3.12 || true
command -v python3.11 || true
command -v python3.10 || true
python -V || true
~~~

CodeAgent把下面变量换成实际可用的Python3.10–3.12绝对路径；
如果当前已有兼容环境，跳过创建/安装这块，直接进入GPU检查。
若PATH未列出，允许检查已有conda环境列表及本任务根中的环境，避免重复下载大依赖。
无兼容解释器或缺权限时再明确报告人工动作。

~~~bash
export NAS8_PYTHON='<新服务器实际已有的Python3.10–3.12绝对路径>'
"$NAS8_PYTHON" -m venv "$NAS8_RUN/venv"
source "$NAS8_RUN/venv/bin/activate"
python -m pip install --no-cache-dir --index-url https://pypi.org/simple pip==25.2
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1
python -m pip install --no-cache-dir --index-url https://pypi.org/simple \
  torch==2.4.1 torchvision==0.19.1 \
  -r "$NAS8_REPO/nas_benchmark/scene8_mobileclip2/requirements.txt"
python -m pip check
~~~

OpenCLIP3.2.0存在；内部镜像不列出不等于PyPI不存在。
wheel运行依赖装在用户环境；系统cuDNN8不等于只能运行cuDNN8。
实际CUDA/驱动兼容性以以下加载与真实推理为准，不预先宣称已解决。

~~~bash
python -c "import sys,torch,torchvision,open_clip,timm; print(sys.executable); print('torch',torch.__version__,'torchvision',torchvision.__version__,'timm',timm.__version__); print('CUDA',torch.version.cuda,'cuDNN',torch.backends.cudnn.version()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
cd "$NAS8_REPO/nas_benchmark/scene8_mobileclip2"
python -m unittest test_quick_infer -v
python quick_infer.py --help
~~~

这3项测试检查抽样排除mask、无标签输入流程、八类可视化输出，不等于GPU测试。
不需要先跑Rev3数据准备的38项测试，也不调用旧run.py。

## 5. 抽32张并确认是原图

~~~bash
python quick_infer.py \
  --input-dir "$NAS8_IMAGES" \
  --output-dir "$NAS8_RUN/input_check" \
  --count 32 --seed 20260914 --preflight-only \
  2>&1 | tee "$NAS8_RUN/input_check.log"
~~~

查看input_check/selected_images.json和selected_previews中的若干图，确认是自然彩色原图而非mask。
preflight只扫描文件名并解码抽到的图，不解码整个数据集。
出现P/L模式mask或损坏文件会报出路径；不要把这些转换RGB后冒充原图。
如选错目录，修正NAS8_IMAGES变量，使用新输出目录重新预检。
若真实可用图片少于32但仍有几十张，可以继续并报告实际数量；无图/非原图才停。
不需要summary.json、PREPARED.json、test.jsonl；其他目录存在BLOCKED.json也与本任务无关。

## 6. 直接跑32张真实GPU推理

~~~bash
cd "$NAS8_REPO/nas_benchmark/scene8_mobileclip2"
python quick_infer.py \
  --input-dir "$NAS8_IMAGES" \
  --output-dir "$NAS8_RUN/preview32" \
  --count 32 --seed 20260914 --batch-size 4 \
  --insecure-downloads 2>&1 | tee "$NAS8_RUN/preview32.log"
~~~

固定MobileCLIP2-S0、FP32、官方256×256预处理。
数据原图640×360保留；预训练resize/center-crop与原图长宽比不同，不强改为640×360模型输入。
可视化展示完整原图，旁边列出八类得分与0/1结果，不裁切原图或覆盖原文件。
模型加载会检查重参数化前后FP32特征一致、256输入、有限八类得分。
没有八类GT，不计算accuracy/F1/mAP，不把随机几十张的视觉效果当正式benchmark。

模型使用 timm/MobileCLIP2-S0-OpenCLIP 的 open_clip_model.safetensors，
首次下载到NAS8_ROOT/hf_cache，后续复用。不是旧MobileCLIP-S0。
下载及模型初始化可能比32张推理本身更久；不得仅因短期没逐图日志就判卡住。
OOM允许把batch4降为1，用新输出目录preview32_b1重跑，不自行改代码。
CUDA/导入/权重不匹配的真实错误保留FAILED.json和完整脱敏traceback，回报ChatGPT修复。
不自动切CPU后宣称GPU跑通。

若已在新服务器有来源确认的OpenCLIP权重，可增加 --checkpoint /真实绝对路径/open_clip_model.safetensors。
若HF下载确实被阻断，人工动作明确为：
从 https://huggingface.co/timm/MobileCLIP2-S0-OpenCLIP/blob/main/open_clip_model.safetensors 下载，
上传到本轮NAS8_RUN/manual_weights/open_clip_model.safetensors，然后用 --checkpoint 参数重跑；
回报需将NAS8_RUN展开成实际绝对路径。不使用未经确认的镜像或Apple原始键名权重。
手动checkpoint会记录SHA256，但需单独核实来源。

## 7. 完成检查与交付

成功必须有STATUS.json=QUICK_INFERENCE_COMPLETE、gpu_tested=true，无FAILED.json，
predictions.csv有实际图片数量的行，八类得分均已输出。仅input_check通过不能说模型已跑通。

~~~bash
git -C "$NAS8_REPO" diff --exit-code
git -C "$NAS8_REPO" diff --cached --exit-code
git -C "$NAS8_REPO" status --short
~~~

交付NAS8_RUN/preview32（或实际重跑目录）中的：
- index.html：逐图总览，可直接打开；需与visualizations/保持相对位置。
- visualizations/：每张原图+八类得分侧栏。
- contact_01.jpg等：每8张一页，32张共4张总览。
- predictions.csv：路径、八类score和pred；无GT。
- selected_images.json、prompts.json、environment.json、REPORT.md、STATUS.json。
- 完整运行日志；如失败加FAILED.json。

直接给用户展示几张实际contact图或代表性预测图，并给所有产物实际绝对路径。
不必让用户先做GT100审核；本轮没有必须填的标签表。
报告以下字段：

STATUS:
HUMAN_ACTION_REQUIRED / EXACT_USER_ACTION:
REPO_URL / BRANCH / COMMIT:
CODE_ROOT / OUTPUT_ROOT / INPUT_IMAGES_ROOT:
SOURCE_CODE_MODIFIED / TRACKED_WORKTREE_CLEAN:
PYTHON_EXECUTABLE / TORCH / TORCHVISION / OPEN_CLIP / TIMM:
GPU_PHYSICAL_ID / GPU_NAME / CUDA / CUDNN_LOADED:
MODEL / WEIGHTS_PATH / WEIGHTS_SHA256:
INPUT_RESOLUTION / PRECISION / SELECTED_COUNT / INFERRED_COUNT:
THREE_OFFLINE_TESTS / GPU_INFERENCE:
INDEX_HTML / CONTACT_IMAGES / PREDICTIONS_CSV / LOG:
QUICK_VISUAL_OBSERVATIONS:
FAILURE_COMMAND / TRACEBACK_IF_ANY:

成功且无实际待办时HUMAN_ACTION_REQUIRED=NO。说明只是快速看效果，不提供正式精度结论。
