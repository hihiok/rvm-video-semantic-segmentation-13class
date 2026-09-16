# CodeAgent：最新版 NAS8 Rev10，双 GPU 训练至测试及 ONNX 完成

## 1. 任务与明确授权

用户已批准开始训练，并接受 Rev10 的类别规则雨雪弱负例。不要再因 PREPARED_REVIEW_REQUIRED、READY_FOR_TRAINING=False 或缺少逐图天气审核而停止本轮训练；不要修改这些数据文件的状态，也不要把弱标签改成人工GT。训练入口通过显式 --accept-weak-weather 记录该授权。

本轮是八类 UltraFace-slim 任务，不执行旧九类/224×224文档。只使用最新版：
`/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev10_20260916_152810`

工作root：`/mnt/ssd1/z00919662/NAS_scene_detection`
环境：已有 `ultraface_new`。不新建环境、不更换现有 PyTorch/CUDA。
不修改原图/标签，不重做采样，不重划分split，不删除正样本，不干预其他训练/watchdog。

## 2. 代理、SSL 与 GitHub

首次 clone/fetch 之前配置代理及跳过 SSL。GitHub 公共文档不含密码；私有下载版提供完整 http_proxy 配置。不要把凭据输出到日志或提交GitHub。
```bash
set +x
: "${http_proxy:?先按私有指令配置代理}"
export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$http_proxy"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=1
```

Repository：https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
Branch：`agent/nas8-mir-nus-clean-gt100-v3`
建议checkout：`/mnt/ssd1/z00919662/NAS_scene_detection/nas8-rev10-training`

若不存在：clone --branch 该分支到上述新路径。若已存在：核对origin和工作区干净，再fetch该分支，checkout并merge --ff-only对应origin分支。记录完整HEAD，确保存在本MD及 `ultraface_scene_multilabel8/train_rev10.py`。
禁止本地patch Python/config。若发现本地代码改动，保留diff及文件，让用户上传ChatGPT同步，不覆盖或reset，不自行修代码。

## 3. 环境、依赖、测试

找到已有conda.sh后source，激活环境。以下操作在仓库根目录进行：
```bash
conda activate ultraface_new
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -c 'import os,sys,torch,cv2,numpy; assert os.environ.get("CONDA_DEFAULT_ENV")=="ultraface_new"; print(sys.version); print(torch.__version__,torch.version.cuda); print(torch.cuda.is_available())'
python -c 'import onnx,onnxruntime; print(onnx.__version__,onnxruntime.__version__)'
```
若仅缺 onnx/onnxruntime，可在当前环境安装缺失项（不要重装/升级现有torch、CUDA、numpy、opencv）：Python3.8可使用 onnx==1.16.2、onnxruntime==1.19.2。按缺少的包分别安装，安装前核查pip依赖计划；若要求更改核心环境则停止报告。公司代理下 pip 使用 --trusted-host pypi.org --trusted-host files.pythonhosted.org。依赖完整时不执行安装。

```bash
(cd ultraface_scene_multilabel8 && NAS8_RUN_DDP_TEST=1 python -m unittest test_rev10_train)
bash -n ultraface_scene_multilabel8/run_rev10.sh
```
要求6项全部通过，不跳过分布式测试。开发侧5项本地测试通过（含ONNX数值检查、未知标签零梯度、分片损失等价、泄漏检查），单进程CPU smoke与续训通过；当前开发环境禁止Gloo socket，真实双进程测试留给服务器执行，不能声称开发侧GPU测试已通过。

## 4. GPU与速度设置

查看 nvidia-smi GPU列表及compute-apps进程，优先选择两张无其他训练进程、显存基本空闲的GPU；尽量相同型号。不得抢占或终止他人任务。只有一张空闲时允许单GPU启动；没有空闲则报告资源阻塞。

网络继续使用原始 UltraFace Mb_Tiny（base_channel=16）+ GAP + Dropout + Linear(256,8)，不加检测头，不换模型。
训练：双进程 DDP/NCCL、AMP、默认全局batch128（双卡每卡64）、workers每卡2。SGD lr=.01，momentum=.9，weight_decay=.0001；60 epochs，35/50降学习率。沿用masked BCE与现有正例权重计算，unknown=-1不参与loss。双卡loss按全局已知标签数归一化，避免不同卡标签密度造成偏差。

输入分辨率读取最终TRAIN清单中的已审计尺寸：中位长边<=500选择320×180，否则384×216，保持16:9。报告每来源尺寸及候选放大比例。此为速度优先的首个基线，不宣称已通过分辨率精度对照。模型、校准、测试、导出共享这个尺寸，不能调用旧固定640×360导出入口。

## 5. 一路执行 smoke → 60 epochs → 校准 → 测试 → ONNX

设置GPU_IDS为刚核实空闲的物理GPU编号（以下0,1为示例，必须替换为实际选择），RUN_ROOT用新时间戳。可在已有tmux会话运行以防断连，持续监控本任务到完成，不仅启动就结束报告。

```bash
export GPU_IDS=0,1
export DATA_ROOT=/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev10_20260916_152810
export RUN_ROOT="/mnt/ssd1/z00919662/NAS_scene_detection/runs/nas8_rev10_$(date +%Y%m%d_%H%M%S)"
export BATCH=128 WORKERS=2
bash ultraface_scene_multilabel8/run_rev10.sh
```

脚本先跑1个完整训练epoch的smoke（验证最多10批），smoke通过后自动从头正式训练60epochs，不需要再问用户。smoke仅验证运行链路，不当最终模型。

smoke若明确CUDA OOM，允许只通过环境变量 BATCH 改为64、再32，在全新RUN_ROOT重试，记录实际全局batch，不改源码和分辨率。若NCCL在单机遇到IB设备问题，可设置 NCCL_IB_DISABLE=1重试新smoke；其它通信错误保留日志，不能掩盖失败。正式训练出现数据/代码问题需保留checkpoint并报告；不要跳过坏图或改标签。

输出每epoch耗时及剩余时间估计。前3个正式epoch后按实测报告ETA，不预先承诺双卡线性加速。DDP每epoch末尾可能为对齐rank补不足world_size张记录，这是临时采样，不写回manifest、不造成跨split泄漏。

## 6. 恢复

若本任务意外中断，原RUN_ROOT、GPU_IDS、BATCH、WORKERS保持一致，从自身last_train_state续训；不要加载旧9类或旧数据模型冒充新训练。
```bash
export RESUME="$RUN_ROOT/train/last_train_state.pth"
bash ultraface_scene_multilabel8/run_rev10.sh
```
恢复模型/优化器/调度器/AMP状态和epoch；输入哈希、分辨率、batch等必须一致。续训不承诺随机增强逐位复现。

## 7. 评估、导出和完成报告

每epoch用val主清单macro-AP选best；训练结束仅在val校准8类阈值，然后对独立test评估。test不得参与选checkpoint/阈值。

- 主清单指标包含采样和弱标签，必须如实标注；不能与旧自然分布指标直接比较。
- 严格指标仅使用manual/human_review/user_review证据；weak_user_rule排除。某类缺正例或负例时F1、AP、precision、balanced accuracy等需要双侧支持的指标为null；单侧recall/specificity按可用支持报告。宏平均给出实际支持类别名单。
- 雨雪严格负例仍为0时，不得声称测得可信雨雪误检率。用户已授权训练，该评估限制不阻止完成训练。

最终应有：
`train/best_train_state.pth`、`last_train_state.pth`、`best_ultraface_scene8.pth`、`thresholds.json`、`test_per_class_calibrated.csv`、`test_summary.json`、val/test严格指标、预测npz（含图片路径）、`model.onnx`、`model.json`、`onnx_check.json`、`COMPLETE.json`。

ONNX checker及ONNX Runtime对PyTorch随机输入+实际测试图片数值对齐必须通过；失败保留权重并报告，不重训。

报告完整commit、环境/GPU、数据根/输出根、分辨率与依据、batch/epochs、实际总耗时、best epoch、八类指标和宏指标、严格指标支持范围、阈值、checkpoint及ONNX绝对路径、数据哈希未变。
STATUS: TRAINING_COMPLETE 仅在训练、test与ONNX全部成功后输出。
HUMAN_ACTION_REQUIRED: NO（本轮用户已授权弱标签训练，不以逐图审核再次阻断）。真正无法继续的依赖/数据/代码/资源问题才YES，写清人工动作。
