# CodeAgent：NAS8 已完成测试结果可视化，每类随机10张正确、10张错误

## 1. 任务范围
使用本次已完成训练保存的test预测，生成GT和预测分类对照。只做可视化，不重新训练，不重跑全量推理，不改标签，不重新校准阈值，不调用旧九类代码。

环境：已有 `ultraface_new`（Python3.8.20）。此脚本只需 numpy、Pillow，不需要GPU，不更新torch/CUDA或其他环境。
工作root：`/mnt/ssd1/z00919662/NAS_scene_detection`
训练结果：`/mnt/ssd1/z00919662/NAS_scene_detection/runs/nas8_rev10_20260916_173205/train`
标签根：`/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev10_20260916_152810`

## 2. 代理、SSL与代码同步
clone/fetch前加载私有代理配置。公共GitHub文档不含凭据。禁止将代理凭据打印到日志或提交仓库。
```bash
set +x
: "${http_proxy:?请先加载私有代理配置}"
export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$http_proxy"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=1
```
Repository： https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
Branch：`agent/nas8-mir-nus-clean-gt100-v3`
建议使用已有 `/mnt/ssd1/z00919662/NAS_scene_detection/nas8-rev10-training` checkout。
核对origin及working tree。干净时fetch该分支、checkout、merge --ff-only对应origin；没有checkout才clone该分支。记录完整HEAD。
不得本地改Python/config。若发现CodeAgent此前有本地修改，保留diff和文件，让用户上传ChatGPT同步；不覆盖或reset。

## 3. 抽样与展示规则
八类：night、indoor、rain_snow、office、outdoor、landscape、sports、objective_image。
每类分别从正确和错误候选池均匀随机抽10张，固定seed=20260917；不按置信度挑选，不另加人为比例。

- 预测：保存的sigmoid probability >=该类 thresholds.json 阈值，判为1。
- 正确池：TP（GT1预测1）或TN（GT0预测0）。
- 错误池：FP（GT0预测1，误报）或FN（GT1预测0，漏报）。
- GT=-1显示问号，不判对错，不进入该类别任一抽样池。
- 每组不重复；不同类别可以选中同一图。160是最多展示条目数，不保证160张不同图片。
- 错例或正确例不足10时展示全部并报告缺口，不重复凑数、不借用train/val。例如客观图可能只有约7个错误，以实际结果为准。
- “正确”只针对当前抽样类别，同一图片其他类别仍可能预测错误。
- 使用主清单标签，弱/规则证据明确标注WEAK/RULE；人工证据标HUMAN/MANUAL。雨雪弱负样本下的错误表示与弱规则标签不一致，不自动证明模型判断错。

每张卡展示原图、来源、当前类别及TP/TN/FP/FN、GT正类别列表、预测正类别列表、八类完整GT/预测/概率/阈值/结果/证据等级。HTML可展开完整原始证据和路径。右上角是按320×180比例展示的输入预览，不表示重新运行推理。
脚本先校验config中的八类顺序和test manifest哈希；以完整图片路径关联NPZ和标签，要求覆盖一致且逐图GT一致。发现错配/缺文件立即停止，不猜对应关系。

## 4. 执行
加载已有conda初始化文件，进入上述GitHub checkout根目录：
```bash
conda activate ultraface_new
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -c 'import os,numpy,PIL; assert os.environ.get("CONDA_DEFAULT_ENV")=="ultraface_new"; print(numpy.__version__,PIL.__version__)'
(cd ultraface_scene_multilabel8 && python -m unittest test_visualize_test_rev10)
```
四项测试必须通过，包含路径反序关联、GT错配拒绝、未知标签排除、固定随机种子、数量不足处理，以及完整160条目渲染和ZIP输出。

```bash
set -euo pipefail
umask 077
RUN_DIR=/mnt/ssd1/z00919662/NAS_scene_detection/runs/nas8_rev10_20260916_173205/train
DATA_ROOT=/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev10_20260916_152810
VIS_STAMP=$(date +%Y%m%d_%H%M%S)
VIS_ROOT="/mnt/ssd1/z00919662/NAS_scene_detection/runs/nas8_rev10_20260916_173205/test_gt_pred_vis_${VIS_STAMP}"
python -u ultraface_scene_multilabel8/visualize_test_rev10.py --run-dir "$RUN_DIR" --data-root "$DATA_ROOT" --output-dir "$VIS_ROOT" --per-group 10 --seed 20260917 2>&1 | tee "${VIS_ROOT}.log"
```
不要预建VIS_ROOT；必须新目录，脚本拒绝覆盖已有目录或同名ZIP。无需用户逐图填写CSV。若生成失败保留BLOCKED.json和日志，不自行修代码。

## 5. 产物与报告
总览：`VIS_ROOT/index.html`。
每类：`VIS_ROOT/<label>/correct/index.html` 和 `error/index.html`，独立JPEG卡片、各分组contact.jpg（非空时）。最多160张独立卡片、16张contact图。
明细：selected_cases.csv、selected_cases.jsonl、summary.json。
打包：`VIS_ROOT.zip`，包含完整网页、图片和明细。下载解压后打开index.html，无需服务器或原图目录即可查看；ZIP不包含模型和整个数据集。

完成后检查各类两组抽样数量、TP/TN/FP/FN总池和抽取数、缺口、总展示条目与唯一图片数。至少打开一张正确和一张错误卡片核查布局、八类标签和源图片一致；不把模拟测试图片当真实结果。
输出完整commit、输入/输出/ZIP/日志绝对路径，以及八类抽样计数表。不要用这个人为等量抽样的展示集重新计算或替代完整test精度。
STATUS: VISUALIZATION_COMPLETE
OLD_IMAGES_LABELS_CHANGED: NO
MODEL_OR_THRESHOLDS_CHANGED: NO
NEW_TRAINING_STARTED: NO
HUMAN_ACTION_REQUIRED: NO（任务可自动完成，用户之后查看可视化即可）。
