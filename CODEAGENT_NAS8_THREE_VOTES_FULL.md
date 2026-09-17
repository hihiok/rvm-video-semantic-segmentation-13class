# CodeAgent：NAS8 全量三方标注 + 200 张可视化

用户已授权全量标注。不是只标200张：当前 Rev10 train/val/test 的每一行都要由旧标签、MobileCLIP2-S0、Qwen3-VL-8B-Instruct 三方给出结果。只可视化200张，先出预览后自动继续，不等待人工批准全量运行。本轮不训练、不改旧GT/原图/旧checkpoint，不执行历史九类训练文档。

## 1. 固定输入、仓库和环境

- Repository: https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
- Branch: `agent/nas8-mir-nus-clean-gt100-v3`
- 必须包含的基线 commit: `f9611f2fecc8b2d64378deb587dd3b7098316a3b`。以本次新 clone 的 branch HEAD 为执行 commit 并立即锁定；交付版有更精确的 pin 时优先用交付版。
- 工作根：`/mnt/ssd1/z00919662/NAS_scene_detection`
- INPUT_ROOT：`/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev10_20260916_152810`
- 新结果建议：`/mnt/ssd1/z00919662/datasets/NAS8_three_votes_v1_<timestamp>`
- MobileCLIP 现有环境：`/mnt/ssd1/z00919662/NAS_scene_detection/mobileclip2_quick_20260914_jEp8AV/venv`
- MobileCLIP env.sh：同目录下 `env.sh`，包含代理、HF_HOME及之前GPU配置。不要输出其内容。
- 参考旧代码：同目录 `repo/nas_benchmark/scene8_mobileclip2/quick_infer.py`，分支 `agent/nas8-mobileclip2-quick-infer-20260914`，用户报告 commit `f1cc914`。
- 本次代码：`nas_data/scene8_three_votes/`，无需改旧 quick_infer.py。
- 旧训练环境 `ultraface_new` 和旧 checkout 中 rev10_data.py 的 NAS8_INPUT_WH 本地 patch 保留不动；本轮用新 clone，不能 reset/覆盖本地 patch。

## 2. 网络设置在 clone/fetch 前完成

使用用户已有 env.sh，不在仓库写入代理口令，不把 env.sh、令牌或带凭据日志放入ZIP。

```bash
set -euo pipefail
set +x
export MOBILE_BASE=/mnt/ssd1/z00919662/NAS_scene_detection/mobileclip2_quick_20260914_jEp8AV
source "$MOBILE_BASE/env.sh" >/dev/null
# env.sh 中的大写或小写代理都可；HTTPS 代理使用同一 HTTP CONNECT 地址。
export http_proxy="${http_proxy:-${HTTP_PROXY:-}}"
: "${http_proxy:?Missing proxy in the existing env.sh; report the missing file/config without printing credentials}"
export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
export GIT_SSL_NO_VERIFY=1
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$http_proxy"
git config --global http.sslVerify false
export NAS8_TASK_ROOT=/mnt/ssd1/z00919662/NAS_scene_detection
export VOTE_STAMP="$(date +%Y%m%d_%H%M%S)"
export VOTE_REPO="$NAS8_TASK_ROOT/three_votes_code_$VOTE_STAMP"
git clone --single-branch --branch agent/nas8-mir-nus-clean-gt100-v3 \
  https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git "$VOTE_REPO"
cd "$VOTE_REPO"
git merge-base --is-ancestor f9611f2fecc8b2d64378deb587dd3b7098316a3b HEAD
export VOTE_COMMIT="$(git rev-parse HEAD)"
git checkout --detach "$VOTE_COMMIT"
test -z "$(git status --porcelain)"
test -f nas_data/scene8_three_votes/run_all.sh
```

若 env.sh 缺失/代理失败：停止并报确切缺少配置，不猜代理。不需要重新下载图片数据集。

## 3. 环境：复用 MobileCLIP，单独建 Qwen venv

用户实际成功环境：Python3.11.7 / torch2.4.1+cu118 / torchvision0.19.1+cu118 / open_clip3.2.0 / timm1.0.20 / huggingface_hub0.34.4。先导入检查。不要把“驱动显示CUDA11.3”改写为“cu118绝对不能用”；用户已有实测成功。不要安装cu121或升级驱动。

```bash
export MOBILE_PY="$MOBILE_BASE/venv/bin/python"
"$MOBILE_PY" -c 'import sys,torch,torchvision,open_clip,timm; print(sys.version,torch.__version__,torchvision.__version__,timm.__version__); assert torch.cuda.is_available()'
"$MOBILE_PY" -m unittest discover -s nas_data/scene8_three_votes -p test_votes.py -v
bash -n nas_data/scene8_three_votes/run_all.sh
export QWEN_ENV="$NAS8_TASK_ROOT/qwen3vl_teacher_env_$VOTE_STAMP"
"$MOBILE_PY" -m venv --system-site-packages "$QWEN_ENV"
export QWEN_PY="$QWEN_ENV/bin/python"
"$QWEN_PY" -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
  -r nas_data/scene8_three_votes/requirements_qwen.txt
"$QWEN_PY" -c 'import sys,torch,torchvision,transformers; from transformers import Qwen3VLForConditionalGeneration,AutoProcessor; print(sys.prefix,torch.__version__,torchvision.__version__,transformers.__version__); assert torch.cuda.is_available()'
```

Qwen venv 使用系统可见依赖复用已有 torch/torchvision，新增包只安装进新 venv；不要在 MobileCLIP venv 或 ultraface_new 执行 pip install/uninstall。记录两个环境的 `pip freeze` 到新任务日志目录并检查 MobileCLIP 原有版本未变。若依赖导入失败，报告实际包冲突；不改本地Python代码、不偷偷改旧环境。

Qwen 默认 FP16、eager attention、每卡batch1、最大图像面积512×512；避免V100的BF16/FlashAttention2兼容问题。MobileCLIP保持FP32官方256×256预处理。这里的尺寸用于teacher，不修改学生网络320×180。

## 4. 选空闲GPU并持续全量执行

先 `nvidia-smi` 查看显存与进程，选择一到两张真正空闲的32GB V100；优先两张。env.sh 原 GPU2 只是历史值，不得直接抢占。下面 `2,3` 必须替换为检查后实际空闲编号；单卡可设一个编号，已开始后 shard数量不能改变，物理GPU编号可在暂停后更换。不得终止他人进程。

```bash
nvidia-smi
export VOTE_GPUS=2,3
export VOTE_INPUT=/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev10_20260916_152810
export VOTE_ROOT="/mnt/ssd1/z00919662/datasets/NAS8_three_votes_v1_$VOTE_STAMP"
export VOTE_LOG="$NAS8_TASK_ROOT/three_votes_$VOTE_STAMP.log"
# 运行前检查磁盘：Qwen权重约数十GB，加上两个环境、全量JSON审计/SQLite需额外空间。
# 不下载图片；MobileCLIP权重只读取现有HF缓存。
nohup bash "$VOTE_REPO/nas_data/scene8_three_votes/run_all.sh" > "$VOTE_LOG" 2>&1 &
export VOTE_PID=$!
printf 'PID=%s\nOUTPUT=%s\nLOG=%s\nCOMMIT=%s\n' "$VOTE_PID" "$VOTE_ROOT" "$VOTE_LOG" "$VOTE_COMMIT"
```

CodeAgent 必须保存以上无密钥路径、环境、PID及GPU编号到自己的任务记录，持续观察运行状态，不能提交任务后就谎报完成。长任务允许先报告 RUNNING + 已生成预览 + 实测吞吐/ETA，并明确后台还在标注。不要因为已出现 preview200/index.html 就退出/终止全量作业。

自动步骤：
1. 校验所有 train/val/test 记录与 split/group/path；给全量图片计算SHA256。固定200张预览名单。
2. 复用MobileCLIP权重；Qwen首次下载官方权重，锁定HF revision与全部权重哈希。
3. 前200张 MobileCLIP + 两卡Qwen → `preview200/index.html`、`preview200.zip`。这是早期反馈，无人工暂停门槛。
4. 自动继续所有剩余 MobileCLIP 和 Qwen，SQLite逐行提交；两个Qwen worker按固定分片分工。
5. 全量三方合并 → `merged/`，只保留200张可视化。

**断点续跑：**使用同一个VOTE_ROOT、同一个checkout/commit、同一个QWEN_ENV、同一shards数量，确认没有旧进程仍在写SQLite后重跑同一个run_all.sh。已完成行复用；模型/提示词/图片/源清单/代码不一致会拒绝续跑。不要为了“修复”删除SQLite或把新旧模型输出混用。初次prepare若中断在plan生成前，保留半成品并选择新VOTE_ROOT重新开始。

## 5. 下载失败、环境失败怎么办

MobileCLIP来源： https://huggingface.co/timm/MobileCLIP2-S0-OpenCLIP ，需要已有 `open_clip_model.safetensors`。优先复用env.sh设置的HF_HOME缓存。缓存找不到时先在现有 quick-infer 输出 environment.json 中核对checkpoint路径，可使用 mobile.py 的 `--checkpoint` 参数做诊断；不要换模型。

Qwen来源： https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct ，下载完整safetensors分片及index、config、tokenizer、processor和chat template；不是GGUF/AWQ。自动下载失败才请求用户人工下载到例如 `$NAS8_TASK_ROOT/models/Qwen3-VL-8B-Instruct`。

人工文件齐全后，先执行以下命令锁定，之后run_all.sh会复用锁定路径：

```bash
"$QWEN_PY" "$VOTE_REPO/nas_data/scene8_three_votes/qwen.py" \
  --output-root "$VOTE_ROOT" --download-only \
  --model-path "$NAS8_TASK_ROOT/models/Qwen3-VL-8B-Instruct"
```

只有真正失败才报告 `HUMAN_ACTION_REQUIRED: YES` + 缺失模型/文件 + 官方链接 + 目标路径 + 已脱敏错误。不得上传env.sh或代理信息。Qwen解析错误会保存raw输出并弃权，连续5条解析失败时停止供诊断；不把解析失败改成8个0。GPU OOM/依赖失败报告实际错误，不做未经GitHub同步的本地patch。

## 6. 三方合并及提示词验收

- 每类旧GT、MobileCLIP新提示词、Qwen各一票；-1弃权。至少两票同意形成proposal，票不足/平票则unknown。
- MobileCLIP旧提示词也计算，作为漏检对照；不算第四票。每类独立，无八类softmax。新提示词的三组一致性只决定该模型的一票。
- 自然风景与outdoor允许共存，不因outdoor分数更高而丢landscape。sports补充具体活动/场地描述；但模型是否改善必须从图中核实，不能仅凭旧GT宣称提升。
- 旧正例被两模型否定：保留旧标签和majority proposal，最终建议暂置-1并标记审核。没有删除任何正例图片记录；特别统计雨雪、办公、自然风景、运动、客观图的1→-1待审核数。
- Qwen blind prompt 不包含旧GT、MobileCLIP输出、路径或数据源名。雨和雪单独判断；任一1→rain_snow1，两者0才0，其他-1。
- indoor/outdoor双正冲突置unknown待审。标注阶段不执行“至少一个label”产品推理规则，不凭空补正例。
- 输出三方票数/原始得分/视觉依据，不把投票比例、CLIP相似度或Qwen自述当成校准概率，不蒸馏词表logits。
- 全量train/val/test都生成新伪标签清单；旧val/test和strict在original_evaluation中原样保留。以后不得用teacher生成的val/test标签宣称独立人工GT准确率。

200张参考配额：coco15、Places365 60、SEG13 20、MIR35、NUS35、10_scenes35。每源优先轮换自然风景、运动和少样本正例、街景/户外非风景对照；不足配额如实补其他来源。所有全量记录都标，配额只控制可视化，不控制参与标注的图片。

## 7. 最终产物和报告

输出根下：`plan.json` / `items.jsonl` / `qwen_model.json` / `mobile/results.sqlite` / `qwen_00/results.sqlite` / `qwen_01/results.sqlite` / `logs/` / `preview200/`。

最终 `merged/`：
- `train.jsonl`, `val.jsonl`, `test.jsonl`：全量三方标签建议，每行保留original_labels/original_evidence/split/group。
- `votes.jsonl`：三方原始票、Qwen依据、MobileCLIP新旧提示词得分、每类合并原因。
- `conflicts.jsonl`：分歧及需人工处理条目。
- `summary.json`, `COMPLETE.json`：全量计数、每类1/0/-1、原标签到新标签变化、解析失败、输入不变检查。
- `index.html`, `review200/`（200 JPEG）, `contact_01..20.jpg`, `review200_template.csv`。
- `original_evaluation/`：原val/test与strict，不被teacher覆盖。
- 根目录 `review200_final.zip`：只打包审核网页、200卡片、总览、模板、摘要，不包含全量图片/模型/代理。

最终必须报告：执行commit、环境版本、GPU、原清单全量行数/各split行数、Mobile/Qwen完成行数、每类三方覆盖率及分歧、每类1/0/-1与旧正例待审数、Mobile旧新提示词landscape/sports差异、Qwen解析失败数、200张实际来源配额、文件路径、实测用时；**不把旧GT一致率写成准确率**。

全量完成才可 `STATUS: ANNOTATION_COMPLETE_REVIEW_REQUIRED`。仅预览完成则 `STATUS: RUNNING_FULL_ANNOTATION`。记录 `OLD_IMAGES_LABELS_CHANGED: NO`、`SPLITS_CHANGED: NO`、`NEW_TRAINING_STARTED: NO`。人工后续动作是查看200图及冲突，不影响本轮已授权全量标注继续执行。
