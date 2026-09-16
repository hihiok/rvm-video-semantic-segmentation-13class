# NAS8 Rev9：真实照片雨雪负例审核与应用

只处理八类标签，不执行旧九类训练文档。使用 ultraface_new；不训练、不下载数据、不改变图片/split、不修改本地代码。若发现本地代码改动，保留diff交给用户上传ChatGPT，不覆盖。

输入固定 `/mnt/ssd1/z00919662/datasets/NAS8_multilabel_clean_v3_rev8_20260916_121032`，输出全新同级 Rev9 目录，原Rev8保持只读。

## 网络和代码
先按私有会话配置代理，保持 http_proxy/https_proxy/HTTP_PROXY/HTTPS_PROXY 一致。公共文档不存密码。clone/fetch之前执行：
```bash
: "${http_proxy:?请先加载私有代理配置}"
export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$http_proxy"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=1
```
仓库 https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
分支 agent/nas8-mir-nus-clean-gt100-v3
使用 `/mnt/ssd1/z00919662/NAS_scene_detection/nas8-mir-nus-clean-v3` 现有checkout，先核对origin及工作区干净，再fetch并ff-only更新。没有checkout时clone该分支到此目录。记录commit。禁止自行patch。

## 规则
1. 保留全部图片、所有正标签、其他七类标签及原split。仅修改rain_snow监督。
2. objective_image=1且rain_snow=0时，把雨雪负监督改为-1，本轮完全停用这类合成图雨雪负例；不改objective_image正例。
3. 从rain_snow=-1、objective_image!=1、night/indoor/office/sports至少一类为1的照片中，轮流按四类选候选，每split最多200张，去除重复。候选类别不能直接证明无雨雪。
4. 人工逐图确认无雨 AND 无雪，才填 reviewed=1、rain_absent=1、snow_absent=1、reviewer；不明确或有雨雪则保持未审核，不把未知自动填0。
5. 通过图片SHA256和split匹配应用审核，禁止覆盖现有雨雪正标签。候选网页只供审核，未审核候选不增加负监督。
6. 不重新配平其他类别；此次任务不恢复Rev8之前被移除的图片。保留原GT100作历史参考，新版天气标签以本轮网页与manifest为准，不能将旧GT100冒充已更新。

## 第一步：生成Rev9和审核包
加载已有conda初始化文件，进入仓库根目录：
```bash
conda activate ultraface_new
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -c 'import os; assert os.environ.get("CONDA_DEFAULT_ENV")=="ultraface_new"'
(cd nas_data/scene8_v3 && python -m unittest test_weather_negatives)
set -euo pipefail
umask 077
INPUT_ROOT=/mnt/ssd1/z00919662/datasets/NAS8_multilabel_clean_v3_rev8_20260916_121032
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT=/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev9_${RUN_STAMP}
python -u nas_data/scene8_v3/weather_negatives.py --input-root "$INPUT_ROOT" --output-root "$OUTPUT_ROOT" --candidates-per-split 200 2>&1 | tee "/mnt/ssd1/z00919662/datasets/nas8_weather_rev9_${RUN_STAMP}.log"
```
不要预建OUTPUT_ROOT。完成后给用户输出绝对路径：weather_review/index.html、weather_review_template.csv。网页图片是缩略图，有疑问查看CSV中原图；不能仅凭类别名称或昏暗缩略图审核。

这一阶段雨雪真实负例仍可能为0，属于预期，不得声称缺口已解决。HUMAN_ACTION_REQUIRED: YES，明确需要逐图审核。

## 第二步：应用用户审核
用户填好的CSV保存为独立文件（保留原模板），记录路径。以第一步输出目录为输入，再写全新同级目录：
```bash
# 用报告中的真实路径设置，不保留示例占位符
REVIEW_INPUT_ROOT="第一步OUTPUT_ROOT绝对路径"
REVIEWED_CSV="用户填写的CSV绝对路径"
FINAL_ROOT="/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_reviewed_$(date +%Y%m%d_%H%M%S)"
python -u nas_data/scene8_v3/weather_negatives.py --input-root "$REVIEW_INPUT_ROOT" --output-root "$FINAL_ROOT" --reviews "$REVIEWED_CSV" --candidates-per-split 200
```
不需要写代码。若有现成审核CSV，仅在其字段、图片指纹和split全部验证通过时使用。禁止伪造reviewer或批量勾选reviewed。新生成的模板用于后续审核，不覆盖已提交审核。

## 验收报告
输出实际commit、环境、输入/输出/日志；各split雨雪正/负/unknown、撤销的合成负标签数、确认应用的真实负例数、候选数及来源；所有图片和正例保留，其他七类标签不变，split不变，输入六文件哈希不变。
产物包括train/val/test JSONL+CSV、strict子集、weather_audit.json、weather_removed_synthetic.jsonl、weather_applied_reviews.jsonl、source_audit.json、summary.json、PREPARED.json、weather_review网页及CSV。
状态PREPARED_REVIEW_REQUIRED；READY_FOR_TRAINING=False。即使应用审核也不自动训练，需要确认审核质量、数量和覆盖范围。失败保留BLOCKED.json并回报异常，不自行改代码。
