# NAS8 Rev10：按用户指定规则生成雨雪弱负标签，排除运动图

环境 ultraface_new。只准备八类标签，不执行旧九类训练任务，不训练、不改原图、不改旧manifest、不重划分split。

## 网络和代码
在clone/fetch前配置代理与SSL。公共文件不含密码，http_proxy需由私有指令加载：
```bash
: "${http_proxy:?先加载私有代理配置}"
export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$http_proxy"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=1
```
仓库 https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
分支 agent/nas8-mir-nus-clean-gt100-v3
代码位置 `/mnt/ssd1/z00919662/NAS_scene_detection/nas8-mir-nus-clean-v3`。
核对origin、工作树干净，fetch对应分支并merge --ff-only。没有仓库时clone对应分支。记录HEAD。不得本地修改代码；若发现改动，保留diff请用户上传ChatGPT，不覆盖。

## 标签规则
- 输入用户已完成Rev9：`/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev9_20260916_145206`。
- night/indoor/office至少一个正标签、sports不为1、objective_image不为1、rain_snow为-1的图片，可按用户指定规则赋rain_snow=0。
- 所有sports=1图片排除于雨雪负监督，即使同时属于上述三类。已有sports图片的雨雪负监督设-1，保留sports正例。客观图雨雪负监督同样停用。
- 已有rain_snow=1绝不改。全部图片、其他七类标签、正标签和split保持。
- 按三类轮换选图、去重，每split的新增负例数量不超过该split正例数扣除已有负例的余额。不为凑数覆盖未知以外的标签。
- 证据标为weak_user_rule，不标manual/human_review。类别规则并不证明逐图无雨无雪；不能宣称已完成逐图审核。strict清单屏蔽弱规则负标签。
- 本次已获用户授权应用该类别规则，不要求先填600张审核CSV。不给reviewed/reviewer造值。

## 执行
加载已有conda初始化文件并进入仓库根目录：
```bash
conda activate ultraface_new
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -c 'import os; assert os.environ.get("CONDA_DEFAULT_ENV")=="ultraface_new"'
(cd nas_data/scene8_v3 && python -m unittest test_weather_negatives)
set -euo pipefail
umask 077
INPUT_ROOT=/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev9_20260916_145206
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT=/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev10_${RUN_STAMP}
python -u nas_data/scene8_v3/weather_negatives.py --input-root "$INPUT_ROOT" --output-root "$OUTPUT_ROOT" --user-scene-negative-rule --candidates-per-split 200 2>&1 | tee "/mnt/ssd1/z00919662/datasets/nas8_weather_rev10_${RUN_STAMP}.log"
```
新建同级输出，不预建目录，不覆盖Rev9。四项测试必须通过。

## 报告与审核
报告真实commit、input/output/log；各split雨雪正/弱负/逐图确认负/unknown数量；新增弱负例数；sports负监督数必须0；稀缺正例损失0；其他七类不变、split不变、输入哈希不变。
查看weather_weak_rule_changes.jsonl及weather_audit.json。weather_review网页仍展示余下未知候选，不是此次已赋0的图片；已赋0图片清单在weather_weak_rule_changes.jsonl，标签在新manifest。旧GT100仍为历史版本，不声称已刷新。
STATUS: PREPARED_REVIEW_REQUIRED，NEW_TRAINING_STARTED: NO。本轮标签转换无需人工逐图操作。严格评估仍需可信天气标注，报告该质量缺口，不以弱负例冒称解决。用户后续若发现具体错例，记录路径交回纠正。
若失败保存BLOCKED和异常，停止，不自行patch。
