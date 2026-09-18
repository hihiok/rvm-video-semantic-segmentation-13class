# CodeAgent：修复 Three-Votes JSON 类型比较并原地安全续跑

本任务由用户已授权的全量三方标注延续而来。不要让用户在“删缓存 / 本地改Python / 等待”之间重新选择。GitHub已提供修复；只拉取修复代码，复用原结果继续全量标注。只可视化200张，不停止在200张。本轮不训练。

## 固定路径与状态

- Repository: https://github.com/hihiok/rvm-video-semantic-segmentation-13class.git
- Branch: `agent/nas8-mir-nus-clean-gt100-v3`
- 旧执行 commit：`0701108c148583c7e800666b205b41fa717394a6`
- 修复 commit：下载交付MD中固定的 `FIX_COMMIT`；GitHub公开版按branch HEAD锁定并检查本MD和修复文件存在。
- REPO：`/mnt/ssd1/z00919662/NAS_scene_detection/three_votes_code_20260918_095522`
- VOTE_ROOT：`/mnt/ssd1/z00919662/datasets/NAS8_three_votes_v1_20260918_095522`
- VOTE_INPUT：`/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev10_20260916_152810`
- MOBILE_BASE：`/mnt/ssd1/z00919662/NAS_scene_detection/mobileclip2_quick_20260914_jEp8AV`
- MOBILE_PY：MOBILE_BASE 下 `venv/bin/python`。
- QWEN_PY：复用上次已成功完成preview200的Qwen venv和.pth配置，不重建、不升级包。预期目录名 `qwen3vl_teacher_env_20260918_095522`；若实际不同，从上次运行记录找回精确路径，不要猜测后直接新建。
- 原失败主PID用户报告为89693。**只读确认该PID对应进程已退出，不能因PID复用而杀掉无关进程。** 同时确认没有仍在写 mobile/qwen SQLite 的本任务进程；如有仍在工作，等其退出后更新，不并发重启。

已完成：全量prepare（175966行）、Qwen下载与锁定、preview200。不得删除 mobile/、qwen_*/、qwen_model.json、plan.json、items.jsonl 或 preview200，不得重新下载图片/已缓存模型。

## 修复内容

1. stage_lock 对新metadata做JSON规范化，使tuple/list在持久化后的语义一致。
2. 精确白名单只接受GitHub旧0701108代码哈希到此次修复的代码哈希。MobileCLIP/Qwen推理代码、提示词、投票定义没有修改；只有续跑比较与已完成merge兼容检查改变。
3. 新旧模型哈希、包版本、选择清单、参数和提示词必须相同。旧metadata与SQLite记录完全保留，继续使用原run_digest，另写 resume_compatibility_audit.json 记录当前执行代码。
4. 旧preview200完整文件哈希通过后直接复用，不重新渲染或改写。之后继续剩余175766张（若已有更多完成行，按SQLite实际计数跳过）。
5. 任何真实模型/提示词/输入/非白名单代码差异仍然BLOCKED，不使用 --force，不清空SQLite。

## 1. 网络设置必须先于fetch

以下公开文档从已有私有env.sh读取代理；交付私有MD包含明确代理配置。禁止把带口令的MD/env.sh提交GitHub或放入结果ZIP。

```bash
set -euo pipefail
set +x
export MOBILE_BASE=/mnt/ssd1/z00919662/NAS_scene_detection/mobileclip2_quick_20260914_jEp8AV
source "$MOBILE_BASE/env.sh" >/dev/null
export http_proxy="${http_proxy:-${HTTP_PROXY:-}}"
: "${http_proxy:?Existing proxy configuration is missing}"
export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
git config --global http.proxy "$http_proxy"
git config --global https.proxy "$http_proxy"
git config --global http.sslVerify false
export GIT_SSL_NO_VERIFY=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export VOTE_REPO=/mnt/ssd1/z00919662/NAS_scene_detection/three_votes_code_20260918_095522
export VOTE_ROOT=/mnt/ssd1/z00919662/datasets/NAS8_three_votes_v1_20260918_095522
export VOTE_INPUT=/mnt/ssd1/z00919662/datasets/NAS8_multilabel_weather_rev10_20260916_152810
export MOBILE_PY="$MOBILE_BASE/venv/bin/python"
cd "$VOTE_REPO"
git status --short
```

若仓库有本地Python/config修改：保存 `git diff --binary` 和修改后的对应文件到私有诊断目录，向用户提供路径并请其上传ChatGPT；**不要reset/stash覆盖，不允许本地patch续跑**。用户报告的.pth环境修复及HF缓存目录修复不等于仓库Python修改，已跑通的环境/缓存保留即可。

working tree干净且本任务旧进程全部停止后：

```bash
test -z "$(git status --porcelain)"
git fetch origin agent/nas8-mir-nus-clean-gt100-v3
export FIX_COMMIT="$(git rev-parse origin/agent/nas8-mir-nus-clean-gt100-v3)"
git merge-base --is-ancestor 0701108c148583c7e800666b205b41fa717394a6 "$FIX_COMMIT"
git checkout --detach "$FIX_COMMIT"
test -f nas_data/scene8_three_votes/resume_compatibility.json
test -f nas_data/scene8_three_votes/test_resume.py
test -z "$(git status --porcelain)"
"$MOBILE_PY" -m unittest discover -s nas_data/scene8_three_votes -p 'test_*.py' -v
bash -n nas_data/scene8_three_votes/run_all.sh
```

预期16项离线测试通过，包括原11项和5项续跑回归：真实PAIRS/LEGACY_PROMPTS序列化往返、旧SQLite不改写、旧预览复用、真实配置变化仍拒绝、非白名单代码变化仍拒绝。

## 2. 恢复实际环境和GPU配置

- 从上次任务记录确认 QWEN_PY、HF_HOME、checkpoint路径，必须复用，不改metadata里的路径，不把cached权重移动到新目录。
- 读取 `qwen_00/metadata.json` 的 `shards`，沿用相同数量；通常2。不能由双分片改为单分片。GPU物理编号可变化。
- `nvidia-smi` 选两张真正空闲GPU，不直接沿用env.sh旧GPU2，不杀别人的进程。
- Qwen包版本与上次相同；不要pip install/uninstall，不改.pth，不重新建venv。

```bash
export QWEN_PY=/mnt/ssd1/z00919662/NAS_scene_detection/qwen3vl_teacher_env_20260918_095522/bin/python
# 上行必须先与已成功运行的实际venv核对；路径不一致时只改此环境变量。
test -x "$QWEN_PY"
"$QWEN_PY" -c 'import sys,torch,transformers; print(sys.executable,torch.__version__,transformers.__version__)'
nvidia-smi
# 按nvidia-smi和原shards选择；此处2,3仅示例，必须换成实际空闲物理编号。
export VOTE_GPUS=2,3
export VOTE_LOG="/mnt/ssd1/z00919662/NAS_scene_detection/three_votes_resume_$(date +%Y%m%d_%H%M%S).log"
nohup bash "$VOTE_REPO/nas_data/scene8_three_votes/run_all.sh" > "$VOTE_LOG" 2>&1 &
export VOTE_PID=$!
printf 'PID=%s\nROOT=%s\nLOG=%s\nCOMMIT=%s\n' "$VOTE_PID" "$VOTE_ROOT" "$VOTE_LOG" "$FIX_COMMIT"
```

无需换VOTE_ROOT、删除Mobile阶段或重做prepare。正常日志顺序应包含：
- `PREPARE_REUSED 175966`
- `QWEN_SNAPSHOT_LOCKED`（锁定本地模型复用；可能重新核验权重哈希）
- `MOBILE_REUSED` / `QWEN_REUSED`
- `MERGE_REUSED .../preview200`
- 随后全量Mobile/Qwen剩余记录开始增长。

每个复用旧metadata的阶段会新增 `resume_compatibility_audit.json`，其中 `metadata_and_existing_rows_rewritten=false`。若仍出现真正的不一致，保留现状，返回脱敏错误以及对应metadata/compatibility audit；不修改这些文件来强行通过。

## 3. 进度与交付

**先确认本次不再因为metadata比较失败退出，并且SQLite完成行数持续增长。** 不能只提交nohup就报告完成。返回已有 `preview200/index.html` / `preview200.zip` 路径让用户看效果；后台继续全量，不等待审核。模型标注时间按实际吞吐更新ETA，不能把训练4小时当成标注耗时。

全量完成后检查：
- train/val/test共175966条都有三方记录；解析失败保持显式unknown并报告数量，不能补0。
- 完整 `merged/COMPLETE.json`、train/val/test、votes/conflicts/summary、200张可视化、review200_final.zip。
- 旧图/旧标签/原split不变；已完成200张Mobile/Qwen结果复用；旧正例图片记录未删除。
- 新合并结果仍为伪标签建议，不宣称已获人工GT；本轮不训练。

报告运行commit、环境路径、实际GPU、旧记录复用数/新增数、各阶段进度、预览/最终ZIP路径、是否有任何本地代码修改。运行中 `STATUS: RUNNING_FULL_ANNOTATION`；完成才 `STATUS: ANNOTATION_COMPLETE_REVIEW_REQUIRED`。只有环境/数据/模型等真实阻塞才请求用户处理；正常恢复无需用户再次批准。
