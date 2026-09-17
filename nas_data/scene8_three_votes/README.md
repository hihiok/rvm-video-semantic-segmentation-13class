# NAS8 full three-voter annotation with 200 review images

Scope: every record in the approved Rev10 train/val/test manifests. Exactly 200 selected rows are rendered; all remaining rows are still annotated. No retraining, source deletion, split changes, or original GT edits. The pipeline is not the stale 9-label training task.

## Voters and prompts

1. Existing label contributes one vote: 1 present / 0 absent / -1 abstain. Existing weak weather rules remain visibly tagged in original_evidence.
2. MobileCLIP2-S0 uses the user's proven OpenCLIP 3.2.0 + timm 1.0.20 environment and cached safetensors. Each class has three paired positive/negative descriptions. It does not apply softmax across the eight classes. A vote requires at least two prompt-pair cosine margins >= +0.02, no opposite margin <= -0.02, and mean >= +0.02; symmetric for negative. Everything else abstains. This is a conservative initial heuristic, **not calibrated confidence**. Legacy prompt scores/decisions are recorded for comparison, not counted as a fourth voter.
3. Qwen3-VL-8B-Instruct sees only the image and frozen class definitions, never previous labels/filename/source or MobileCLIP output. It returns ten structured labels: eight task labels plus separate rain and snow. `rain_snow` is derived with OR and unknown handling. Short visual evidence is saved, not invented numeric probabilities.

Prompt emphasis: natural landscape may coexist with outdoor; small people/buildings do not automatically negate scenery, but streets and background grass do not establish landscape. Sports covers recognizable exercise/events and dedicated sports venues, including running/cycling/swimming/skiing/surfing and marked courts. Ordinary commuting, balls alone and grass alone are not sufficient. Snow on mountains counts as visible snow. Night/indoor/office do not prove weather absence. See prompts.py for complete definitions.

## Fusion

- At least two agreeing votes produce a majority proposal. One vote only / tie / all unknown => -1.
- Majority proposals and all original votes are retained even when the conservative final proposal abstains.
- An original positive opposed by both teachers is marked for review and final -1, never silently changed to 0. No image record is removed. This safeguards scarce classes while acknowledging label noise.
- Conflicts with explicit human_review/user_review are likewise held for review. Source manual annotations are still one vote, not assumed infallible.
- If both fused indoor/outdoor are positive, both become -1 and a review flag is added. Annotation is allowed to abstain; the product's “at least one prediction” fallback must not fabricate training labels.
- No other hard exclusions. No automatic outdoor from landscape, indoor from office, or weather negative from night.
- Agreement fraction is an audit number, **not a soft probability target**. This version does not distill vocabulary logits or claim generated scores are calibrated.

## Stages

`prepare.py` fingerprints all source manifests and images; keeps row/split identities; orders 200 source-stratified review rows first with extra landscape/sports attention. Reference quotas: coco15, Places60, SEG13 20, MIR35, NUS35, 10_scenes35. Short source pools are reported and filled from other sources, no invented rows. Review selection is biased for diagnosis, not an accuracy test.

`mobile.py`, `qwen.py` use transactional SQLite per worker. Interrupted runs reuse completed rows only when image/model/prompt/code fingerprints match. Never run two writers on the same shard. GPU IDs can change on resume; shard count must not change. Qwen uses FP16/eager/batch1, max image area 512² on V100; no FlashAttention2/BF16 requirement. Fine weather details still need original-image review. Neither teacher has been GPU-tested in the ChatGPT workspace.

`run_all.sh`: download/lock Qwen once, annotate first200 with both teachers, make preview200/index.html, **continue automatically**, annotate remaining MobileCLIP and sharded Qwen, merge full output. Two GPUs run independent Qwen shards; this is offline inference, not DDP training. Progress logs print measured ETA. Full Qwen annotation can be much slower than MobileCLIP; do not promise the 4-hour training runtime.

`merge.py`: write proposed train/val/test.jsonl, votes.jsonl, conflicts.jsonl, summary.json, review200, review200_template.csv and review ZIP. All original validation/test and strict manifests are copied unchanged under original_evaluation/. Proposed val/test labels are pseudo-labels and cannot establish independent accuracy. Do not automatically train or declare pseudo-labels human GT. The final status is ANNOTATION_COMPLETE_REVIEW_REQUIRED.

Tests: `python -m unittest discover -s nas_data/scene8_three_votes -p test_votes.py -v`. Offline tests include a 204-image full workflow proving preview is limited to200 but all204 receive labels, immutable inputs, resume checks, majority/tie handling, weather OR, scarce-positive protection, class coexistence, and JSON validation.

Official model references:
- https://huggingface.co/timm/MobileCLIP2-S0-OpenCLIP
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen3_vl
