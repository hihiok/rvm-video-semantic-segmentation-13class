# FSD scene 640x360 input adaptation

The user reference training script uses `--input_size 240`, calls `define_img_size(240)`, and the FSD configuration resolves that to a 240x320 scene tensor. That scalar API follows the original UltraFace 4:3 resolution convention and must not be used to imply 640x360.

For the 16:9 scene-classification task, the prepared trainer uses:

- FSD factory bootstrap: `define_img_size(640)` only to initialize the existing FSD module/config in the supported UltraFace resolution family.
- Actual scene transform size: explicit `[360, 640]` passed to `YUV444TrainAugmentation_scene` / `YUV444TestTransform_scene`.
- Mandatory runtime assertion: transformed tensor must be `[1, 360, 640]` per sample.
- Mandatory factory smoke: `create_Mb_Tiny_RFB_fd_3_scene_noRFB(2, 8)` must accept `[1, 1, 360, 640]` and return `[1, 8]` before any dataset training is allowed.

If the existing FSD scene head is spatially hard-coded and this smoke fails, CodeAgent must stop and report. It must not locally alter the FSD source code.