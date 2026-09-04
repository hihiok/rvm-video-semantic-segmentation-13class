# FSD 8-label scene classification at 640x360

Target labels: night, indoor, rain_snow, office, outdoor, landscape, sports, objective_image.

This variant preserves the existing FSD/UltraFace factory `create_Mb_Tiny_RFB_fd_3_scene_noRFB(2, 8)` and existing YUV scene transforms, but explicitly resizes scene input to `[height=360, width=640]` rather than relying on the original UltraFace scalar input-size API.

Source datasets are read-only. New multi-label manifests use values `1/0/-1` where `-1` is unknown and excluded from loss and metrics.