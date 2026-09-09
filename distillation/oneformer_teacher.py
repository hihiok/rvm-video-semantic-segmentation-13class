"""OneFormer soft semantic output and the audited ADE20K-to-RVM13 projection."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F


def load_mapping(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    names = payload.get("target_class_names")
    if payload.get("source_num_classes") != 150 or not isinstance(names, list) or len(names) != 13:
        raise ValueError("Mapping must describe ADE20K-150 to exactly 13 target classes")
    assignment = [-1] * 150
    for target_name, source_ids in payload["target_to_source_ids"].items():
        if target_name not in names:
            raise ValueError(f"Unknown target class in mapping: {target_name}")
        target_id = names.index(target_name)
        for source_id in source_ids:
            if not 0 <= int(source_id) < 150 or assignment[int(source_id)] != -1:
                raise ValueError(f"Invalid or duplicate ADE20K source id: {source_id}")
            assignment[int(source_id)] = target_id
    if payload.get("unmapped_source_policy") != "background":
        raise ValueError("Only the explicit unmapped-to-background policy is supported")
    assignment = [0 if item < 0 else item for item in assignment]
    payload["source_to_target_ids"] = assignment
    return payload


class OneFormer13ClassTeacher:
    def __init__(self, model_name, mapping_path, device="cuda:0", amp=True):
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

        self.mapping = load_mapping(mapping_path)
        if model_name != self.mapping["teacher_model"]:
            raise ValueError(f"Mapping is pinned to {self.mapping['teacher_model']}, received {model_name}")
        self.device = torch.device(device)
        self.amp = bool(amp and self.device.type == "cuda")
        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name).to(self.device).eval()
        projection = torch.zeros(13, 150, dtype=torch.float32)
        for source_id, target_id in enumerate(self.mapping["source_to_target_ids"]):
            projection[target_id, source_id] = 1.0
        self.projection = projection.to(self.device)

    @torch.inference_mode()
    def __call__(self, images, output_size=(45, 80)):
        images = list(images)
        if not images:
            raise ValueError("At least one teacher image is required")
        inputs = self.processor(
            images=images, task_inputs=["semantic"] * len(images), return_tensors="pt"
        )
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.autocast(device_type="cuda", enabled=self.amp):
            outputs = self.model(**inputs)
            class_logits = outputs.class_queries_logits
            mask_logits = outputs.masks_queries_logits
            if class_logits.shape[-1] != 151:
                raise RuntimeError(f"Expected ADE20K 150 classes plus null, got {class_logits.shape[-1]}")
            class_probability = class_logits.softmax(dim=-1)[..., :-1]
            mask_probability = mask_logits.sigmoid()
            semantic = torch.einsum("bqc,bqhw->bchw", class_probability, mask_probability)
            mapped = torch.einsum("kc,bchw->bkhw", self.projection, semantic)

            # Undo processor batch padding before compressing each map. This is
            # the soft-probability equivalent of HF semantic post-processing.
            pixel_mask = inputs.get("pixel_mask")
            processor_size = inputs["pixel_values"].shape[-2:]
            mapped = F.interpolate(mapped, size=processor_size, mode="bilinear", align_corners=False)
            resized = []
            for index, image in enumerate(images):
                item = mapped[index : index + 1]
                if pixel_mask is not None:
                    valid = pixel_mask[index].bool()
                    rows = valid.any(dim=1).nonzero(as_tuple=False)
                    columns = valid.any(dim=0).nonzero(as_tuple=False)
                    if not len(rows) or not len(columns):
                        raise RuntimeError("OneFormer processor returned an empty pixel mask")
                    item = item[:, :, rows[0, 0] : rows[-1, 0] + 1, columns[0, 0] : columns[-1, 0] + 1]
                item = F.interpolate(item, size=(image.height, image.width), mode="bilinear", align_corners=False)
                item = F.interpolate(item, size=output_size, mode="area")
                item = item.clamp_min(0)
                item /= item.sum(dim=1, keepdim=True).clamp_min(1e-8)
                resized.append(item[0])
        return torch.stack(resized).float().cpu()
