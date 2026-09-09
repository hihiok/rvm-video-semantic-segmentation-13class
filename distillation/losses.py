"""Guarded soft-label distillation losses for a mismatched teacher/student pair."""

import torch
from torch.nn import functional as F


def confidence_gated_kd_loss(
    student_logits,
    teacher_probabilities,
    ground_truth,
    *,
    temperature=2.0,
    confidence_threshold=0.60,
    background_weight=0.25,
    disagreement_weight=0.25,
    unsupported_target_ids=(9,),
    ignore_index=255,
):
    """Pixel KL with confidence, GT-disagreement and unsupported-class guards.

    Inputs may be clips ([B,T,C,H,W]) or image batches ([B,C,H,W]). Teacher
    probabilities may be uint8; they are normalized after dequantization.
    """
    if student_logits.ndim == 5:
        if teacher_probabilities.ndim != 5 or ground_truth.ndim != 4:
            raise ValueError("Video KD expects logits/probabilities [B,T,C,H,W] and GT [B,T,H,W]")
        student_logits = student_logits.flatten(0, 1)
        teacher_probabilities = teacher_probabilities.flatten(0, 1)
        ground_truth = ground_truth.flatten(0, 1)
    if student_logits.ndim != 4 or teacher_probabilities.ndim != 4 or ground_truth.ndim != 3:
        raise ValueError("KD expects image tensors [N,C,H,W] and ground truth [N,H,W]")
    if student_logits.shape != teacher_probabilities.shape:
        raise ValueError(f"Student/teacher shape mismatch: {student_logits.shape}, {teacher_probabilities.shape}")
    if (student_logits.shape[0], *student_logits.shape[-2:]) != ground_truth.shape:
        raise ValueError(f"Student/ground-truth shape mismatch: {student_logits.shape}, {ground_truth.shape}")
    if temperature <= 0 or not 0 <= confidence_threshold <= 1:
        raise ValueError("temperature must be positive and confidence threshold must be in [0,1]")
    if background_weight < 0 or disagreement_weight < 0:
        raise ValueError("KD pixel weights cannot be negative")

    teacher = teacher_probabilities.to(device=student_logits.device, dtype=torch.float32)
    if teacher_probabilities.dtype == torch.uint8:
        teacher = teacher / 255.0
    teacher = teacher.clamp_min(0)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    confidence, teacher_label = teacher.max(dim=1)
    valid = ground_truth.ne(ignore_index) & confidence.ge(confidence_threshold)
    for class_id in unsupported_target_ids:
        valid &= ground_truth.ne(int(class_id))

    weights = torch.ones_like(confidence)
    weights = torch.where(teacher_label.eq(0), weights * background_weight, weights)
    weights = torch.where(teacher_label.ne(ground_truth), weights * disagreement_weight, weights)
    weights = weights * valid
    selected_weight = weights.sum()
    selected_pixels = int(valid.sum().detach().item())
    if selected_weight.detach().item() <= 0:
        zero = student_logits.sum() * 0.0
        return {"loss": zero, "pixels": 0, "weight": 0.0, "agreement": 0.0}

    softened_teacher = teacher.clamp_min(1e-8).pow(1.0 / temperature)
    softened_teacher /= softened_teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    student_log_probability = (student_logits / temperature).log_softmax(dim=1)
    pixel_kl = F.kl_div(student_log_probability, softened_teacher, reduction="none").sum(dim=1)
    loss = (pixel_kl * weights).sum() / selected_weight * (temperature ** 2)
    agreement = (teacher_label.eq(ground_truth) & valid).sum().float() / valid.sum().clamp_min(1)
    return {
        "loss": loss,
        "pixels": selected_pixels,
        "weight": float(selected_weight.detach().item()),
        "agreement": float(agreement.detach().item()),
    }
