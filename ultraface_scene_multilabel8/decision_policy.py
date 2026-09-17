"""Validation-only threshold selection and conservative context policy.
Scores are model outputs, not calibrated probabilities of being correct.
"""
import math
import numpy as np
from rev10_data import LABELS


def check_scores(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != len(LABELS) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Expected Nx8 finite probabilities')
    return scores


def tune_thresholds(gt, scores, original, target_precision=.99, min_predictions=30):
    """Maximize recall subject to empirical precision and support, on VAL only.
    If target unattainable, disable threshold-based emission and mark target unmet.
    If insufficient two-sided GT, disable threshold-based emission (threshold >1; top1 fallback still possible).
    Never lower an original threshold or go below .5. Unknown GT is ignored.
    """
    scores = check_scores(scores); gt = np.asarray(gt)
    original = np.asarray(original, dtype=float)
    if gt.shape != scores.shape or not np.isin(gt, [-1, 0, 1]).all():
        raise ValueError('Invalid GT')
    if original.shape != (8,) or not np.isfinite(original).all() or np.any((original < 0) | (original > 1)):
        raise ValueError('Invalid original thresholds')
    if not 0 < target_precision <= 1 or min_predictions < 1:
        raise ValueError('Invalid tuning controls')
    thresholds = np.full(8, 1.000001); audit = {}
    for j, label in enumerate(LABELS):
        known = gt[:, j] != -1; y = gt[known, j]; s = scores[known, j]
        pos = int(sum(y == 1)); neg = int(sum(y == 0)); floor = max(.5, float(original[j]))
        rec = {'positive': pos, 'negative': neg, 'unknown': int(sum(~known)), 'floor': floor,
               'target_precision': target_precision, 'min_predictions': min_predictions,
               'target_met': False, 'status': 'insufficient_support_no_output'}
        # Minimum support prevents a handful of selected examples deciding a class policy.
        if pos >= min_predictions and neg >= min_predictions:
            order = np.argsort(-s, kind='mergesort'); sorted_s = s[order]; sorted_y = y[order]
            end = np.flatnonzero(np.r_[sorted_s[:-1] != sorted_s[1:], True])
            count = end + 1; tp = np.cumsum(sorted_y == 1)[end]; fp = count - tp
            ts = sorted_s[end]; precision = tp / count; recall = tp / pos
            f05 = 1.25 * precision * recall / np.maximum(.25 * precision + recall, 1e-15)
            valid = np.flatnonzero((ts >= floor) & (count >= min_predictions))
            hits = [i for i in valid if precision[i] >= target_precision]
            if hits:
                k = max(hits, key=lambda i: (recall[i], precision[i], ts[i]))
                status = 'precision_target_met'
            elif len(valid):
                k = None
                status = 'target_unmet_no_threshold_emission'
            else:
                k = None; status = 'insufficient_predictions_no_output'
            rec['status'] = status
            if k is not None:
                thresholds[j] = ts[k]
                rec.update(target_met=bool(precision[k] >= target_precision),
                           empirical_precision=float(precision[k]), recall=float(recall[k]),
                           f05=float(f05[k]), predicted_known=int(count[k]), tp=int(tp[k]), fp=int(fp[k]))
        rec['threshold'] = float(thresholds[j]); audit[label] = rec
    return thresholds, audit


def top1_099(scores):
    """Earlier user proposal, retained ONLY as a comparison baseline."""
    scores = check_scores(scores); result = scores >= .99
    result[np.arange(len(scores)), np.argmax(scores, axis=1)] = True
    return result


def decide(scores, thresholds, conflict_margin=.05):
    """At least one output; indoor/outdoor are mutually exclusive product labels.
    Threshold candidates first. If both indoor/outdoor pass, keep higher raw
    score (stable LABELS order breaks exact ties). Close conflicts are flagged,
    not abstained. Only an empty candidate set triggers top1 fallback.
    """
    scores = check_scores(scores); thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.shape != (8,) or not np.isfinite(thresholds).all() or np.any(thresholds < 0) or not np.isfinite(conflict_margin) or not 0 <= conflict_margin <= 1:
        raise ValueError('Invalid policy parameters')
    candidates = scores >= thresholds; output = candidates.copy()
    indoor = LABELS.index('indoor'); outdoor = LABELS.index('outdoor')
    objective = LABELS.index('objective_image'); landscape = LABELS.index('landscape')
    notes = []
    for i in range(len(scores)):
        note = {'suppressed': {}, 'ambiguities': [], 'fallback_top1': False}
        if candidates[i, indoor] and candidates[i, outdoor]:
            delta = float(scores[i, indoor] - scores[i, outdoor])
            loser = outdoor if delta >= 0 else indoor
            output[i, loser] = False
            note['suppressed'][LABELS[loser]] = 'indoor_outdoor_lower_probability'
            note['indoor_minus_outdoor_score'] = delta
            if abs(delta) <= conflict_margin:
                note['ambiguities'].append('indoor_outdoor_close_scores_winner_kept')
        if not output[i].any():
            top = int(np.argmax(scores[i])); output[i, top] = True
            note.update(fallback_top1=True, fallback_label=LABELS[top],
                        fallback_probability=float(scores[i, top]),
                        fallback_threshold=float(thresholds[top]))
        if output[i, indoor] and output[i, landscape]:
            note['ambiguities'].append('indoor_landscape_possible_mixed_view_kept')
        if output[i, objective] and output[i].sum() > 1:
            note['ambiguities'].append('objective_and_scene_coexistence_kept_for_review')
        notes.append(note)
    if np.any(output[:, indoor] & output[:, outdoor]) or np.any(output.sum(axis=1) == 0):
        raise AssertionError('Output must be nonempty with exclusive indoor/outdoor')
    return output, notes
