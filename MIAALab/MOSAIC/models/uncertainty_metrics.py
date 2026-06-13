"""

  - AUCRef (Area under the Referral Curve)
  - AURC (Area under the Risk-Coverage Curve)
  - ECE (Expected Calibration Error)

"""

import numpy as np
import torch
import torch.nn.functional as F

# ============================================================
# Patch-level uncertainty aggregation
# ============================================================

def patch_aggregate_uncertainty(unc_map, patch_size=5):
    if isinstance(unc_map, np.ndarray):
        unc_map = torch.from_numpy(unc_map).float()
    if unc_map.dim() == 2:
        unc_map = unc_map.unsqueeze(0).unsqueeze(0)
    elif unc_map.dim() == 3:
        unc_map = unc_map.unsqueeze(0)

    sums = F.avg_pool2d(unc_map, kernel_size=patch_size, stride=1) * (patch_size ** 2)
    max_sum = sums.max().item()
    return max_sum / (patch_size ** 2)

# ============================================================
# NCC
# ============================================================

def normalized_cross_correlation(pred_unc_map, ref_unc_map):
    a = ref_unc_map.flatten().astype(np.float64)
    b = pred_unc_map.flatten().astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    sa, sb = a.std(), b.std()
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(((a - mu_a) * (b - mu_b)).mean() / (sa * sb))

# ============================================================
# AUCRef
# ============================================================

def area_under_referral_curve(uncertainties, dice_scores, n_thresholds=20):
    uncertainties = np.asarray(uncertainties, dtype=np.float64)
    dice_scores = np.asarray(dice_scores, dtype=np.float64)
    n = len(uncertainties)
    if n == 0:
        return 0.0, np.array([]), np.array([])

    order = np.argsort(uncertainties)
    sorted_dice = dice_scores[order]

    fractions = np.linspace(0.0, 0.9, n_thresholds)
    dice_curve = []
    for frac in fractions:
        keep_n = max(1, int(n * (1 - frac)))
        dice_curve.append(sorted_dice[:keep_n].mean())
    dice_curve = np.array(dice_curve)

    auc = float(np.trapz(dice_curve, fractions) / (fractions[-1] - fractions[0]))
    return auc, fractions, dice_curve

# ============================================================
# AURC
# ============================================================

def area_under_risk_coverage_curve(uncertainties, dice_scores, n_thresholds=100):
    uncertainties = np.asarray(uncertainties, dtype=np.float64)
    dice_scores = np.asarray(dice_scores, dtype=np.float64)
    risks = 1.0 - dice_scores
    n = len(uncertainties)
    if n == 0:
        return 0.0

    order = np.argsort(uncertainties)
    sorted_risks = risks[order]

    coverages = np.arange(1, n + 1) / n
    cumulative_risks = np.cumsum(sorted_risks) / np.arange(1, n + 1)

    aurc = float(np.trapz(cumulative_risks, coverages))
    return aurc

# ============================================================
# ECE
# ============================================================

def expected_calibration_error(pred_probs, gt_labels, n_bins=15):
    pred_probs = np.asarray(pred_probs, dtype=np.float64).flatten()
    gt_labels = np.asarray(gt_labels, dtype=np.float64).flatten()
    n = len(pred_probs)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (pred_probs >= lo) & (pred_probs <= hi)
        else:
            mask = (pred_probs >= lo) & (pred_probs < hi)
        m = mask.sum()
        if m == 0:
            continue
        bin_conf = pred_probs[mask].mean()
        bin_acc = gt_labels[mask].mean()
        ece += (m / n) * abs(bin_conf - bin_acc)
    return float(ece)

# ============================================================
# ============================================================

def disagreement_stratified_dice(per_sample_pred_prob, per_sample_gt_masks,
                                 per_sample_gt_var=None,
                                 thresholds=None,
                                 use_quantile=True):
    """Dice stratified by ground-truth inter-rater disagreement.

    Args:
        per_sample_gt_masks:  list of [num_anno,H,W] ndarray, GT

    Returns:
        dict: {'DSD_low', 'DSD_med', 'DSD_high', 'DSD_low_n', 'DSD_med_n', 'DSD_high_n',
               'DSD_low_thr', 'DSD_high_thr'}
    """
    n = len(per_sample_pred_prob)
    if n == 0:
        return {}

    if per_sample_gt_var is None:
        per_sample_gt_var = []
        for gt in per_sample_gt_masks:
            gt = np.asarray(gt, dtype=np.float32)
            v = gt.var(axis=0).mean()
            per_sample_gt_var.append(float(v))
    per_sample_gt_var = np.array(per_sample_gt_var)

    if use_quantile and thresholds is None:
        lo_thr = float(np.percentile(per_sample_gt_var, 33.33))
        hi_thr = float(np.percentile(per_sample_gt_var, 66.67))
    else:
        lo_thr, hi_thr = thresholds if thresholds is not None else (0.05, 0.15)

    bins = {
        'low':  per_sample_gt_var < lo_thr,
        'med':  (per_sample_gt_var >= lo_thr) & (per_sample_gt_var < hi_thr),
        'high': per_sample_gt_var >= hi_thr,
    }

    per_sample_dice = []
    for prob, gt in zip(per_sample_pred_prob, per_sample_gt_masks):
        prob = np.asarray(prob, dtype=np.float32)
        gt = np.asarray(gt, dtype=np.float32)
        gt_majority = (gt.mean(axis=0) >= 0.5).astype(np.float32)
        pred_b = (prob > 0.5).astype(np.float32)
        inter = (pred_b * gt_majority).sum()
        denom = pred_b.sum() + gt_majority.sum()
        if denom < 1e-8:
            per_sample_dice.append(1.0)
        else:
            per_sample_dice.append(2 * inter / denom)
    per_sample_dice = np.array(per_sample_dice)

    out = {'DSD_low_thr': lo_thr, 'DSD_high_thr': hi_thr}
    for name, mask in bins.items():
        cnt = int(mask.sum())
        if cnt == 0:
            out[f'DSD_{name}'] = 0.0
            out[f'DSD_{name}_n'] = 0
        else:
            out[f'DSD_{name}'] = float(per_sample_dice[mask].mean())
            out[f'DSD_{name}_n'] = cnt
    return out

# ============================================================
# ============================================================

def au_dice_correlation(per_sample_image_au, per_sample_dice):
    """Correlation between image-level predicted uncertainty and image-level Dice.

    Args:
    Returns:
    """
    a = np.asarray(per_sample_image_au, dtype=np.float64)
    b = np.asarray(per_sample_dice, dtype=np.float64)
    if len(a) < 2 or a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    mu_a, mu_b = a.mean(), b.mean()
    return float(((a - mu_a) * (b - mu_b)).mean() / (a.std() * b.std()))

# ============================================================
# ============================================================

def boundary_dice(per_sample_pred_prob, per_sample_gt_masks, boundary_width=5):
    """Dice computed within a boundary band of width boundary_width around the GT contour.

    Args:
        per_sample_gt_masks:  list of [num_anno,H,W] ndarray, GT
    Returns:
    """
    try:
        import cv2
    except ImportError:
        return 0.0

    bdry_dices = []
    for prob, gt in zip(per_sample_pred_prob, per_sample_gt_masks):
        prob = np.asarray(prob, dtype=np.float32)
        gt = np.asarray(gt, dtype=np.float32)
        gt_majority = (gt.mean(axis=0) >= 0.5).astype(np.uint8)
        if gt_majority.sum() == 0:
            continue

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(gt_majority, kernel, iterations=boundary_width)
        eroded = cv2.erode(gt_majority, kernel, iterations=boundary_width)
        boundary_mask = (dilated - eroded).astype(np.float32)  # 0 or 1

        if boundary_mask.sum() == 0:
            continue

        pred_b = (prob > 0.5).astype(np.float32)

        pred_in_bdry = pred_b * boundary_mask
        gt_in_bdry = gt_majority.astype(np.float32) * boundary_mask
        inter = (pred_in_bdry * gt_in_bdry).sum()
        denom = pred_in_bdry.sum() + gt_in_bdry.sum()
        if denom < 1e-8:
            continue
        bdry_dices.append(2 * inter / denom)

    return float(np.mean(bdry_dices)) if bdry_dices else 0.0

# ============================================================
# ============================================================

def compute_all_uncertainty_metrics(
        per_sample_pred_au,
        per_sample_ref_var,
        per_sample_pred_total_unc,
        per_sample_dice,
        per_sample_pred_prob,
        per_sample_gt_label,
        patch_size=5,
        per_sample_gt_masks=None,
        per_sample_image_au=None):
    """
    Args:
    Returns:
    """
    out = {}

    ncc_vals = []
    for pred_au, ref_var in zip(per_sample_pred_au, per_sample_ref_var):
        ncc_vals.append(normalized_cross_correlation(pred_au, ref_var))
    out['NCC'] = float(np.mean(ncc_vals)) if ncc_vals else 0.0

    aucref, _, _ = area_under_referral_curve(per_sample_pred_total_unc, per_sample_dice)
    out['AUCRef'] = aucref

    out['AURC'] = area_under_risk_coverage_curve(per_sample_pred_total_unc, per_sample_dice)

    all_probs = np.concatenate([p.flatten() for p in per_sample_pred_prob])
    all_labels = np.concatenate([l.flatten() for l in per_sample_gt_label])
    out['ECE'] = expected_calibration_error(all_probs, all_labels)

    if per_sample_gt_masks is not None and len(per_sample_gt_masks) > 0:
        # DSD
        dsd_out = disagreement_stratified_dice(
            per_sample_pred_prob, per_sample_gt_masks)
        out.update(dsd_out)

        # Boundary Dice
        out['BoundaryDice'] = boundary_dice(
            per_sample_pred_prob, per_sample_gt_masks, boundary_width=5)

    # AU-Dice correlation
    if per_sample_image_au is None:
        per_sample_image_au = [float(np.mean(au)) for au in per_sample_pred_au]
    out['AUDC'] = au_dice_correlation(per_sample_image_au, per_sample_dice)

    return out