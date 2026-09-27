"""Métricas reproducibles para detección y restauración."""

from __future__ import annotations

import numpy as np


def detection_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.logical_and(predicted, truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12)}


def restoration_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float32)
    estimate = np.asarray(estimate, dtype=np.float32)
    error = reference - estimate
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error ** 2))
    psnr = float(10 * np.log10((255.0 ** 2) / max(mse, 1e-12)))
    # SSIM global simplificado, suficiente para comparar el mismo par de imágenes.
    mean_x, mean_y = float(reference.mean()), float(estimate.mean())
    var_x, var_y = float(reference.var()), float(estimate.var())
    cov = float(np.mean((reference - mean_x) * (estimate - mean_y)))
    c1, c2 = 6.5025, 58.5225
    ssim = ((2 * mean_x * mean_y + c1) * (2 * cov + c2)) / ((mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2))
    return {"mae": mae, "psnr": psnr, "ssim": float(ssim)}

