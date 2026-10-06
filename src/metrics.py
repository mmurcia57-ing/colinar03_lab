"""Métricas de evaluación (no forman parte del algoritmo propuesto)."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def detection_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Precisión, recall y F1 de la máscara detectada frente a la real."""
    predicted = np.asarray(predicted, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.logical_and(predicted, truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def restoration_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """MAE, PSNR (dB) y SSIM frente a la imagen limpia."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    error = reference - estimate
    mse = float(np.mean(error**2))
    psnr = float(10 * np.log10(255.0**2 / max(mse, 1e-12)))
    channel_axis = 2 if reference.ndim == 3 else None
    ssim = structural_similarity(reference, estimate, data_range=255.0, channel_axis=channel_axis)
    return {"mae": float(np.mean(np.abs(error))), "psnr": psnr, "ssim": float(ssim)}
