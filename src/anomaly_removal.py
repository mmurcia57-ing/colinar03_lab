"""Detector contextual y restaurador propio para ruido impulsivo."""

from __future__ import annotations

import numpy as np


def _as_float_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("La imagen debe ser 2D o 3D.")
    if array.ndim == 3 and array.shape[2] not in (1, 3):
        raise ValueError("La imagen debe tener un canal o tres canales.")
    return array.astype(np.float32, copy=False)


def _window_values(padded: np.ndarray, row: int, col: int, radius: int) -> np.ndarray:
    """Devuelve la ventana sin el píxel central para una imagen 2D."""
    window = padded[row : row + 2 * radius + 1, col : col + 2 * radius + 1]
    values = window.reshape(-1)
    center = (2 * radius + 1) ** 2 // 2
    return np.delete(values, center)


def detect_impulses(
    image: np.ndarray,
    window_size: int = 5,
    mad_factor: float = 3.0,
    min_deviation: float = 22.0,
    saturation_margin: float = 12.0,
) -> np.ndarray:
    """Detecta outliers impulsivos usando contexto local robusto.

    La estadística y el recorrido de ventanas son propios. El detector combina
    MAD, desviación frente a la mediana y compatibilidad con saturación, evitando
    tratar como anomalía cualquier píxel intenso por el mero hecho de serlo.
    """
    array = _as_float_image(image)
    gray = array if array.ndim == 2 else array.mean(axis=2)
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size debe ser impar y >= 3.")
    radius = window_size // 2
    padded = np.pad(gray, radius, mode="reflect")
    mask = np.zeros(gray.shape, dtype=bool)

    for row in range(gray.shape[0]):
        for col in range(gray.shape[1]):
            neighbours = _window_values(padded, row, col, radius)
            local_median = float(np.median(neighbours))
            mad = float(np.median(np.abs(neighbours - local_median)))
            robust_scale = max(1.4826 * mad, 1.0)
            deviation = abs(float(gray[row, col]) - local_median)
            local_range = float(np.percentile(neighbours, 90) - np.percentile(neighbours, 10))
            edge_allowance = 1.0 if local_range < 45.0 else 1.35
            threshold = max(min_deviation, mad_factor * robust_scale) * edge_allowance
            saturated = gray[row, col] <= saturation_margin or gray[row, col] >= 255.0 - saturation_margin
            mask[row, col] = saturated and deviation > threshold

    return mask


def _restore_channel(channel: np.ndarray, mask: np.ndarray, window_size: int) -> np.ndarray:
    radius = window_size // 2
    padded = np.pad(channel, radius, mode="reflect")
    padded_mask = np.pad(mask, radius, mode="reflect")
    restored = channel.copy()

    for row, col in zip(*np.nonzero(mask)):
        values = _window_values(padded, row, col, radius)
        valid = ~_window_values(padded_mask.astype(np.uint8), row, col, radius).astype(bool)
        candidates = values[valid]
        if candidates.size == 0:
            candidates = values
        local_median = float(np.median(candidates))
        # Peso por distancia a la mediana: conserva textura mejor que una mediana fija.
        distances = np.abs(candidates - local_median)
        weights = 1.0 / (1.0 + distances)
        restored[row, col] = float(np.sum(candidates * weights) / np.sum(weights))

    return restored


def restore_impulses(image: np.ndarray, mask: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Restaura los píxeles detectados con vecinos no anómalos."""
    array = _as_float_image(image)
    if mask.shape != array.shape[:2]:
        raise ValueError("La máscara debe coincidir con alto y ancho de la imagen.")
    if array.ndim == 2:
        result = _restore_channel(array, mask, window_size)
    else:
        result = np.stack([_restore_channel(array[:, :, channel], mask, window_size) for channel in range(array.shape[2])], axis=2)
    return np.clip(result, 0, 255).astype(np.uint8)


def remove_impulses(image: np.ndarray, window_size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Ejecuta detección y restauración, devolviendo imagen y mapa."""
    array = _as_float_image(image)
    mask = detect_impulses(array, window_size=window_size)
    return restore_impulses(array, mask, window_size=window_size), mask
