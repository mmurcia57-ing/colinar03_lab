"""Detector y restaurador propio para ruido impulsivo (sal y pimienta).

El método se organiza en tres etapas por canal:

1. Candidatos: píxeles saturados (cerca de 0 o de 255). Un impulso de sal o
   pimienta siempre es extremo, pero no todo píxel extremo es un impulso.
2. Decisión contextual: cada candidato se compara solo con sus vecinos
   *fiables* (no saturados) en una ventana que crece hasta reunir suficiente
   soporte. Se descarta como anomalía si pertenece a una región saturada real
   (varios vecinos inmediatos con la misma saturación) o si su valor cae dentro de la
   envolvente [mín - tol, máx + tol] de los vecinos fiables.
3. Restauración: solo se modifican los píxeles marcados, usando vecinos no
   marcados ponderados por distancia espacial y por parecido a la mediana local.

Ideas de partida (referenciadas en el notebook): filtros de mediana conmutados
(detectar antes de corregir), filtro de mediana adaptativo de Hwang y Haddad
(1995, ventana creciente) y filtro bilateral de Tomasi y Manduchi (1998, pesos
espaciales y de rango). La combinación concreta (contexto solo con vecinos
fiables, test de polaridad para regiones saturadas, envolvente local y
restauración ponderada restringida a vecinos limpios) es propia.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Parámetros por defecto. Se fijan una vez y se usan sin cambios en todas las imágenes.
SATURATION_MARGIN = 10      # niveles: un candidato debe estar en [0, 10] o [245, 255]
MIN_RADIUS = 1              # ventana inicial 3x3
MAX_RADIUS = 4              # ventana máxima 9x9
MIN_SUPPORT = 4             # vecinos fiables necesarios para decidir con estadística
REGION_NEIGHBOURS_3X3 = 3   # vecinos con la misma saturación que tiene la esquina de un objeto en 3x3...
REGION_NEIGHBOURS_5X5 = 8   # ...y en 5x5. Se exigen ambos para considerar región saturada real.
ENVELOPE_TOLERANCE = 20.0   # niveles de margen alrededor de la envolvente de vecinos fiables

# Códigos de la decisión, útiles para visualizar el paso a paso.
NOT_CANDIDATE = 0
IMPULSE = 1
SATURATED_REGION = 2
INSIDE_ENVELOPE = 3


@dataclass
class DetectionDetails:
    """Resultados intermedios de la detección para un canal."""

    candidates: np.ndarray   # bool: píxeles saturados (etapa 1)
    decision: np.ndarray     # int: código de decisión por píxel (etapa 2)
    radius: np.ndarray       # int: radio de ventana usado en cada candidato (0 si no candidato)
    reference: np.ndarray    # float: mediana de los vecinos fiables (NaN si no aplica)

    @property
    def mask(self) -> np.ndarray:
        return self.decision == IMPULSE


def _validate(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("La imagen debe ser 2D (gris) o 3D (color).")
    if array.ndim == 3 and array.shape[2] not in (1, 3):
        raise ValueError("La imagen debe tener uno o tres canales.")
    return array.astype(np.float32)


def _neighbourhood(padded: np.ndarray, row: int, col: int, radius: int, pad: int) -> np.ndarray:
    """Valores de la ventana (2r+1)x(2r+1) centrada en (row, col), sin el centro."""
    r0, c0 = row + pad - radius, col + pad - radius
    window = padded[r0 : r0 + 2 * radius + 1, c0 : c0 + 2 * radius + 1].reshape(-1)
    return np.delete(window, window.size // 2)


def analyze_channel(channel: np.ndarray) -> DetectionDetails:
    """Ejecuta las etapas 1 y 2 sobre un canal y devuelve los mapas intermedios."""
    low = channel <= SATURATION_MARGIN
    high = channel >= 255 - SATURATION_MARGIN
    candidates = low | high

    pad = MAX_RADIUS
    values = np.pad(channel, pad, mode="reflect")
    low_p = np.pad(low, pad, mode="reflect")
    high_p = np.pad(high, pad, mode="reflect")

    decision = np.zeros(channel.shape, dtype=np.uint8)
    radius_map = np.zeros(channel.shape, dtype=np.uint8)
    reference = np.full(channel.shape, np.nan, dtype=np.float32)

    for row, col in zip(*np.nonzero(candidates)):
        x = channel[row, col]
        same_polarity = high_p if high[row, col] else low_p
        for radius in range(MIN_RADIUS, MAX_RADIUS + 1):
            neigh = _neighbourhood(values, row, col, radius, pad)
            unreliable = _neighbourhood(low_p | high_p, row, col, radius, pad)
            reliable = neigh[~unreliable]
            if reliable.size >= MIN_SUPPORT:
                break
        radius_map[row, col] = radius

        # Región saturada real: incluso en la esquina de un objeto saturado hay 3 vecinos
        # iguales en 3x3 y 8 en 5x5. Con ruido de densidad d cada vecino coincide con
        # probabilidad d/2, así que reunir ambas condiciones por azar es improbable.
        same_3 = _neighbourhood(same_polarity, row, col, 1, pad).sum()
        same_5 = _neighbourhood(same_polarity, row, col, 2, pad).sum()
        if same_3 >= REGION_NEIGHBOURS_3X3 and same_5 >= REGION_NEIGHBOURS_5X5:
            decision[row, col] = SATURATED_REGION
            continue
        if reliable.size == 0:
            # Sin contexto fiable y sin región saturada: solo puede ser ruido denso.
            decision[row, col] = IMPULSE
            continue

        median = float(np.median(reliable))
        reference[row, col] = median
        lower = reliable.min() - ENVELOPE_TOLERANCE
        upper = reliable.max() + ENVELOPE_TOLERANCE
        decision[row, col] = IMPULSE if (x < lower or x > upper) else INSIDE_ENVELOPE

    return DetectionDetails(candidates, decision, radius_map, reference)


def detect_impulses(image: np.ndarray) -> np.ndarray:
    """Devuelve la máscara de impulsos con la misma forma que la imagen (por canal)."""
    array = _validate(image)
    if array.ndim == 2:
        return analyze_channel(array).mask
    return np.stack([analyze_channel(array[:, :, c]).mask for c in range(array.shape[2])], axis=2)


def _restore_channel(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pad = MAX_RADIUS
    values = np.pad(channel, pad, mode="reflect")
    flagged = np.pad(mask, pad, mode="reflect")
    restored = channel.copy()

    for row, col in zip(*np.nonzero(mask)):
        for radius in range(MIN_RADIUS, MAX_RADIUS + 1):
            neigh = _neighbourhood(values, row, col, radius, pad)
            valid = ~_neighbourhood(flagged, row, col, radius, pad)
            if valid.sum() >= MIN_SUPPORT:
                break
        if not valid.any():
            continue  # sin vecinos limpios: se conserva el valor (caso extremo)
        side = 2 * radius + 1
        dy, dx = np.divmod(np.arange(side * side), side)
        dist2 = np.delete((dy - radius) ** 2 + (dx - radius) ** 2, side * side // 2)[valid]
        samples = neigh[valid]
        median = np.median(samples)
        # Peso espacial (cercanía) x peso de rango (parecido a la mediana fiable).
        weights = np.exp(-dist2 / (2.0 * radius**2)) / (1.0 + np.abs(samples - median))
        restored[row, col] = float(np.sum(weights * samples) / np.sum(weights))

    return restored


def restore_impulses(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sustituye solo los valores marcados en `mask` (misma forma que la imagen)."""
    array = _validate(image)
    if mask.shape != array.shape:
        raise ValueError("La máscara debe tener la misma forma que la imagen.")
    if array.ndim == 2:
        result = _restore_channel(array, mask)
    else:
        result = np.stack([_restore_channel(array[:, :, c], mask[:, :, c]) for c in range(array.shape[2])], axis=2)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def remove_impulses(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detecta y restaura. Devuelve (imagen restaurada, máscara por píxel)."""
    mask = detect_impulses(image)
    restored = restore_impulses(image, mask)
    pixel_mask = mask if mask.ndim == 2 else mask.any(axis=2)
    return restored, pixel_mask
