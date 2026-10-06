"""Figuras del notebook. Solo presentación; no contiene lógica del algoritmo."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

from src.anomaly_removal import IMPULSE, INSIDE_ENVELOPE, NOT_CANDIDATE, SATURATED_REGION

DECISION_COLORS = {
    NOT_CANDIDATE: ("#f2f2f2", "No candidato"),
    IMPULSE: ("#d62728", "Impulso"),
    SATURATED_REGION: ("#1f77b4", "Región saturada real"),
    INSIDE_ENVELOPE: ("#ff7f0e", "Dentro de la envolvente"),
}


def _show(ax, image, title, cmap=None):
    ax.imshow(image, cmap=cmap if image.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def show_row(images, titles, size=2.6):
    """Muestra una fila de imágenes (gris o color) con títulos."""
    fig, axes = plt.subplots(1, len(images), figsize=(size * len(images), size))
    for ax, image, title in zip(np.atleast_1d(axes), images, titles):
        _show(ax, np.asarray(image), title, cmap="gray")
    fig.tight_layout()
    plt.show()


def show_decisions(corrupted, details, crop):
    """Contaminada, candidatos, decisión por clase y radio de ventana, con recorte ampliado."""
    r0, r1, c0, c1 = crop
    cmap = ListedColormap([DECISION_COLORS[k][0] for k in sorted(DECISION_COLORS)])
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    _show(axes[0], corrupted, "Contaminada (recuadro = zoom)", cmap="gray")
    axes[0].add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, color="yellow", lw=1.5))
    axes[1].imshow(details.candidates[r0:r1, c0:c1], cmap="gray", interpolation="nearest")
    axes[1].set_title("Etapa 1: candidatos (zoom)", fontsize=9)
    axes[2].imshow(details.decision[r0:r1, c0:c1], cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    axes[2].set_title("Etapa 2: decisión (zoom)", fontsize=9)
    im = axes[3].imshow(np.where(details.candidates, details.radius, np.nan)[r0:r1, c0:c1], cmap="viridis", interpolation="nearest")
    axes[3].set_title("Radio de ventana usado (zoom)", fontsize=9)
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    for ax in axes[1:]:
        ax.axis("off")
    handles = [Patch(color=color, label=label) for color, label in DECISION_COLORS.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plt.show()


def show_results_grid(rows, crop_size=64):
    """Una fila por imagen: contaminada, detección, restaurada, error y zoom antes/después."""
    titles = ["Contaminada", "Detección", "Restaurada", "|Error| x4", "Zoom contaminada", "Zoom restaurada"]
    fig, axes = plt.subplots(len(rows), 6, figsize=(13, 2.3 * len(rows)))
    for i, (name, clean, corrupted, detected, restored) in enumerate(rows):
        h, w = clean.shape[:2]
        r0, c0 = (h - crop_size) // 2, (w - crop_size) // 2
        crop = (slice(r0, r0 + crop_size), slice(c0, c0 + crop_size))
        error = np.abs(clean.astype(int) - restored.astype(int))
        error = error.max(axis=2) if error.ndim == 3 else error
        panels = [corrupted, (detected * 255).astype(np.uint8), restored, np.clip(error * 4, 0, 255), corrupted[crop], restored[crop]]
        for j, panel in enumerate(panels):
            _show(axes[i, j], np.asarray(panel), titles[j] if i == 0 else "", cmap="gray")
        axes[i, 0].text(-12, h / 2, name, rotation=90, va="center", ha="right", fontsize=9)
    fig.tight_layout()
    plt.show()


def markdown_table(header, rows):
    """Tabla Markdown para mostrar resultados (se ve bien en Jupyter y en el PDF)."""
    from IPython.display import Markdown

    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return Markdown("\n".join(lines))
