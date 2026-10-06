"""Prepara tres fotografías reales y las contamina con la misma anomalía.

Las imágenes proceden de `skimage.data` (dominio público / CC0):
- astronaut: Eileen Collins, NASA (dominio público).
- coffee: taza de café, Rachel Michetti (CC0).
- camera: cameraman, versión CC0 incluida en scikit-image.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from skimage import data, transform

ROOT = Path(__file__).resolve().parents[1]
SIZE = 256
DENSITY = 0.10
SOURCES = {1: data.astronaut, 2: data.coffee, 3: data.camera}


def prepare(image: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Reescala a size x size y devuelve uint8."""
    resized = transform.resize(image, (size, size), anti_aliasing=True)
    return np.rint(resized * 255).astype(np.uint8)


def add_impulses(image: np.ndarray, seed: int, density: float = DENSITY) -> tuple[np.ndarray, np.ndarray]:
    """Ruido sal y pimienta: una fracción `density` de píxeles pasa a 0 o 255."""
    rng = np.random.default_rng(seed)
    mask = rng.random(image.shape[:2]) < density
    values = np.where(rng.random(mask.sum()) < 0.5, 255, 0).astype(np.uint8)
    corrupted = image.copy()
    corrupted[mask] = values[:, None] if image.ndim == 3 else values
    return corrupted, mask


def save(name: str, image: np.ndarray) -> None:
    folder = ROOT / "data" / name
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(folder / f"{name}.png")


def main() -> None:
    for index, source in SOURCES.items():
        original = prepare(source())
        corrupted, mask = add_impulses(original, seed=2026 + index)
        save(f"original_{index}", original)
        save(f"corrupted_{index}", corrupted)
        save(f"mask_{index}", (mask * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
