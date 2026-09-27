"""Genera dos escenas reproducibles y las contamina con la misma anomalía."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def scene_one(size: int = 128) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, :] = [155, 205, 235]
    image[78:, :, :] = [80, 105, 80]
    draw = ImageDraw.Draw(Image.fromarray(image))
    draw.rectangle((12, 36, 48, 85), fill=(180, 145, 105))
    draw.polygon([(8, 36), (30, 16), (53, 36)], fill=(100, 65, 50))
    draw.rectangle((25, 58, 36, 85), fill=(55, 70, 75))
    draw.rectangle((66, 28, 112, 85), fill=(220, 190, 145))
    draw.polygon([(60, 28), (89, 7), (118, 28)], fill=(95, 75, 65))
    draw.rectangle((77, 50, 90, 64), fill=(60, 125, 175))
    draw.line((0, 108, 128, 96), fill=(230, 220, 175), width=5)
    return np.asarray(draw._image).copy()


def scene_two(size: int = 128) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, 0] = np.clip(30 + y * 0.35, 0, 255)
    image[:, :, 1] = np.clip(110 + y * 0.45, 0, 255)
    image[:, :, 2] = np.clip(180 - y * 0.55, 0, 255)
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    draw.polygon([(0, 88), (28, 45), (55, 88)], fill=(55, 105, 100))
    draw.polygon([(35, 92), (78, 30), (128, 92)], fill=(75, 125, 95))
    draw.ellipse((84, 12, 112, 40), fill=(250, 225, 125))
    for cx, cy, r in [(15, 105, 15), (43, 110, 18), (108, 107, 20)]:
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(35, 80, 55))
    return np.asarray(pil)


def add_impulses(image: np.ndarray, seed: int, density: float = 0.035) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mask = rng.random(image.shape[:2]) < density
    corrupted = image.copy()
    salt = rng.random(mask.sum()) < 0.5
    corrupted[mask] = np.where(salt[:, None], 255, 0).astype(np.uint8)
    return corrupted, mask


def save(name: str, image: np.ndarray) -> None:
    folder = ROOT / "data" / name
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(folder / f"{name}.png")


def main() -> None:
    originals = [scene_one(), scene_two()]
    for index, original in enumerate(originals, 1):
        corrupted, mask = add_impulses(original, seed=2026 + index)
        save(f"original_{index}", original)
        save(f"corrupted_{index}", corrupted)
        save(f"mask_{index}", (mask * 255).astype(np.uint8))


if __name__ == "__main__":
    main()

