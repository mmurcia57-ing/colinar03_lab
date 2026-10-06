import numpy as np

from scripts.generate_data import add_impulses
from src.anomaly_removal import IMPULSE, SATURATED_REGION, analyze_channel, detect_impulses, remove_impulses, restore_impulses
from src.metrics import detection_metrics, restoration_metrics


def textured_image(size=64, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    base = 128 + 60 * np.sin(x / 5.0) * np.cos(y / 7.0) + rng.normal(0, 8, (size, size))
    return np.clip(base, 30, 225).astype(np.uint8)


def test_detects_and_restores_single_impulse():
    clean = np.full((15, 15), 120, dtype=np.uint8)
    corrupted = clean.copy()
    corrupted[7, 7] = 255
    mask = detect_impulses(corrupted)
    restored = restore_impulses(corrupted, mask)
    assert mask[7, 7] and mask.sum() == 1
    assert restored[7, 7] == 120


def test_real_saturated_region_is_preserved():
    image = np.full((20, 20), 100, dtype=np.uint8)
    image[5:15, 5:15] = 255  # objeto blanco real, no ruido
    details = analyze_channel(image.astype(np.float32))
    assert not details.mask.any()
    assert (details.decision[5:15, 5:15] == SATURATED_REGION).all()


def test_clean_image_is_left_untouched():
    clean = textured_image()
    restored, mask = remove_impulses(clean)
    assert not mask.any()
    assert np.array_equal(restored, clean)


def test_same_parameters_work_on_gray_and_color_with_dense_noise():
    gray = textured_image(seed=1)
    color = np.stack([gray, textured_image(seed=2), textured_image(seed=3)], axis=2)
    for clean in (gray, color):
        for density in (0.05, 0.20):
            corrupted, truth = add_impulses(clean, seed=7, density=density)
            restored, mask = remove_impulses(corrupted)
            det = detection_metrics(mask, truth)
            assert det["precision"] > 0.98 and det["recall"] > 0.95
            assert restoration_metrics(clean, restored)["psnr"] > restoration_metrics(clean, corrupted)["psnr"] + 10


def test_decision_codes_mark_impulses():
    clean = textured_image()
    corrupted, truth = add_impulses(clean, seed=3, density=0.1)
    details = analyze_channel(corrupted.astype(np.float32))
    assert ((details.decision == IMPULSE) == details.mask).all()
    assert detection_metrics(details.mask, truth)["f1"] > 0.95
