import numpy as np

from src.anomaly_removal import detect_impulses, restore_impulses
from src.metrics import detection_metrics, restoration_metrics


def test_detects_and_restores_single_impulse():
    clean = np.full((15, 15), 120, dtype=np.uint8)
    clean[7, 7] = 130
    corrupted = clean.copy()
    corrupted[7, 7] = 255
    mask = detect_impulses(corrupted)
    restored = restore_impulses(corrupted, mask)
    assert mask[7, 7]
    assert restored[7, 7] < 150
    truth = np.zeros((15, 15), dtype=bool)
    truth[7, 7] = True
    assert detection_metrics(mask, truth)["recall"] == 1.0
    assert restoration_metrics(clean, restored)["mae"] < 1.0
