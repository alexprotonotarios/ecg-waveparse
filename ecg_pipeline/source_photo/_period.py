"""Frozen estimator with one explicit lag-cap argument; default behavior unchanged."""
import cv2
import numpy as np
from scipy.signal import find_peaks

def grid_period(profile: np.ndarray, cap: int=48) -> tuple[float | None, float]:
    length = profile.size
    if length < 80:
        return (None, 0.0)
    values = profile.astype(np.float64)
    values -= cv2.GaussianBlur(values[:, None].astype(np.float32), (1, 0), sigmaX=0, sigmaY=max(8.0, length / 24)).ravel()
    scale = float(np.linalg.norm(values))
    if scale <= 1e-09:
        return (None, 0.0)
    maximum = min(cap, length // 8)
    correlations = np.asarray([float(np.dot(values[:-lag], values[lag:])) / max(float(np.linalg.norm(values[:-lag]) * np.linalg.norm(values[lag:])), 1e-09) for lag in range(2, maximum + 1)])
    peaks, _ = find_peaks(correlations, prominence=0.03)
    if peaks.size == 0:
        return (None, 0.0)
    credible = [index for index in peaks if correlations[index] >= 0.18]
    if not credible:
        return (None, 0.0)
    index = credible[0]
    subpixel_offset = 0.0
    if 0 < index < correlations.size - 1:
        denominator = correlations[index - 1] - 2.0 * correlations[index] + correlations[index + 1]
        if abs(float(denominator)) > 1e-09:
            subpixel_offset = float(np.clip(0.5 * (correlations[index - 1] - correlations[index + 1]) / denominator, -0.5, 0.5))
    return (float(index + 2 + subpixel_offset), float(min(1.0, correlations[index])))
