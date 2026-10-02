"""Fixed source-row mask comparison, with no trace interpolation or source edit."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib

import cv2
import numpy as np

def _maximum_channel(image):
    if image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3:
        return np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
    return image.max(axis=2)

def _channel_extrema(image):
    if image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3:
        maximum = _maximum_channel(image)
        minimum = np.minimum(np.minimum(image[..., 0], image[..., 1]), image[..., 2])
        return maximum, minimum
    return image.max(axis=2), image.min(axis=2)

def _compute_masks(image, font):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5 * font, sigmaY=1.5 * font)
    channels = image if image.dtype == np.uint8 else image.astype(np.int16)
    maximum, minimum = _channel_extrema(channels)
    chroma = maximum - minimum
    return {'baseline': (maximum < 160).astype(np.uint8), 'neutral-local-contrast': ((maximum < 160) & (chroma <= 28) & (background - gray >= 12)).astype(np.uint8)}

_MASK_CACHE = ContextVar('source_photo_mask_cache', default=None)

@contextmanager
def invocation_masks():
    token = _MASK_CACHE.set({})
    try:
        yield
    finally:
        _MASK_CACHE.reset(token)

def masks(image, font):
    cache = _MASK_CACHE.get()
    if cache is None or not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not isinstance(font, (int, float)):
        return _compute_masks(image, font)
    # Pixel binding prevents reuse after any in-place source change. Two entries
    # cover the full raster and rhythm crop without retaining a growing cache.
    key = (id(image), image.shape, font, hashlib.sha256(image.tobytes()).digest())
    if key not in cache:
        computed = _compute_masks(image, font)
        if len(cache) == 2:
            cache.pop(next(iter(cache)))
        cache[key] = (image, computed)
    original, computed = cache[key]
    assert original is image
    return {name: value.copy() for name, value in computed.items()}

def rows(mask, grid):
    h, w = mask.shape
    spacing, font = (grid['rowSpacing'], grid['fontHeight'])
    centers = []
    details = []
    for row, label_y in enumerate(grid['labelRowCenters']):
        top = max(0, round(label_y - spacing * 0.35))
        bottom = min(h, round(label_y - max(font * 0.85, spacing * 0.04)))
        if bottom - top < 4:
            return {'centers': None, 'reason': 'short-row-band', 'rows': details}
        source_row = min(row, 2)
        columns = grid['xFit'][2] + source_row * grid['xFit'][3]
        ranges = []
        for col in range(4):
            start = float(np.asarray([1, source_row, col, source_row * col]) @ np.asarray(grid['xFit']))
            left, right = (max(0, round(start + max(font * 3.5, columns * 0.12))), min(w, round(start + columns * 0.88)))
            if right - left < 20:
                return {'centers': None, 'reason': 'short-column-range', 'rows': details}
            ranges.append((left, right))
        profiles = [mask[top:bottom, left:right].mean(axis=1) for left, right in ranges]
        smooth = np.convolve(np.mean(profiles, axis=0), np.ones(3) / 3, mode='same')
        local = int(np.argmax(smooth))
        center = top + local
        radius = max(2, round(spacing * 0.025))
        support = [float(np.max(p[max(0, local - radius):min(len(p), local + radius + 1)])) for p in profiles]
        details.append({'row': row, 'band': [top, bottom], 'ranges': [list(r) for r in ranges], 'support': support, 'smoothedPeak': float(smooth[local]), 'center': center})
        if min(support) < 0.025 or float(smooth[local]) < 0.04:
            return {'centers': None, 'reason': 'insufficient-source-row-ink', 'rows': details}
        centers.append(center)
    gaps = np.diff(centers)
    if not np.all((gaps > spacing * 0.7) & (gaps < spacing * 1.3)):
        return {'centers': None, 'reason': 'inconsistent-row-gaps', 'rows': details}
    return {'centers': centers, 'reason': None, 'rows': details}
