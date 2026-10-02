"""Shared unchanged source-colour detector for annotation preparation and support."""
from __future__ import annotations

import numpy as np

ANNOTATION_COLOR_RULE = "red-blue-source-pixel-rule-v1"


def color_candidates(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = image.astype(np.int16)
    red, green, blue = (rgb[..., index] for index in range(3))
    saturation = rgb.max(axis=2) - rgb.min(axis=2)

    blue_pixels = (
        (blue >= 100)
        & (blue - red >= 35)
        & (blue - green >= 15)
        & (saturation >= 50)
    )
    red_pixels = (
        (red >= 120)
        & (green <= 100)
        & (blue <= 100)
        & (red - green >= 35)
        & (red - blue >= 35)
        & (saturation >= 50)
    )
    return red_pixels, blue_pixels
