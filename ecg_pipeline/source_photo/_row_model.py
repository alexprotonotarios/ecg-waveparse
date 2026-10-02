"""Preserve vertical QRS and ablate source-row profile prominence; source raster is untouched."""
import cv2, numpy as np

def masks(image, grid):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    font = grid['fontHeight']
    field = cv2.GaussianBlur(gray, (0, 0), 1.5 * font, sigmaY=1.5 * font)
    raw = ((image.max(axis=2) < 160) & (field - gray >= 12)).astype(np.uint8)
    channels = image.astype(np.int16)
    neutral = (raw & (channels.max(axis=2) - channels.min(axis=2) <= 28)).astype(np.uint8)
    kw = max(3, round(0.4 * grid['columnSpacing']))
    kh = max(3, round(0.24 * grid['rowSpacing']))
    horizontal = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((1, kw), np.uint8))
    vertical = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((kh, 1), np.uint8))
    excluded = cv2.dilate(np.maximum(horizontal, vertical), np.ones((3, 3), np.uint8))
    h_excluded = cv2.dilate(horizontal, np.ones((3, 3), np.uint8))
    return ({'grid-clean': neutral * (1 - excluded), 'horizontal-only': neutral * (1 - h_excluded)}, {'horizontalKernelPixels': kw, 'verticalKernelPixels': kh, 'exclusionRadiusPixels': 1, 'excludedPixels': int(excluded.sum()), 'removedNeutralPixels': int((excluded * neutral).sum())})

def rows(mask, grid, prominence=False):
    h, w = mask.shape
    spacing, font = (grid['rowSpacing'], grid['fontHeight'])
    centers = []
    details = []
    for row, label_y in enumerate(grid['labelRowCenters']):
        top = max(0, round(label_y - spacing * 0.35))
        bottom = min(h, round(label_y - max(font * 0.6, spacing * 0.04)))
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
        if prominence:
            background = float(np.median(smooth))
            excess = float(smooth[local] - background)
            details[-1].update(profileBackground=background, peakAboveBackground=excess, requiredPeakAboveBackground=0.04)
            if excess < 0.04:
                return {'centers': None, 'reason': 'insufficient-row-profile-prominence', 'rows': details}
        centers.append(center)
    gaps = np.diff(centers)
    if not np.all((gaps > spacing * 0.7) & (gaps < spacing * 1.3)):
        return {'centers': None, 'reason': 'inconsistent-row-gaps', 'rows': details}
    return {'centers': centers, 'reason': None, 'rows': details}
