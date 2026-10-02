"""Observe source separator candidates; only extend coordinate evaluation to four coefficients."""
import hashlib
import numpy as np

def windows(image, geometry, offset):
    grid = geometry['sourceLabelGrid']
    font = grid['fontHeight']
    xfit = np.asarray(grid['xFit'])
    assert xfit.shape in [(3,), (4,)]
    height, width = image.shape[:2]
    mask = image.max(axis=2) < 160
    records = []
    for row, center in enumerate(geometry['rowCenters'][:3]):
        for column in [1, 2, 3]:
            basis = [1, row, column] if xfit.shape == (3,) else [1, row, column, row * column]
            predicted = float(xfit @ basis)
            left, right = (max(0, round(predicted - font)), min(width, round(predicted + font) + 1))
            top, bottom = (max(0, round(center - font * 0.8) + offset), min(height, round(center + font * 0.55) + offset))
            record = {'row': row, 'column': column, 'predictedX': predicted, 'bounds': [left, top, right, bottom]}
            if right - left < 5 or bottom - top < 5:
                records.append(record | {'candidateCount': -1, 'failureReason': 'window-outside-image'})
                continue
            profile = mask[top:bottom, left:right].mean(axis=0)
            indices = np.flatnonzero(profile >= 0.8)
            parts = [part for part in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if part.size]
            runs = []
            lower = max(3, round(font * 0.08))
            upper = max(4, round(font * 0.3))
            for part in parts:
                runs.append({'centerX': float(left + (part[0] + part[-1]) / 2), 'bounds': [int(left + part[0]), top, int(left + part[-1] + 1), bottom], 'widthPixels': int(part.size), 'minimumBlackColumnSupport': float(np.min(profile[part])), 'meanBlackColumnSupport': float(np.mean(profile[part])), 'widthAdmissible': bool(lower <= part.size <= upper)})
            records.append(record | {'profileSha256': hashlib.sha256(profile.tobytes()).hexdigest(), 'maximumColumnSupport': float(np.max(profile)), 'widthRange': [lower, upper], 'supportRuns': runs, 'candidateCount': sum((r['widthAdmissible'] for r in runs))})
    return records
