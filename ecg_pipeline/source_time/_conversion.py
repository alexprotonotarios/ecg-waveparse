"""Fixed 25 mm/s, 500 Hz conversion within connected retained support."""
import bisect
import math
from fractions import Fraction
import numpy as np


def runs(mask, connected):
    output = []
    for index in np.flatnonzero(mask):
        index = int(index)
        if not output or index != output[-1][-1] + 1 or not connected[index - 1]:
            output.append([index])
        else:
            output[-1].append(index)
    return output


def convert(values, paper, connected, origin, count):
    values = np.asarray(values, dtype=float)
    paper = np.asarray(paper, dtype=float)
    connected = np.asarray(connected)
    if (values.ndim != 1 or values.shape != paper.shape or connected.dtype != bool
            or connected.shape != (len(values) - 1,) or not math.isfinite(origin)
            or type(count) is not int or not 1 <= count <= 5000):
        raise ValueError('invalid_conversion_input')
    if not np.array_equal(np.isfinite(values), np.isfinite(paper)):
        raise ValueError('signal_and_coordinate_missingness_differ')
    spans = runs(np.isfinite(paper), connected)
    output = np.full(count, np.nan)
    left = np.full(count, -1, np.int32); right = left.copy()
    weights = np.full(count, np.nan); left_weights = weights.copy()
    fractions = [None] * count
    targets = [Fraction(float(origin)) + Fraction(j, 20) for j in range(count)]
    for span in spans:
        positions = [Fraction(float(paper[i])) for i in span]
        if not all(b > a for a, b in zip(positions, positions[1:])):
            raise ValueError('non_increasing_source_time')
        for j, position in enumerate(targets):
            if position < positions[0] or position > positions[-1]:
                continue
            if left[j] != -1:
                raise ValueError('overlapping_source_time_support')
            k = bisect.bisect_left(positions, position)
            if positions[k] == position:
                lo = hi = span[k]; exact = Fraction(0)
            else:
                lo, hi = span[k - 1], span[k]
                exact = (position - positions[k - 1]) / (positions[k] - positions[k - 1])
                if hi != lo + 1 or not connected[lo] or not 0 < exact < 1:
                    raise ValueError('unsupported_conversion_interval')
            # Rounding the right weight before subtracting would silently discard
            # a required contributor when an interior weight rounds to one.
            weight, left_weight = float(exact), float(1 - exact)
            output[j] = float(values[lo]) if lo == hi else float(values[lo]) * left_weight + float(values[hi]) * weight
            left[j] = lo; right[j] = hi; weights[j] = weight; left_weights[j] = left_weight
            fractions[j] = [exact.numerator, exact.denominator]
    return {'values': output, 'left': left, 'right': right, 'rightWeight': weights, 'leftWeight': left_weights,
            'rightWeightRational': fractions, 'targetPaperXmm': np.array([float(x) for x in targets]),
            'sourceRuns': [[span[0], span[-1] + 1] for span in spans]}
