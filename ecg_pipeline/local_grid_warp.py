"""Experimental vertical grid correction. No production image path calls this module."""
from __future__ import annotations

import numpy as np
import cv2
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks


class VerticalGridWarp:
    """Smooth nonfolding vertical displacement supported by multiple grid rows.

    A restricted experiment for page curvature shared by grid rows. General
    perspective remains upstream's global homography. Unsupported extrapolation,
    folds, row disagreement and large local scale changes are refused.
    """
    def __init__(self, x_nodes: np.ndarray, observed_grid_y: np.ndarray, reference_grid_y: np.ndarray, *, max_slope: float = .25, max_row_residual_pixels: float = 1.0):
        if not np.isfinite(max_slope) or max_slope <= 0 or not np.isfinite(max_row_residual_pixels) or max_row_residual_pixels < 0:
            raise ValueError("Local warp budgets must be finite, with positive slope and nonnegative residual.")
        x=np.asarray(x_nodes,dtype=float);observed=np.asarray(observed_grid_y,dtype=float);reference=np.asarray(reference_grid_y,dtype=float)
        if x.ndim!=1 or reference.ndim!=1 or len(x)<5 or observed.ndim!=2 or observed.shape!=(len(reference),len(x)) or len(reference)<3:
            raise ValueError("Local warp requires five X nodes and three independent grid rows.")
        if not all(np.isfinite(a).all() for a in (x,observed,reference)) or np.any(np.diff(x)<=0) or np.any(np.diff(reference)<=0):
            raise ValueError("Local grid nodes must be finite and strictly ordered.")
        if np.any(np.diff(observed,axis=0)<=0):raise ValueError("Folded grid rows are invalid.")
        row_displacements=observed-reference[:,None]
        displacement=np.median(row_displacements,axis=0)
        if np.max(np.abs(row_displacements-displacement))>max_row_residual_pixels:
            raise ValueError("Grid rows disagree with the restricted vertical mapping.")
        self.spline=CubicSpline(x,displacement,bc_type="natural",extrapolate=False)
        probes=np.linspace(x[0],x[-1],max(100,len(x)*20))
        slopes=self.spline(probes,1)
        if np.max(np.abs(slopes))>max_slope:
            raise ValueError("Implausible local slope exceeds the experimental budget.")
        self.bounds=(float(x[0]),float(x[-1]))
        self.reference_bounds=(float(reference[0]),float(reference[-1]))
        self.evidence={"version":1,"method":"experimental-shared-vertical-grid-spline-v1","gridRowCount":len(reference),"nodeCount":len(x),
            "maxAbsSlope":float(np.max(np.abs(slopes))),"jacobianDeterminant":1.0,"boundsX":list(self.bounds),"referenceBoundsY":list(self.reference_bounds),
            "maxRowResidualPixels":float(np.max(np.abs(row_displacements-displacement))),"productionEnabled":False}
        self.evidence["transform"] = {"version":1,"mapping":"working_y = source_y - displacement(source_x)",
            "inverseMapping":"source_y = working_y + displacement(working_x)",
            "interpolation":"natural-cubic-spline-no-extrapolation", "xNodes":x.tolist(),
            "displacementAtNodes":displacement.tolist(), "resamplingCount":1}

    def map(self, points: np.ndarray, *, inverse: bool = False) -> np.ndarray:
        points=np.asarray(points,dtype=float)
        if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():raise ValueError("Expected finite XY points.")
        if np.any((points[:,0]<self.bounds[0])|(points[:,0]>self.bounds[1])):raise ValueError("Grid extrapolation is unsupported.")
        displacement=self.spline(points[:,0])
        reference_y=points[:,1] if inverse else points[:,1]-displacement
        if np.any((reference_y<self.reference_bounds[0])|(reference_y>self.reference_bounds[1])):raise ValueError("Point lies outside supported grid rows.")
        result=points.copy();result[:,1]+=displacement if inverse else -displacement
        return result


def detect_vertical_grid_warp(image: np.ndarray, *, node_step_pixels: int = 12) -> VerticalGridWarp:
    """Detect a restricted shared vertical warp from a visible coloured grid.

    No waveform, known distortion, or truth coordinate enters this detector.
    It anchors the grid to its observed left edge, so absolute translation and
    physical calibration remain separate. Grey grids, gaps in support, phase
    slips and row-dependent distortions are refused, never extrapolated.
    BGR input is the unmodified output of cv2.imread.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size > 36_000_000:
        raise ValueError("Expected a bounded BGR uint8 grid image.")
    height, width = image.shape[:2]
    if min(height, width) < 80 or not isinstance(node_step_pixels, int) or not 3 <= node_step_pixels <= 32:
        raise ValueError("Grid image or node spacing is outside experimental bounds.")
    channels = image.astype(np.int16)
    chroma = np.max(channels, axis=2) - np.min(channels, axis=2)
    coloured = (chroma >= 18) & (np.max(channels, axis=2) >= 150)
    # Median across a narrow strip rejects vertical grid lines and dark traces.
    def profile(x: int) -> np.ndarray:
        left, right = max(0, x - 3), min(width, x + 4)
        return np.mean(coloured[:, left:right], axis=1)
    seed = profile(0)
    peaks, _ = find_peaks(seed, height=.55, prominence=.25, distance=3)
    peaks = peaks[(peaks >= 6) & (peaks < height - 6)]
    if len(peaks) < 8:
        raise ValueError("Insufficient independently visible coloured grid rows.")
    periods = np.diff(peaks)
    period = float(np.median(periods))
    if period < 3 or np.mean(np.abs(periods-period) <= 1) < .9:
        raise ValueError("Grid row period is ambiguous or unsupported.")
    nodes = np.unique(np.append(np.arange(0, width, node_step_pixels), width-1)).astype(float)
    reference = peaks.astype(float)
    observed = np.empty((len(reference), len(nodes)), dtype=float)
    observed[:, 0] = reference
    # Small inter-node motion prevents silently switching to an adjacent line.
    radius = max(1, int(np.floor(period * .4)))
    previous_shift = 0.0
    support = []
    for column, node in enumerate(nodes[1:], 1):
        values = profile(int(node))
        found = np.full(len(reference), np.nan)
        for row, expected in enumerate(reference + previous_shift):
            center = int(round(expected))
            lo, hi = max(0, center-radius), min(height, center+radius+1)
            if hi <= lo:
                continue
            local = values[lo:hi]
            maximum = float(local.max())
            minimum = float(local.min())
            if maximum < .55 or maximum-minimum < .25:
                continue
            # Centroid of the strongest connected response preserves subpixel
            # shifts from raster antialiasing without changing waveform pixels.
            at = int(np.argmax(local))
            a, b = max(0, at-1), min(len(local), at+2)
            weights = np.maximum(0, local[a:b]-minimum)
            found[row] = float(np.average(np.arange(lo+a, lo+b), weights=weights))
        valid = np.isfinite(found)
        fraction = float(np.mean(valid))
        if fraction < .8:
            raise ValueError("Grid support is missing across part of the image.")
        shifts = found[valid] - reference[valid]
        shift = float(np.median(shifts))
        if abs(shift-previous_shift) >= period * .45 or np.max(np.abs(shifts-shift)) > 1:
            raise ValueError("Grid phase or shared vertical mapping is ambiguous.")
        # No per-row invention: only rows observed at every node are fitted.
        observed[:, column] = found
        previous_shift = shift
        support.append(fraction)
    complete = np.isfinite(observed).all(axis=1)
    if np.count_nonzero(complete) < 8:
        raise ValueError("Too few complete grid rows support the mapping.")
    warp = VerticalGridWarp(nodes, observed[complete], reference[complete], max_row_residual_pixels=1)
    warp.evidence.update(gridNodesIndependentlyDetected=True,
                         detector="coloured-grid-row-tracking-v1",
                         minimumNodeSupportFraction=min(support),
                         gridPeriodPixels=period,
                         sourceWidth=width, sourceHeight=height,
                         absoluteTranslationKnown=False)
    return warp


def apply_vertical_grid_warp(image: np.ndarray, warp: VerticalGridWarp) -> tuple[np.ndarray, np.ndarray]:
    """One inverse raster resampling, with an explicit supported-pixel mask."""
    height, width = image.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    inside_x = (xx >= warp.bounds[0]) & (xx <= warp.bounds[1])
    inside_y = (yy >= warp.reference_bounds[0]) & (yy <= warp.reference_bounds[1])
    shift = warp.spline(np.clip(xx[0], *warp.bounds)).astype(np.float32)
    source_y = yy + shift[None, :]
    supported = inside_x & inside_y & (source_y >= 0) & (source_y <= height-1)
    output = cv2.remap(image, xx, source_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(255,255,255))
    output[~supported] = 255
    return output, supported
