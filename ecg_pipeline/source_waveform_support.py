"""Source ink and annotation evidence, with no morphology synthesis."""
from __future__ import annotations

from typing import Any
import hashlib
import cv2
import numpy as np
from ecg_pipeline.annotation_color import ANNOTATION_COLOR_RULE, color_candidates


def _pulse_exclusion_boxes(image: np.ndarray, geometry: dict[str, Any],
                           timing: dict[str, Any]) -> list[dict[str, Any]]:
    """Extend the measured descending edge only within its own source row.

    Timing measures the middle of each calibration edge. Its columns cannot be
    interpreted as waveform just above or below that measurement window either.
    The one-pixel margin matches the existing evidence/support kernel radius.
    """
    height, width = image.shape[:2]
    try:
        rows = np.asarray(geometry.get("rowCenters"), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Pulse exclusion requires four ordered source rows.") from error
    if (rows.shape != (4,) or not np.isfinite(rows).all()
            or not np.all(np.diff(rows) > 0) or rows[0] < 0 or rows[-1] >= height):
        raise ValueError("Pulse exclusion requires four ordered source rows.")
    measured_rows = timing.get("rowGridMeasurements")
    if (not isinstance(measured_rows, list) or len(measured_rows) != 4
            or any(not isinstance(mark, dict) or mark.get("rowCenter") != rows[row]
                   for row, mark in enumerate(measured_rows))):
        raise ValueError("Pulse exclusion rows must match the source timing measurements.")
    edges = timing.get("sourcePulseEdges")
    if not isinstance(edges, list) or len(edges) != 4:
        raise ValueError("Pulse exclusion requires all four observed source edges.")
    row_bounds = [0, *np.ceil((rows[:-1] + rows[1:]) / 2).astype(int).tolist(), height]
    boxes = []
    for row, edge in enumerate(edges):
        if (not isinstance(edge, dict) or type(edge.get("row")) is not int
                or edge["row"] != row):
            raise ValueError("Pulse edge identities must match the four ordered source rows.")
        bounds = edge.get("bounds")
        if (not isinstance(bounds, list) or len(bounds) != 4
                or any(type(value) is not int for value in bounds)):
            raise ValueError("Observed pulse edge bounds must be integer source coordinates.")
        left, top, right, bottom = bounds
        if not (0 <= left < right <= width
                and row_bounds[row] <= top < bottom <= row_bounds[row + 1]):
            raise ValueError("Observed pulse edge must lie within its source row and raster.")
        center = edge.get("centerX")
        measured_support = edge.get("minimumBlackColumnSupport")
        if (any(isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) for value in (center, measured_support))
                or center != (left + right - 1) / 2 or not .65 <= measured_support <= 1):
            raise ValueError("Pulse exclusion requires a supported observed descending edge.")
        observed = image[top:bottom, left:right].max(axis=2) < 160
        if not np.isclose(float(observed.mean(axis=0).min()), measured_support,
                          rtol=0, atol=1e-12):
            raise ValueError("Pulse edge support must match the untouched source pixels.")
        footprint = bounds
        joint = timing.get("version") in (4, 5)
        if joint:
            footprint = edge.get("observedBounds")
            if (not isinstance(footprint, list) or len(footprint) != 4
                    or any(type(value) is not int for value in footprint)
                    or not (0 <= footprint[0] <= left < right <= footprint[2] <= width
                            and row_bounds[row] <= footprint[1] <= top < bottom
                            <= footprint[3] <= row_bounds[row + 1])
                    or edge.get("sourceEndX") != footprint[2]):
                raise ValueError("Joint pulse exclusion requires its complete source footprint.")
        boxes.append({"kind": ("observed-calibration-full-edge-columns" if joint
                               else "observed-calibration-edge-columns"), "row": row,
                      "observedBounds": footprint.copy(), "observedCenterX": center,
                      **({"coreBounds": bounds.copy()} if joint else {}),
                      "minimumBlackColumnSupport": measured_support,
                      "bounds": [max(0, footprint[0] - 1), row_bounds[row],
                                 min(width, footprint[2] + 1), row_bounds[row + 1]],
                      "horizontalMarginPixels": 1,
                      "verticalExtent": "adjacent-source-row-midpoints",
                      "reason": "calibration-edge-is-not-waveform",
                      "originalTimeCoordinatesPreserved": True})
    return boxes


def _separator_exclusion_boxes(image: np.ndarray, timing: dict[str, Any]) -> list[dict[str, Any]]:
    """A consensus mask is the union of independently observed source rectangles."""
    height, width = image.shape[:2]
    if timing.get("version") == 5:
        from ecg_pipeline.joint_source_timing import FAMILY_METHOD
        from ecg_pipeline.source_separator_families import families
        if timing.get("method") != FAMILY_METHOD or "sourceSeparators" in timing:
            raise ValueError("Invalid endpoint-family source timing identity.")
        selected = timing.get("sourceSeparatorFamilies")
        if not isinstance(selected, list) or len(selected) != 9:
            raise ValueError("Family timing requires nine complete source families.")
        groups = families(image, [family['members'] for family in selected])
        if any(len(group) != 1 for group in groups) or [group[0] for group in groups] != selected:
            raise ValueError("Family masks must retain every endpoint and parent footprint.")
        boxes = []
        for family in selected:
            parent = family['parentComponent']['bounds']
            left, top, right, bottom = parent
            boxes.append({'kind': 'observed-separator-endpoint-family-parent-footprint',
                          'bounds': [max(0, left - 1), top, min(width, right + 1), bottom],
                          'parentBounds': parent.copy(),
                          'memberUnionBounds': family['memberUnionBounds'].copy(),
                          'memberCount': len(family['members']),
                          'horizontalCore': family['horizontalCore'].copy(),
                          'centerIntervalPixels': family['centerIntervalPixels'].copy(),
                          'allEndpointMembersRetained': True})
        return boxes
    if timing.get("version") == 4:
        if timing.get("method") != "source-joint-corresponding-grid-row-timing-v4":
            raise ValueError("Invalid joint source timing identity.")
        boxes = []
        marks = timing.get("sourceSeparators")
        if not isinstance(marks, list) or len(marks) != 9:
            raise ValueError("Joint timing requires nine ordered source separators.")
        for index, mark in enumerate(marks):
            core, observed = mark.get("bounds"), mark.get("observedBounds")
            if (mark.get("row") != index // 3 or mark.get("column") != index % 3 + 1
                    or any(not isinstance(box, list) or len(box) != 4
                           or any(type(v) is not int for v in box) for box in (core, observed))):
                raise ValueError("Joint separators require ordered integer source footprints.")
            left, top, right, bottom = core
            ol, ot, rr, bb = observed
            if not (0 <= ol <= left < right <= rr <= width and 0 <= ot == top < bottom == bb <= height):
                raise ValueError("Joint separator footprint must contain its observed core.")
            interval = sorted([(left + right - 1) / 2, (ol + rr - 1) / 2])
            actual = float((image[top:bottom, left:right].max(axis=2) < 160).mean(axis=0).min())
            if (mark.get("centerX") != (left + right - 1) / 2
                    or mark.get("centerIntervalPixels") != interval or interval[1] - interval[0] > 1
                    or actual < .8 or abs(actual - mark.get("minimumBlackColumnSupport", -1)) > 1e-12):
                raise ValueError("Joint separator support must match source pixels.")
            boxes.append({"kind": "observed-separator-joint-full-footprint",
                          "bounds": [max(0, ol - 1), ot, min(width, rr + 1), bb],
                          "representativeBounds": core.copy(), "observedBounds": observed.copy(),
                          "centerIntervalPixels": interval})
        return boxes
    consensus = timing.get("separatorWindowConsensus")
    if consensus is None:
        if timing.get("version") == 3:
            raise ValueError("Consensus timing requires observed separator masks.")
        return [{"kind": "observed-separator", "bounds": [max(0, mark["bounds"][0]-1),
                mark["bounds"][1], min(width, mark["bounds"][2]+1), mark["bounds"][3]]}
                for mark in timing["sourceSeparators"]]
    if (timing.get("version") != 3 or timing.get("method") != "source-separator-window-consensus-timing-v3"
            or consensus.get("method") != "bounded-source-separator-window-consensus-v1"
            or consensus.get("truthUsed") is not False):
        raise ValueError("Invalid source separator consensus identity.")
    groups = consensus.get("sourceSeparators");offsets = consensus.get("successfulOffsets")
    selected = consensus.get("selectedOffsetPixels")
    if (not isinstance(groups, list) or len(groups) != 9 or not isinstance(offsets, list)
            or len(offsets) < 2 or offsets != sorted(set(offsets))
            or selected != min(offsets, key=lambda offset: (abs(offset), offset))):
        raise ValueError("Source separator consensus requires all ordered observations.")
    boxes = []
    for i, group in enumerate(groups):
        observations = group.get("observations")
        if (group.get("row") != i//3 or group.get("column") != i%3+1
                or not isinstance(observations, list)
                or [o.get("offsetPixels") for o in observations] != offsets):
            raise ValueError("Source separator observations must retain row/column/offset identities.")
        rectangles, centers = [], []
        for observation in observations:
            bounds = observation.get("bounds")
            if (observation.get("row") != i//3 or observation.get("column") != i%3+1
                    or not isinstance(bounds, list) or len(bounds) != 4
                    or any(type(v) is not int for v in bounds)):
                raise ValueError("Separator observations require integer source rectangles.")
            left, top, right, bottom = bounds
            center, support = observation.get("centerX"), observation.get("minimumBlackColumnSupport")
            if (not 0 <= left < right <= width or not 0 <= top < bottom <= height
                    or center != (left+right-1)/2
                    or not isinstance(support, (int, float)) or isinstance(support, bool)
                    or not np.isfinite(support) or not .8 <= support <= 1):
                raise ValueError("Separator observation is not a supported source stroke.")
            actual = float((image[top:bottom, left:right].max(axis=2) < 160).mean(axis=0).min())
            if not np.isclose(actual, support, rtol=0, atol=1e-12):
                raise ValueError("Separator support must reproduce the original pixels.")
            if observation["offsetPixels"] == selected:
                if {k:v for k,v in observation.items() if k != "offsetPixels"} != timing["sourceSeparators"][i]:
                    raise ValueError("Representative separator must be an observed proposal.")
            rectangles.append(bounds);centers.append(center)
        union = [min(b[0] for b in rectangles), min(b[1] for b in rectangles),
                 max(b[2] for b in rectangles), max(b[3] for b in rectangles)]
        interval = [min(centers), max(centers)]
        intersection = [max(b[0] for b in rectangles), min(b[2] for b in rectangles)]
        if (group.get("unionBounds") != union or group.get("centerIntervalPixels") != interval
                or interval[1]-interval[0] > 1 or group.get("horizontalIntersection") != intersection
                or intersection[1] <= intersection[0]):
            raise ValueError("Separator uncertainty and masks must match all source observations.")
        boxes.append({"kind": "observed-separator-consensus-union",
            "bounds": [max(0,union[0]-1),union[1],min(width,union[2]+1),union[3]],
            "representativeBounds": timing["sourceSeparators"][i]["bounds"].copy(),
            "sourceObservationCount": len(observations), "centerIntervalPixels": interval})
    return boxes


def build_source_ink(image: np.ndarray, geometry: dict[str, Any], timing: dict[str, Any],
                     annotation_mask: np.ndarray | None = None):
    if (timing.get("state") != "source_supported" or image.ndim != 3 or image.shape[2] != 3
            or image.dtype != np.uint8 or timing.get("decodedRasterSha256") != hashlib.sha256(image.tobytes()).hexdigest()):
        raise ValueError("Source ink requires timing proved on the identical decoded raster.")
    grid = geometry["sourceLabelGrid"]
    if grid["imageSize"] != [image.shape[1], image.shape[0]]:
        raise ValueError("Source labels and ink must share their coordinate frame.")
    if timing.get("version") in (4, 5):
        from ecg_pipeline.joint_source_timing import validate_joint_timing_fields
        validate_joint_timing_fields(image, geometry, timing)
    black=(image.max(axis=2)<160).astype(np.uint8)
    excluded=np.zeros(black.shape,np.uint8);boxes=[]
    annotation_receipt = None
    if annotation_mask is not None:
        identity = timing.get("sourceIdentity") or {}
        mask_hash = identity.get("annotationMaskFileSha256", "")
        if (annotation_mask.shape != black.shape or annotation_mask.dtype != np.uint8
                or not np.isin(annotation_mask, [0, 255]).all()
                or identity.get("coordinateSpace") != "working"
                or len(mask_hash) != 64 or any(c not in "0123456789abcdef" for c in mask_hash)):
            raise ValueError("Annotation mask must be binary, source-bound and in the identical working frame.")
        red, blue = color_candidates(image[..., ::-1])  # source image is OpenCV BGR
        applied_mask = (annotation_mask > 0) & (red | blue)
        excluded[applied_mask] = 1
        annotation_receipt = {"method": "source-colour-seeds-before-vectorization-v2",
            "maskFileSha256": mask_hash, "decodedMaskSha256": hashlib.sha256(annotation_mask.tobytes()).hexdigest(),
            "imageSize": [image.shape[1], image.shape[0]], "coordinateSpace": "working",
            "inputExcludedPixels": int((annotation_mask > 0).sum()),
            "appliedMaskSha256": hashlib.sha256((applied_mask.astype(np.uint8)*255).tobytes()).hexdigest(),
            "colourRule": ANNOTATION_COLOR_RULE,
            "excludedPixels": int(excluded.sum()), "removedBlackPixels": int((black*excluded).sum()),
            "pixelExclusionApplied": True, "gapsPreserved": True}
    for slot in [*grid['slots'],grid['rhythmSlot']]:
        left,top,right,bottom=slot['box']
        if not (0 <= left < right <= black.shape[1] and 0 <= top < bottom <= black.shape[0]):
            raise ValueError("Source label crop is outside the raster.")
        anchor=next((a for a in grid['anchors'] if a['lead']==slot['lead']),None) if slot['row']<3 else None
        tightened=False
        if anchor is not None:
            x0=max(left,int(np.floor(anchor['x']))-2);x1=min(right,int(np.ceil(anchor['x']+anchor['width']))+2)
            y0=max(top,int(np.floor(anchor['y']-anchor['height']/2))-2);y1=min(bottom,int(np.ceil(anchor['y']+anchor['height']/2))+2)
            if x0<x1 and y0<y1:
                left,top,right,bottom=x0,y0,x1,y1;tightened=True
        excluded[top:bottom,left:right]=1;boxes.append({'kind':'verified-label-crop','lead':slot['lead'],'bounds':[left,top,right,bottom],
            'originalVerificationCrop':slot['box'],'tightenedToObservedNamedOcrBox':tightened,'sourceAnchor':anchor if tightened else None})
    for box in _separator_exclusion_boxes(image, timing):
        left,top,right,bottom=box['bounds']
        excluded[top:bottom,left:right]=1;boxes.append(box)
    pulse_excluded = np.zeros(black.shape, dtype=bool)
    for box in _pulse_exclusion_boxes(image, geometry, timing):
        left, top, right, bottom = box["bounds"]
        pulse_excluded[top:bottom, left:right] = True
        boxes.append(box)
    excluded[pulse_excluded] = 1
    clean=black*(1-excluded)
    evidence=cv2.GaussianBlur(clean.astype(np.float32),(3,3),.45)
    evidence[pulse_excluded] = 0
    supported=cv2.dilate(clean,np.ones((3,3),np.uint8)).astype(bool)
    # Do not recover evidence through an excluded label/mark using nearby ink.
    supported[excluded.astype(bool)]=False
    consensus_masks = timing.get("version") == 3
    family_masks = timing.get("version") == 5
    joint_masks = timing.get("version") in (4, 5)
    receipt = {
        "version": 5 if family_masks else 4 if joint_masks else 3 if consensus_masks else 2,
        "imageSize": [image.shape[1], image.shape[0]],
        "decodedRasterSha256": timing["decodedRasterSha256"],
        "method": ("neutral-source-ink-with-endpoint-family-parent-exclusions-v6" if family_masks else
                   "neutral-source-ink-with-joint-full-footprint-exclusions-v5" if joint_masks else
                   "neutral-source-ink-with-separator-consensus-exclusions-v4" if consensus_masks
                   else "neutral-source-ink-with-observed-calibration-exclusions-v3"),
        "thresholdMaximumChannelBelow": 160, "supportRadiusPixels": 1,
        "removedBlackPixels": int((black*excluded).sum()), "exclusionBoxes": boxes,
        "truthUsed": False, "gapsPreserved": True,
        **({"annotationExclusion": annotation_receipt} if annotation_receipt is not None else {}),
        **({"excludedMaskSha256": hashlib.sha256(excluded.tobytes()).hexdigest(),
            "sourceSupportSha256": hashlib.sha256(supported.astype(np.uint8).tobytes()).hexdigest(),
            "sourceEvidenceSha256": hashlib.sha256(evidence.tobytes()).hexdigest()}
           if joint_masks else {}),
    }
    if joint_masks and annotation_mask is None and "sourceInkSupport" in timing:
        from ecg_pipeline.joint_source_timing import source_ink_support
        if timing["sourceInkSupport"] != source_ink_support(geometry, timing, supported):
            raise ValueError("Joint timing ink support must match the original source mask.")
    return evidence, supported, excluded, receipt


def source_supported_validity(support: np.ndarray, columns: np.ndarray, path: np.ndarray,
                              current: np.ndarray | None) -> np.ndarray:
    """Keep the same gaps for canonical conversion, fidelity and display."""
    retained = support[path, columns]
    return retained if current is None else np.asarray(current, dtype=bool) & retained


def connected_source_validity(evidence: np.ndarray, columns: np.ndarray,
                              path: np.ndarray, validity: np.ndarray, *,
                              source_interval: tuple[int, int] | list[int]):
    """Abstain at unsupported transitions without changing source paths or gaps.

    Point support can admit isolated grid/label fragments. Check transitions
    beyond its three-pixel diameter using the existing vertical-ink test. Both
    endpoints become gaps; source evidence does not establish which one is wrong.
    """
    if (not isinstance(source_interval, (tuple, list)) or len(source_interval) != 2
            or any(isinstance(x, (bool, np.bool_))
                   or not isinstance(x, (int, np.integer)) for x in source_interval)):
        raise ValueError("Two verified integer interval bounds are required.")
    lo, hi = source_interval
    if evidence.ndim != 2 or not 0 <= lo < hi <= evidence.shape[1]:
        raise ValueError("Verified interval is outside source evidence.")
    if (columns.ndim != 1 or path.shape != columns.shape or validity.shape != columns.shape
            or validity.dtype != np.bool_ or not np.issubdtype(columns.dtype, np.integer)
            or not np.issubdtype(path.dtype, np.integer) or np.any(np.diff(columns) <= 0)
            or np.any(columns < 0) or np.any(columns >= evidence.shape[1])
            or np.any(path < 0) or np.any(path >= evidence.shape[0])):
        raise ValueError("Invalid source path or support validity.")
    retained = validity.copy()
    inside = (columns >= lo) & (columns < hi)
    jumps = np.abs(np.diff(path.astype(np.float64)))
    pairs = (validity[:-1] & validity[1:] & inside[:-1] & inside[1:]
             & (np.diff(columns) == 1))
    transitions = []
    # Decide from the original validity, so an earlier gap cannot conceal a
    # second unsupported transition at the same fragment.
    for index in np.flatnonzero(pairs & (jumps > 3)):
        lower = int(min(path[index], path[index + 1]))
        upper = int(max(path[index], path[index + 1])) + 1
        left = max(lo, int(columns[index]) - 3)
        right = min(hi, int(columns[index + 1]) + 4)
        block = evidence[lower:upper, left:right]
        if not np.isfinite(block).all():
            raise ValueError("Invalid source evidence.")
        support = float(np.mean(np.max(block, axis=1) >= .24))
        if support < .35:
            retained[index:index + 2] = False
            transitions.append({
                "fromSourcePixel": [int(columns[index]), int(path[index])],
                "toSourcePixel": [int(columns[index + 1]), int(path[index + 1])],
                "evidenceWindowBounds": [left, lower, right, upper],
                "verticalSupportFraction": support,
                "displacementPixels": float(jumps[index]),
            })
    return retained, {
        "version": 1, "method": "verified-source-transition-gaps-v1",
        "sourceInterval": [int(lo), int(hi)], "supportRadiusPixels": 1,
        "minimumDisplacementExclusivePixels": 3, "evidenceThreshold": .24,
        "minimumVerticalSupportFraction": .35, "horizontalMarginPixels": 3,
        "unsupportedTransitions": transitions,
        "removedSourceColumns": columns[validity & ~retained].tolist(),
        "pathModified": False, "existingGapsPreserved": True, "truthUsed": False,
    }


def piecewise_rhythm_to_canonical(convert, columns: np.ndarray, path: np.ndarray,
                                  panel_ranges: list[list[int]], **kwargs) -> np.ndarray:
    """Retain the full path in every call to share one baseline across panels."""
    if (kwargs["panel_samples"] != 5000 or len(panel_ranges) != 4
            or kwargs["panel_start"] != panel_ranges[0][0]
            or kwargs["panel_width"] != panel_ranges[-1][1]-panel_ranges[0][0]
            or any(b <= a for a, b in panel_ranges)
            or any(panel_ranges[i][1] != panel_ranges[i+1][0] for i in range(3))):
        raise ValueError("Rhythm conversion requires four contiguous verified source panels.")
    return np.concatenate([
        convert(columns, path, **{**kwargs, "panel_start": left,
                                 "panel_width": right-left, "panel_samples": 1250})
        for left, right in panel_ranges
    ])
