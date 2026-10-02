"""Conservative below-trace 3x4/rhythm proposals from named source labels.

Partial OCR proposes crops only. The existing twelve-name verifier, a separate
rhythm-II check and independent waveform-row support must all pass before this
module returns a layout. No waveform truth, file names or patient metadata enter.
"""
from __future__ import annotations

import json
import hashlib
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import cv2
import numpy as np

from .lead_label_identity import (
    STANDARD_LEADS, STANDARD_THREE_BY_FOUR, _label_mask,
    _rapidocr_ocr_runner, attach_semantic_lead_identity,
)

METHOD = "source-value-grid-below-trace-v1"

_anchor_observer: ContextVar[Any] = ContextVar("source_label_anchor_observer", default=None)


@contextmanager
def source_anchor_observer(observer):
    """Observe actual source tokens in this context without changing recognition."""
    token = _anchor_observer.set(observer)
    try:
        yield
    finally:
        _anchor_observer.reset(token)


def read_label_anchors(image: np.ndarray, ocr_runner=None) -> list[dict[str, Any]]:
    tokens = _read_label_anchors(image, ocr_runner)
    observer = _anchor_observer.get()
    if observer is not None:
        observer(image, tokens)
    return tokens


def _read_label_anchors(image: np.ndarray, ocr_runner=None) -> list[dict[str, Any]]:
    """Retain exact lead tokens only; header text never enters the report."""
    runner = ocr_runner or _rapidocr_ocr_runner
    allowed = {name.casefold(): name for name in STANDARD_LEADS}
    h, w = image.shape[:2]
    scale = min(1.0, 3200 / max(w, h))
    small = cv2.resize(image, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1 else image
    tokens = []
    for raster in (small, (255*(1-_label_mask(small))).astype(np.uint8)):
        output = runner(raster)
        for observation in output.get("observations", []):
            candidates = observation.get("candidates") or []
            if not candidates:
                continue
            name = "".join(str(candidates[0].get("text", "")).split()).casefold()
            if name not in allowed:
                continue
            values = [observation.get(key) for key in ("x", "y", "width", "height")]
            confidence = candidates[0].get("confidence")
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) for value in [*values, confidence]):
                continue
            x, y, width, height = values
            if not (0 <= x < x+width <= 1 and 0 <= y < y+height <= 1 and .65 <= confidence <= 1):
                continue
            tokens.append({"lead": allowed[name], "confidence": confidence,
                           "x": x*w, "y": (1-y-height/2)*h,
                           "width": width*w, "height": height*h})
    return tokens


def propose_label_grid(image: np.ndarray, tokens: list[dict[str, Any]]) -> dict[str, Any]:
    h, w = image.shape[:2]
    rejected = {"layoutHint": None, "method": METHOD, "confidence": 0.0}
    positions = {lead: (row, col) for row, names in enumerate(STANDARD_THREE_BY_FOUR)
                 for col, lead in enumerate(names) if col > 0}
    selected = {}
    for token in tokens:
        lead = token["lead"]
        if lead not in positions:
            continue
        previous = selected.get(lead)
        if previous and (abs(token["x"]-previous["x"]) > .025*w or abs(token["y"]-previous["y"]) > .025*h):
            return {**rejected, "failureReason": "duplicate-source-label-locations"}
        if previous is None or token["confidence"] > previous["confidence"]:
            selected[lead] = token
    row_counts = [sum(positions[lead][0] == row for lead in selected) for row in range(3)]
    if len(selected) < 6 or min(row_counts) < 2 or {positions[lead][1] for lead in selected} != {1, 2, 3}:
        return {**rejected, "failureReason": "insufficient-source-anchors"}
    design = np.asarray([[1, *positions[lead]] for lead in selected], dtype=float)
    xs = np.asarray([token["x"] for token in selected.values()])
    ys = np.asarray([token["y"] for token in selected.values()])
    xfit = np.linalg.lstsq(design, xs, rcond=None)[0]
    yfit = np.linalg.lstsq(design, ys, rcond=None)[0]
    spacing, columns = float(yfit[1]), float(xfit[2])
    font = float(np.median([token["height"] for token in selected.values()]))
    xerror = float(np.max(np.abs(design@xfit-xs)))
    yerror = float(np.max(np.abs(design@yfit-ys)))
    if not (.09*h < spacing < .30*h and .12*w < columns < .30*w
            and abs(xfit[1]) < columns*.06 and abs(yfit[2]) < spacing*.06
            and xerror < columns*.04 and yerror < max(font*.6, spacing*.04)
            and .02*spacing < font < .18*spacing):
        return {**rejected, "failureReason": "inconsistent-source-label-grid"}
    slots = []
    for row, names in enumerate(STANDARD_THREE_BY_FOUR):
        for col, lead in enumerate(names):
            point = np.asarray([1, row, col])
            left, center = float(point@xfit), float(point@yfit)
            box = [round(left-font*.65), round(center-font*.6), round(left+font*3.4), round(center+font*.6)]
            if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                return {**rejected, "failureReason": "source-label-crop-outside-image"}
            slots.append({"lead": lead, "row": row, "column": col, "box": box})
    # This fallback supports the independently named fourth row. Never erase an
    # unrecognized extra strip by silently falling back to a plain 3x4 layout.
    rhythm = [token for token in tokens if token["lead"] == "II"
              and abs(token["x"]-float(xfit@[1, 3, 0])) < columns*.12
              and abs(token["y"]-float(yfit@[1, 3, 0])) < spacing*.25]
    if not rhythm:
        return {**rejected, "failureReason": "rhythm-ii-source-anchor-missing"}
    anchor = max(rhythm, key=lambda token: token["confidence"])
    if any(abs(token["y"]-anchor["y"]) > font for token in rhythm):
        return {**rejected, "failureReason": "ambiguous-rhythm-source-anchor"}
    rhythm_box = [round(anchor["x"]-font*.65), round(anchor["y"]-font*.6),
                  round(anchor["x"]+font*3.4), round(anchor["y"]+font*.6)]
    if not (0 <= rhythm_box[0] < rhythm_box[2] <= w and 0 <= rhythm_box[1] < rhythm_box[3] <= h):
        return {**rejected, "failureReason": "rhythm-crop-outside-image"}
    label_rows = [float(yfit@[1, row, 1.5]) for row in range(3)] + [anchor["y"]]
    return {"layoutHint": "standard_3x4_with_r1", "method": METHOD, "confidence": .85,
            "rowCenters": [round(center-spacing*.16) for center in label_rows],
            "sourceLabelGrid": {"version": 1, "method": METHOD, "imageSize": [w, h],
                "rowSpacing": spacing, "columnSpacing": columns, "fontHeight": font,
                "labelRowCenters": label_rows, "xFit": xfit.tolist(), "yFit": yfit.tolist(),
                "maximumXResidual": xerror, "maximumYResidual": yerror,
                "anchors": list(selected.values()), "slots": slots,
                "rhythmSlot": {"lead": "II", "row": 3, "column": 0, "box": rhythm_box},
                "maskMethod": "maximum-channel-below-160-v1"}}


def supported_trace_rows(image: np.ndarray, grid: dict[str, Any]) -> list[int] | None:
    """Locate observed ink above each below-trace label; names alone are insufficient."""
    h, w = image.shape[:2]
    mask = (image[..., :3].max(axis=2) < 160).astype(float) if image.ndim == 3 else (image < 160).astype(float)
    spacing, columns, font = grid["rowSpacing"], grid["columnSpacing"], grid["fontHeight"]
    xfit = np.asarray(grid["xFit"])
    centers = []
    for row, label_y in enumerate(grid["labelRowCenters"]):
        top = max(0, round(label_y-spacing*.35))
        bottom = min(h, round(label_y-max(font*.85, spacing*.04)))
        if bottom-top < 4:
            return None
        ranges = []
        for col in range(4):
            start = float(xfit@[1, min(row, 2), col])
            left, right = max(0, round(start+max(font*3.5, columns*.12))), min(w, round(start+columns*.88))
            if right-left < 20:
                return None
            ranges.append((left, right))
        profiles = [mask[top:bottom, left:right].mean(axis=1) for left, right in ranges]
        profile = np.mean(profiles, axis=0)
        smooth = np.convolve(profile, np.ones(3)/3, mode="same")
        local = int(np.argmax(smooth)); center = top+local
        radius = max(2, round(spacing*.025))
        support = [float(np.max(p[max(0, local-radius):min(len(p), local+radius+1)])) for p in profiles]
        if min(support) < .025 or float(smooth[local]) < .04:
            return None
        centers.append(center)
    gaps = np.diff(centers)
    if not np.all((gaps > spacing*.70) & (gaps < spacing*1.30)):
        return None
    return centers


# Same non-Roman source positions as the frozen strict proposal.
POSITIONS = {"aVR": (0,1), "V1": (0,2), "V4": (0,3),
             "aVL": (1,1), "V2": (1,2), "V5": (1,3),
             "aVF": (2,1), "V3": (2,2), "V6": (2,3)}


def _valid_local_label_token(token, width, height):
    if not isinstance(token, dict):
        return False
    values = [token.get(key) for key in ["x", "y", "width", "height", "confidence"]]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in values):
        return False
    x,y,w,h,confidence = values
    return (0 <= x < x+w <= width and 0 <= y-h/2 < y+h/2 <= height
            and .65 <= confidence <= 1)


def propose_local_label_crops(image, tokens):
    rejected = {"state": "ineligible", "geometryConfirmed": False, "truthUsed": False}
    def reject(reason, **details):
        return {**rejected, "failureReason": reason, **details}
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        return reject("invalid-source-raster")
    height,width = image.shape[:2]
    selected = {}
    for token in tokens:
        if not _valid_local_label_token(token,width,height):
            return reject("invalid-source-token")
        lead = token.get("lead")
        if lead not in POSITIONS:
            continue
        previous = selected.get(lead)
        if previous and (abs(token["x"]-previous["x"]) > .025*width or
                         abs(token["y"]-previous["y"]) > .025*height):
            return reject("duplicate-source-label-locations")
        if previous is None or token["confidence"] > previous["confidence"]:
            selected[lead] = dict(token)
    row_counts = [sum(POSITIONS[lead][0] == row for lead in selected) for row in range(3)]
    columns_seen = sorted({POSITIONS[lead][1] for lead in selected})
    if len(selected) < 6 or min(row_counts) < 1 or len(columns_seen) < 2:
        return reject("insufficient-preliminary-source-anchors", rowCounts=row_counts, columnsObserved=columns_seen)
    design = np.asarray([[1,*POSITIONS[lead]] for lead in selected],dtype=float)
    rank = int(np.linalg.matrix_rank(design))
    if rank != 3:
        return reject("rank-deficient-source-anchors")
    xs = np.asarray([token["x"] for token in selected.values()])
    ys = np.asarray([token["y"] for token in selected.values()])
    xfit = np.linalg.lstsq(design,xs,rcond=None)[0]
    yfit = np.linalg.lstsq(design,ys,rcond=None)[0]
    spacing,columns = float(yfit[1]),float(xfit[2])
    font = float(np.median([token["height"] for token in selected.values()]))
    xerror = float(np.max(np.abs(design@xfit-xs)))
    yerror = float(np.max(np.abs(design@yfit-ys)))
    checks = {"rowSpacing": .09*height < spacing < .30*height,
              "columnSpacing": .12*width < columns < .30*width,
              "horizontalSkew": abs(xfit[1]) < columns*.06,
              "verticalSkew": abs(yfit[2]) < spacing*.06,
              "horizontalResidual": xerror < columns*.04,
              "verticalResidual": yerror < max(font*.6,spacing*.04),
              "fontHeight": .02*spacing < font < .18*spacing}
    checks = {key: bool(value) for key, value in checks.items()}
    fit = {"xFit": xfit.tolist(), "yFit": yfit.tolist(), "rowSpacing": spacing,
           "columnSpacing": columns, "fontHeight": font, "maximumXResidual": xerror,
           "maximumYResidual": yerror, "rank": rank, "rowCounts": row_counts,
           "columnsObserved": columns_seen, "numericChecks": checks}
    if not all(checks.values()):
        return reject("unchanged-numeric-grid-bound-failed",fit=fit)
    crops = []
    for row,column in [*( (r,c) for r in range(3) for c in range(4)), (3,0)]:
        point=np.asarray([1,row,column]);left,center=float(point@xfit),float(point@yfit)
        box=[round(left-font*.65),round(center-font*.6),round(left+font*3.4),round(center+font*.6)]
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            return reject("recognition-crop-outside-source",fit=fit)
        crops.append({"row":row,"column":column,"role":"rhythm" if row==3 else "panel","bounds":box})
    return {"state":"crop_proposal","geometryConfirmed":False,"truthUsed":False,
            "fit":fit,"originalAnchors":list(selected.values()),"crops":crops,
            "finalStrictProposalAndAllNameRhythmRowChecksStillRequired":True,
            "expectedNamesProvidedToRecognizer":False}


def _translate_local_label_observations(tokens, bounds):
    left,top,right,bottom=bounds
    result=[]
    for token in tokens:
        if not _valid_local_label_token(token,right-left,bottom-top):
            raise ValueError("OCR token must remain within its actual source crop.")
        result.append({**token,"x":token["x"]+left,"y":token["y"]+top})
    return result



def recover_local_label_grid(image, tokens, original_proposal, *, ocr_runner=None):
    """Use image-only local observations, followed by the unchanged strict grid."""
    projected = propose_local_label_crops(image, tokens)
    if projected["state"] != "crop_proposal":
        return original_proposal
    combined = list(tokens)
    observations = []
    for crop in projected["crops"]:
        left, top, right, bottom = crop["bounds"]
        raster = image[top:bottom, left:right].copy()
        local = read_label_anchors(raster, ocr_runner)
        translated = _translate_local_label_observations(local, crop["bounds"])
        observations.append({**crop,
            "rasterSha256": hashlib.sha256(raster.tobytes()).hexdigest(),
            "localTokens": local, "sourceTokens": translated})
        combined.extend(translated)
    proposal = propose_label_grid(image, combined)
    if proposal.get("layoutHint"):
        proposal["sourceLabelGrid"]["localRecognition"] = {
            "version": 1, "method": "sparse-source-proposal-image-only-local-ocr-v1",
            "sourceRasterSha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "originalFailureReason": original_proposal["failureReason"],
            "originalAnchors": tokens, "proposalFit": projected["fit"],
            "crops": observations, "expectedNamesProvidedToRecognizer": False,
            "strictGridRequired": True, "truthUsed": False,
        }
    return proposal


def source_label_tile_bounds(width: int, height: int) -> list[list[int]]:
    """Nine fixed overlapping half-page windows, independent of observed labels."""
    return [[round(x*width/4), round(y*height/4),
             round((x+2)*width/4), round((y+2)*height/4)]
            for y in range(3) for x in range(3)]


def valid_tiled_label_grid(image: np.ndarray, grid: dict[str, Any]) -> bool:
    """Bind tiled observations and their strict fitted grid to actual source pixels."""
    proof = grid.get("tiledRecognition")
    if proof is None:
        return True
    try:
        h, w = image.shape[:2]
        if (not isinstance(proof, dict) or grid.get("localRecognition") is not None
                or proof.get("version") != 1
                or proof.get("method") != "fixed-half-page-source-tiles-image-only-ocr-v1"
                or proof.get("sourceRasterSha256") != hashlib.sha256(image.tobytes()).hexdigest()
                or not isinstance(proof.get("originalFailureReason"), str) or not proof["originalFailureReason"]
                or proof.get("expectedNamesProvidedToRecognizer") is not False
                or proof.get("strictGridRequired") is not True or proof.get("truthUsed") is not False):
            return False
        original = proof.get("originalAnchors")
        crops = proof.get("crops")
        if not isinstance(original, list) or not isinstance(crops, list) or len(crops) != 9:
            return False
        if any(not _valid_local_label_token(t, w, h) or t.get("lead") not in STANDARD_LEADS for t in original):
            return False
        combined = list(original)
        for index, (crop, bounds) in enumerate(zip(crops, source_label_tile_bounds(w, h), strict=True)):
            left, top, right, bottom = bounds
            if (crop.get("tileRow") != index//3 or crop.get("tileColumn") != index%3
                    or crop.get("bounds") != bounds or not (0 <= left < right <= w and 0 <= top < bottom <= h)
                    or crop.get("rasterSha256") != hashlib.sha256(image[top:bottom, left:right].tobytes()).hexdigest()):
                return False
            local = crop.get("localTokens")
            if not isinstance(local, list) or any(not isinstance(t, dict) or t.get("lead") not in STANDARD_LEADS for t in local):
                return False
            translated = _translate_local_label_observations(local, bounds)
            if translated != crop.get("sourceTokens"):
                return False
            combined.extend(translated)
        strict = propose_label_grid(image, combined)
        return bool(strict.get("layoutHint") and strict["sourceLabelGrid"] ==
                    {k: v for k, v in grid.items() if k != "tiledRecognition"})
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def recover_tiled_label_grid(image, tokens, original, *, ocr_runner=None):
    """A bounded second context for recognition; every name still needs full checks."""
    h, w = image.shape[:2]
    combined, crops = list(tokens), []
    for index, bounds in enumerate(source_label_tile_bounds(w, h)):
        left, top, right, bottom = bounds
        if not (0 <= left < right <= w and 0 <= top < bottom <= h):
            return original
        raster = image[top:bottom, left:right].copy()
        digest = hashlib.sha256(raster.tobytes()).hexdigest()
        local = read_label_anchors(raster, ocr_runner)
        translated = _translate_local_label_observations(local, bounds)
        crops.append({"tileRow": index//3, "tileColumn": index%3, "bounds": bounds,
                      "rasterSha256": digest, "localTokens": local, "sourceTokens": translated})
        combined.extend(translated)
    proposal = propose_label_grid(image, combined)
    if proposal.get("layoutHint"):
        grid = proposal["sourceLabelGrid"]
        grid["tiledRecognition"] = {
            "version": 1, "method": "fixed-half-page-source-tiles-image-only-ocr-v1",
            "sourceRasterSha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "originalFailureReason": original["failureReason"], "originalAnchors": tokens,
            "crops": crops, "expectedNamesProvidedToRecognizer": False,
            "strictGridRequired": True, "truthUsed": False,
        }
        if not valid_tiled_label_grid(image, grid):
            return original
    return proposal


def _complete_source_label_geometry(image, proposal, *, semantic_runner=None):
    if not proposal.get("layoutHint"):
        return proposal
    verified = attach_semantic_lead_identity(image, proposal, ocr_runner=semantic_runner)
    if not (verified.get("leadLabelValidation", {}).get("semanticIdentityConfirmed")
            and verified.get("rhythmLeadValidation", {}).get("semanticIdentityConfirmed")):
        return {**verified, "layoutHint": None, "confidence": 0.0,
                "failureReason": "source-label-values-unconfirmed"}
    centers = supported_trace_rows(image, proposal["sourceLabelGrid"])
    if centers is None:
        return {**verified, "layoutHint": None, "confidence": 0.0,
                "failureReason": "source-waveform-rows-unconfirmed"}
    return {**verified, "rowCenters": centers, "medianRowSpacing": float(np.median(np.diff(centers))),
            "verifiedRhythmLead": "II"}


def _complete_tiled_source_label_geometry(image, tokens, ordinary, *, ocr_runner=None, semantic_runner=None):
    errors = (OSError, RuntimeError, ValueError, subprocess.SubprocessError, json.JSONDecodeError, cv2.error)
    try:
        tiled = recover_tiled_label_grid(image, tokens, ordinary, ocr_runner=ocr_runner)
        recovered = _complete_source_label_geometry(image, tiled, semantic_runner=semantic_runner)
        if recovered.get("layoutHint"):
            return recovered
    except errors:
        pass
    return ordinary


def _complete_deferred_source_label_geometry(image, tokens, ordinary, *, ocr_runner=None, semantic_runner=None):
    """Complete every maintained local/tiled gate from this invocation's source tokens."""
    errors = (OSError, RuntimeError, ValueError, subprocess.SubprocessError, json.JSONDecodeError, cv2.error)
    try:
        proposal = propose_label_grid(image, tokens)
        if not proposal.get("layoutHint"):
            proposal = recover_local_label_grid(image, tokens, proposal, ocr_runner=ocr_runner)
            ordinary = _complete_source_label_geometry(image, proposal, semantic_runner=semantic_runner)
    except errors:
        return {"layoutHint": None, "method": METHOD, "confidence": 0.0,
                "failureReason": "source-label-recognizer-unavailable-or-invalid"}
    if ordinary.get("layoutHint"):
        return ordinary
    return _complete_tiled_source_label_geometry(image, tokens, ordinary, ocr_runner=ocr_runner, semantic_runner=semantic_runner)


def detect_source_label_geometry(image: np.ndarray, *, ocr_runner=None, semantic_runner=None, recover_tiles=True, recover_local=True) -> dict[str, Any]:
    errors = (OSError, RuntimeError, ValueError, subprocess.SubprocessError, json.JSONDecodeError, cv2.error)
    try:
        tokens = read_label_anchors(image, ocr_runner)
        proposal = propose_label_grid(image, tokens)
        if recover_local and not proposal.get("layoutHint"):
            proposal = recover_local_label_grid(image, tokens, proposal, ocr_runner=ocr_runner)
        ordinary = _complete_source_label_geometry(image, proposal, semantic_runner=semantic_runner)
    except errors:
        return {"layoutHint": None, "method": METHOD, "confidence": 0.0,
                "failureReason": "source-label-recognizer-unavailable-or-invalid"}
    if ordinary.get("layoutHint"):
        return ordinary
    if not recover_tiles:
        return ordinary
    return _complete_tiled_source_label_geometry(image, tokens, ordinary, ocr_runner=ocr_runner, semantic_runner=semantic_runner)
