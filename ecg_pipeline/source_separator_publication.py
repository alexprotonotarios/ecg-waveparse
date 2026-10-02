"""Conservative publication gaps at independently observed separator columns."""
from __future__ import annotations
import hashlib
from typing import Any
import numpy as np
from ecg_pipeline.source_waveform_support import _separator_exclusion_boxes


def separator_publication_mask(image: np.ndarray, geometry: dict[str, Any],
                                  timing: dict[str, Any], support: np.ndarray):
    """Leave raster/evidence/path intact; remove support only in certified columns."""
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
            or support.shape != image.shape[:2] or support.dtype != np.bool_
            or timing.get('state') != 'source_supported'
            or timing.get('truthUsed') is not False
            or timing.get('decodedRasterSha256') != hashlib.sha256(image.tobytes()).hexdigest()
            or timing.get('imageSize') != [image.shape[1],image.shape[0]]
            or geometry.get('layoutHint') != 'standard_3x4_with_r1'):
        raise ValueError('Separator publication requires identical source/support/timing.')
    height,width=support.shape
    raw_rows=geometry.get('rowCenters')
    if (not isinstance(raw_rows,list) or len(raw_rows)!=4
            or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in raw_rows)):
        raise ValueError('Four ordered source rows are required.')
    rows=np.asarray(raw_rows,dtype=np.float64)
    if not np.isfinite(rows).all() or not np.all(np.diff(rows)>0) or rows[0]<0 or rows[-1]>=height:
        raise ValueError('Four ordered source rows are required.')
    measurements=timing.get('rowGridMeasurements')
    if (not isinstance(measurements,list) or len(measurements)!=4
            or any(not isinstance(m,dict) or m.get('rowCenter')!=rows[i] for i,m in enumerate(measurements))):
        raise ValueError('Source rows must match the measured timing.')
    if not ((timing.get('version')==2 and timing.get('method')=='source-separator-local-grid-timing-v2')
            or (timing.get('version')==3 and timing.get('method')=='source-separator-window-consensus-timing-v3')
            or (timing.get('version')==4 and timing.get('method')=='source-joint-corresponding-grid-row-timing-v4')
            or (timing.get('version')==5 and timing.get('method')=='source-endpoint-family-corresponding-grid-row-timing-v5')):
        raise ValueError('Unsupported source timing method.')
    if timing.get('version') in (4, 5):
        from ecg_pipeline.joint_source_timing import validate_joint_timing_fields
        validate_joint_timing_fields(image, geometry, timing)
    row_bounds=[0,*np.ceil((rows[:-1]+rows[1:])/2).astype(int).tolist(),height]
    family_timing = timing.get('version') == 5
    marks=timing.get('sourceSeparatorFamilies' if family_timing else 'sourceSeparators')
    if not isinstance(marks,list) or len(marks)!=9:
        raise ValueError('All nine ordered source marks are required.')
    checks = [(i, member) for i, family in enumerate(marks) for member in family['members']] if family_timing else enumerate(marks)
    for i,mark in checks:
        if (not isinstance(mark,dict) or type(mark.get('row')) is not int or mark['row']!=i//3
                or type(mark.get('column')) is not int or mark['column']!=i%3+1):
            raise ValueError('Source mark row/column identities must be exact.')
        bounds=mark.get('bounds')
        if not isinstance(bounds,list) or len(bounds)!=4 or any(type(v) is not int for v in bounds):
            raise ValueError('Source marks require integer rectangles.')
        left,top,right,bottom=bounds;row=i//3
        if not (0<=left<right<=width and row_bounds[row]<=top<bottom<=row_bounds[row+1]):
            raise ValueError('Source mark must lie wholly in its own source row.')
        center,minimum=mark.get('centerX'),mark.get('minimumBlackColumnSupport')
        if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in [center,minimum])
                or center!=(left+right-1)/2 or not .8<=minimum<=1):
            raise ValueError('Source mark geometry or support is invalid.')
        actual=float((image[top:bottom,left:right].max(axis=2)<160).mean(axis=0).min())
        if not np.isclose(actual,minimum,rtol=0,atol=1e-12):
            raise ValueError('Source mark support must reproduce the untouched raster.')
    # Reuse the existing observed/consensus footprint validation and one-pixel margin.
    observed_boxes=_separator_exclusion_boxes(image,timing)
    if len(observed_boxes)!=9:raise ValueError('All nine mask footprints are required.')
    mask=np.zeros(support.shape,dtype=bool);entries=[]
    for i,(mark,box) in enumerate(zip(marks,observed_boxes,strict=True)):
        left,top,right,bottom=box['bounds'];row=i//3
        if not (0<=left<right<=width and row_bounds[row]<=top<bottom<=row_bounds[row+1]):
            raise ValueError('Observed footprint must lie wholly in its own source row.')
        bounds=[left,row_bounds[row],right,row_bounds[row+1]]
        mask[bounds[1]:bounds[3],left:right]=True
        entries.append({'row':row,'column':i%3+1,'observedMark':mark.copy(),
                        'existingExclusion':box.copy(),'publicationBounds':bounds})
    retained=support & ~mask
    def sha(a):return hashlib.sha256(a.astype(np.uint8).tobytes()).hexdigest()
    proof={'version':1,'method':'observed-separator-column-publication-gaps-v1',
           'decodedRasterSha256':timing['decodedRasterSha256'],'imageSize':[width,height],
           'sourceTimingVersion':timing['version'],'sourceTimingMethod':timing['method'],
           'rowCenters':rows.tolist(),'rowBounds':row_bounds,'sourceSeparators':entries,
           'horizontalMarginPixels':1,'verticalExtent':'adjacent-source-row-midpoints',
           'publicationValidityOnly':True,'oldConnectedSourceValidityAppliedFirst':True,
           'sourceSupportChanged':False,'sourceImageChanged':False,'sourceEvidenceChanged':False,
           'sourcePathChanged':False,'timeCoordinatesChanged':False,'oldGapsRecovered':0,
           'sourceSupportSha256':sha(support),'sourceSupportOutsideMaskSha256':sha(retained),
           'publicationMaskSha256':sha(mask),'maskPixelCount':int(mask.sum()),
           'newlyExcludedSupportedPixelCount':int((support & mask).sum()),
           'separateRhythmRowUnchanged':bool(np.array_equal(support[row_bounds[3]:],retained[row_bounds[3]:])),
           'truthUsed':False}
    return mask,proof


def apply_separator_publication_validity(mask: np.ndarray, columns: np.ndarray,
                                         path: np.ndarray, validity: np.ndarray):
    """Apply after the old connected-source rule, preserving every old gap."""
    if (mask.ndim!=2 or mask.dtype!=np.bool_ or columns.ndim!=1
            or path.shape!=columns.shape or validity.shape!=columns.shape
            or validity.dtype!=np.bool_ or not np.issubdtype(columns.dtype,np.integer)
            or not np.issubdtype(path.dtype,np.integer)
            or np.any(np.diff(columns)<=0) or np.any(columns<0) or np.any(columns>=mask.shape[1])
            or np.any(path<0) or np.any(path>=mask.shape[0])):
        raise ValueError('Publication validity requires bounded integer paths and boolean masks.')
    return validity & ~mask[path,columns]


def separator_publication_path_evidence(
    mask: np.ndarray,
    columns: np.ndarray,
    path: np.ndarray,
    prior_validity: np.ndarray,
    *,
    row: int,
    row_bounds: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bind the final publication gaps to the unchanged source path and old rule."""
    retained = apply_separator_publication_validity(mask, columns, path, prior_validity)
    if (type(row) is not int or not 0 <= row < 4 or len(row_bounds) != 5
            or row_bounds[0] != 0 or row_bounds[-1] != mask.shape[0]
            or any(type(v) is not int for v in row_bounds)
            or any(a >= b for a, b in zip(row_bounds, row_bounds[1:]))):
        raise ValueError("Publication evidence requires four bounded source rows.")
    removed = prior_validity & ~retained
    if (row == 3 and np.any(removed)) or np.any(
        removed & ((path < row_bounds[row]) | (path >= row_bounds[row + 1]))
    ):
        raise ValueError("Publication gaps must remain in the path's own non-rhythm row.")

    def sha(array: np.ndarray, dtype: str) -> str:
        return hashlib.sha256(array.astype(dtype).tobytes()).hexdigest()

    proof = {
        "row": row,
        "priorValidityRule": "verified-source-transition-gaps-v1",
        "pathEncoding": "int32-le-y-by-source-column-v1",
        "validityEncoding": "uint8-by-source-column-v1",
        "columnsSha256": sha(columns, "<i4"),
        "sourcePathSha256": sha(path, "<i4"),
        "priorValiditySha256": sha(prior_validity, "u1"),
        "publishedValiditySha256": sha(retained, "u1"),
        "sourceColumnCount": int(columns.size),
        "priorValidColumnCount": int(prior_validity.sum()),
        "publishedValidColumnCount": int(retained.sum()),
        "removedSourcePixels": np.column_stack((columns[removed], path[removed])).tolist(),
        "oldGapsRecovered": 0,
        "sourcePathChanged": False,
        "truthUsed": False,
    }
    return retained, proof
