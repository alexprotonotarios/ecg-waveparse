"""Two-sided pixel evidence, independent of waveform plausibility and acceptance."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SourceVerification:
    summary: dict
    local_distances: np.ndarray
    omitted_mask: np.ndarray


def verify_source_trace(source: np.ndarray, points: np.ndarray, *, region: np.ndarray,
                        excluded: dict[str, np.ndarray] | None = None,
                        ambiguous: np.ndarray | None = None,
                        tolerance_pixels: float = 2.0,
                        mask_dependency: str = "independent-source-pixels",
                        coordinate_space: str = "original") -> SourceVerification:
    """Points are source pixel coordinates (x,y), with NaNs separating unavailable intervals.

    Dark source pixels are candidate waveform evidence only within the caller's
    lead region. Text/grid/pulse/annotation masks must have independent provenance;
    unknown masks make attribution incomplete. This never fills missing samples.
    """
    if coordinate_space != "original":
        raise ValueError("Verification requires points mapped to the untouched original raster.")
    if source.ndim not in (2,3) or (source.ndim==3 and source.shape[2] not in (3,4)) or source.dtype != np.uint8 or source.size > 120_000_000:
        raise ValueError("Expected a bounded original uint8 raster.")
    height,width=source.shape[:2]
    if min(height,width)<3 or region.shape!=(height,width) or region.dtype!=bool:
        raise ValueError("Lead region must be a matching boolean mask.")
    if not np.isfinite(tolerance_pixels) or tolerance_pixels<=0 or tolerance_pixels>20:
        raise ValueError("Source distance tolerance must be in (0,20] pixels.")
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or np.isinf(points).any():
        raise ValueError("Source path must be finite (x,y) points or explicit NaNs.")
    if np.any(np.isfinite(points).any(axis=1) != np.isfinite(points).all(axis=1)):
        raise ValueError("A missing source point must mark both coordinates missing.")
    masks=excluded or {}
    excluded_mask=np.zeros_like(region)
    for mask in list(masks.values())+([ambiguous] if ambiguous is not None else []):
        if mask.shape!=region.shape or mask.dtype!=bool:
            raise ValueError("Exclusion and ambiguity masks must match original dimensions.")
    for mask in masks.values():excluded_mask|=mask
    ambiguous_mask=np.zeros_like(region) if ambiguous is None else ambiguous
    if source.ndim==3:
        channels=source[:,:,:3].astype(float)
        dark=(np.max(channels,axis=2)<170)&((np.max(channels,axis=2)-np.min(channels,axis=2))<45)
    else:dark=source<170
    evidence=dark&region&~excluded_mask&~ambiguous_mask
    distance_to_ink=cv2.distanceTransform((~evidence).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE) if evidence.any() else np.full((height,width),np.inf)
    valid=np.isfinite(points).all(axis=1)
    rounded=np.zeros(points.shape,dtype=np.int64)
    rounded[valid]=np.rint(points[valid]).astype(np.int64)
    inside=valid&(rounded[:,0]>=0)&(rounded[:,0]<width)&(rounded[:,1]>=0)&(rounded[:,1]<height)
    path_mask=np.zeros((height,width),dtype=np.uint8)
    # Never draw a line over an absent interval or through out-of-image coordinates.
    for start,end in zip(np.flatnonzero(np.diff(np.pad(inside.astype(int),(1,1)))==1),np.flatnonzero(np.diff(np.pad(inside.astype(int),(1,1)))==-1),strict=True):
        if end-start>1:cv2.polylines(path_mask,[rounded[start:end].astype(np.int32)],False,1,1)
        else:path_mask[rounded[start,1],rounded[start,0]]=1
    local=np.full(len(points),np.nan)
    local[valid&~inside]=np.inf
    local[inside]=distance_to_ink[rounded[inside,1],rounded[inside,0]]
    distance_to_path=cv2.distanceTransform((path_mask==0).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE) if path_mask.any() else np.full((height,width),np.inf)
    omitted=evidence&(distance_to_path>tolerance_pixels)
    inside_region=np.zeros(len(points),dtype=bool)
    inside_region[inside]=region[rounded[inside,1],rounded[inside,0]]
    excluded_points=np.zeros(len(points),dtype=bool)
    excluded_points[inside]=excluded_mask[rounded[inside,1],rounded[inside,0]]
    ambiguous_points=np.zeros(len(points),dtype=bool)
    ambiguous_points[inside]=ambiguous_mask[rounded[inside,1],rounded[inside,0]]
    assessable=valid&~ambiguous_points
    unsupported=assessable&((local>tolerance_pixels)|~inside_region|excluded_points)
    count=int(np.count_nonzero(assessable));ink_count=int(np.count_nonzero(evidence))
    n,labels,stats,_=cv2.connectedComponentsWithStats(evidence.astype(np.uint8),connectivity=8)
    omitted_counts=np.bincount(labels[omitted],minlength=n)
    component_ids=np.argsort(stats[1:,cv2.CC_STAT_AREA])[::-1][:1000]+1
    components=[{"pixels":int(stats[i,cv2.CC_STAT_AREA]),"omittedFraction":float(omitted_counts[i]/stats[i,cv2.CC_STAT_AREA])} for i in component_ids]
    reasons=[]
    if unsupported.any():reasons.append("trace_without_visible_waveform_support")
    if omitted.any():reasons.append("visible_source_structure_omitted")
    if np.any(valid&~inside_region):reasons.append("trace_outside_lead_region")
    if excluded_points.any():reasons.append("trace_follows_excluded_nonwaveform_marks")
    if ambiguous_points.any():reasons.append("ambiguous_source_overlap")
    attribution_complete=set(masks)>={"text","grid","calibration","annotation"}
    return SourceVerification({"version":1,"method":"two-sided-original-dark-ink-distance-v1","coordinateSpace":"original",
        "maskDependency":mask_dependency,"attributionComplete":attribution_complete,"tolerancePixels":tolerance_pixels,
        "assessablePathPoints":count,"ambiguousPathPoints":int(np.count_nonzero(ambiguous_points)),
        "unsupportedPathPoints":int(np.count_nonzero(unsupported)),"unsupportedFraction":float(np.count_nonzero(unsupported)/count) if count else None,
        "visibleInkPixels":ink_count,"omittedInkPixels":int(np.count_nonzero(omitted)),"omittedFraction":float(np.count_nonzero(omitted)/ink_count) if ink_count else None,
        "componentCount":n-1,"componentDetailsTruncated":n-1>1000,"components":components,"reasons":reasons,"publicationPolicyChanged":False,"calibrationAndIdentityVerified":False},local,omitted)
