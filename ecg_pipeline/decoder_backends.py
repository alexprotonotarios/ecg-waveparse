"""Bounded crop experiment interface. Backends cannot assign identity, units or acceptance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from ecg_pipeline.probability_path import trace_probability_path


@dataclass(frozen=True)
class TraceCrop:
    probability: np.ndarray
    mask: np.ndarray
    source_sha256: str
    segment_id: str
    pixels_per_mm_x: float
    pixels_per_mm_y: float

    def validate(self) -> None:
        if self.probability.ndim != 2 or self.mask.shape != self.probability.shape or self.mask.dtype != bool:
            raise ValueError("Trace crop requires matching 2D probability and boolean mask arrays.")
        if not 3 <= min(self.mask.shape) or max(self.mask.shape) > 4096 or self.mask.size > 4_000_000:
            raise ValueError("Experimental crop exceeds its bounded dimensions.")
        if not np.isfinite(self.probability).all() or np.any((self.probability < 0) | (self.probability > 1)):
            raise ValueError("Probability must be finite within [0,1].")
        if not all(np.isfinite(v) and v > 0 for v in (self.pixels_per_mm_x,self.pixels_per_mm_y)):
            raise ValueError("Separate positive X/Y physical scales are required.")
        if len(self.source_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.source_sha256) or not self.segment_id:
            raise ValueError("Immutable source and segment identity are required.")


@dataclass(frozen=True)
class TraceCandidate:
    backend: str
    source_sha256: str
    segment_id: str
    y_pixels: np.ndarray
    ambiguous_columns: np.ndarray
    evidence_family: str
    acceptance: str = "not_evaluated"


class DecoderBackend(Protocol):
    def __call__(self, crop: TraceCrop) -> TraceCandidate: ...


def direction_connected_path(crop: TraceCrop) -> tuple[np.ndarray, np.ndarray]:
    """Second-order graph path with physical slope and connected-component evidence.

    It retains column run endpoints before path selection. Multiple branches stay
    flagged ambiguous. Empty columns divide paths; they are never interpolated.
    This experiment is not selected by the production candidate planner.
    """
    probability, mask = crop.probability, crop.mask
    height, width = mask.shape
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    path = np.full(width,np.nan)
    ambiguous = np.zeros(width,dtype=bool)
    starts = np.flatnonzero(np.diff(np.pad(mask.any(axis=0).astype(int),(1,1))) == 1)
    ends = np.flatnonzero(np.diff(np.pad(mask.any(axis=0).astype(int),(1,1))) == -1)
    for start,end in zip(starts,ends,strict=True):
        states=[]
        for x in range(start,end):
            indices=np.flatnonzero(mask[:,x])
            runs=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
            ambiguous[x] = len(runs)>1 or any(run.size/crop.pixels_per_mm_y > 1 for run in runs)
            # Keep connected stroke endpoints and probability maximum; bound graph size.
            candidates=sorted(set(int(v) for run in runs for v in (run[0],run[-1],run[np.argmax(probability[run,x])])))
            if len(candidates)>32:
                candidates=sorted(candidates,key=lambda y:(-probability[y,x],y))[:32]
                ambiguous[x]=True
            states.append(np.asarray(sorted(candidates),dtype=int))
        if len(states)==1:
            path[start]=states[0][np.argmax(probability[states[0],start])]
            continue
        # State contains two adjacent rows. Acceleration penalises changes of
        # direction in physical space, independent of arbitrary pixel density.
        first,second=states[:2]
        slope=(second[None,:]-first[:,None])*crop.pixels_per_mm_x/crop.pixels_per_mm_y
        costs=-np.log(np.maximum(probability[first,start],1e-6))[:,None]-np.log(np.maximum(probability[second,start+1],1e-6))[None,:]+.002*np.abs(slope)
        back=[]
        for i in range(2,len(states)):
            prior,current,next_rows=states[i-2:i+1]
            old_slope=(current[None,:]-prior[:,None])*crop.pixels_per_mm_x/crop.pixels_per_mm_y
            new_slope=(next_rows[None,:]-current[:,None])*crop.pixels_per_mm_x/crop.pixels_per_mm_y
            acceleration=np.abs(new_slope[None,:,:]-old_slope[:,:,None])
            shared=labels[current,start+i-1][:,None]==labels[next_rows,start+i][None,:]
            edge=.002*np.abs(new_slope)+.03*acceleration+.5*(~shared)[None,:,:]
            total=costs[:,:,None]+edge
            chosen=np.argmin(total,axis=0)
            costs=np.min(total,axis=0)-np.log(np.maximum(probability[next_rows,start+i],1e-6))[None,:]
            back.append(chosen)
        a,b=np.unravel_index(np.argmin(costs),costs.shape)
        chosen=[int(b),int(a)]
        for references in reversed(back):
            a,b=int(references[a,b]),a
            chosen.append(a)
        chosen.reverse()
        for i,j in enumerate(chosen):path[start+i]=states[i][j]
    return path,ambiguous


def decode_crop(crop: TraceCrop, backend: str) -> TraceCandidate:
    crop.validate()
    probability,mask=crop.probability,crop.mask
    ambiguous=np.sum(np.diff(np.pad(mask.astype(np.int8),((1,0),(0,0))),axis=0)==1,axis=0)>1
    if backend == "upstream-probability-centroid":
        # Same weighted-column operation as upstream; no model inference is implied.
        masked=np.where(mask,probability,0)
        path=np.sum(masked*np.arange(mask.shape[0])[:,None],axis=0)/np.maximum(masked.sum(axis=0),1e-6)
        path[path < 1/mask.shape[0]]=np.nan
    elif backend == "waveparse-probability-ridge":
        path=trace_probability_path(probability,mask,continuity_weight=.001,jump_weight=.000005,max_expected_jump=96)
    elif backend == "native-connected-ink":
        from ecg_pipeline.native_grid_digitizer import trace_crossing_path
        _,path=trace_crossing_path(np.where(mask,probability,0),y_start=0,y_end=mask.shape[0],x_start=0,x_end=mask.shape[1],row_center=mask.shape[0]/2,row_spacing=mask.shape[0],ink_connected_displacement_reward=.03)
        path=path.astype(float)
    elif backend == "experimental-direction-connected-v1":
        path,ambiguous=direction_connected_path(crop)
    else:
        raise ValueError("Unknown bounded decoder backend.")
    path=np.asarray(path,dtype=float)
    path[~mask.any(axis=0)]=np.nan
    if path.shape!=(mask.shape[1],) or np.any(np.isfinite(path)&((path<0)|(path>=mask.shape[0]))):
        raise ValueError("Backend returned an invalid crop path.")
    return TraceCandidate(backend,crop.source_sha256,crop.segment_id,path,ambiguous,"shared-input-mask")
