"""Research-only adapter for a pinned 12-lead nnUNet segmentation checkpoint.

This module is never imported by production extraction or shipped in runtime
payloads. It uses the checkpoint's plan, channel convention, normalization,
patch size and upstream sliding-window utilities. The bounded comparison uses
one fold, float32 accumulation and no test-time mirroring, all explicit below.
No acceptance, physical calibration or lead relabelling is performed here.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pydoc
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CHECKPOINT_SHA256='d0e0d4c578d457367d3f78ae3739cf86a8d8524cd4f669c24140184e8cce5355'
PLANS_SHA256='ef1cbaecb2ef899c376e50345c8deb4ebd40dbef63d941f6331b726563487f36'


def digest_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):digest.update(chunk)
    return digest.hexdigest()


class PretrainedLeadHeatmap:
    def __init__(self, comparison_root: Path, plans_path: Path, *, device: str='cpu'):
        if device not in {'cpu','mps'}:raise ValueError('Only explicitly measured local devices are admitted.')
        sys.path.insert(0,str(comparison_root/'deps'))
        sys.path.insert(0,str(comparison_root/'upstream/nnUNet'))
        import torch
        from dynamic_network_architectures.architectures.unet import PlainConvUNet
        from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        self.torch=torch;torch.set_num_threads(2)
        self.device=torch.device(device)
        if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('Requested MPS device unavailable.')
        checkpoint=comparison_root/'checkpoint_final.pth'
        if digest_file(checkpoint)!=CHECKPOINT_SHA256 or digest_file(plans_path)!=PLANS_SHA256:
            raise ValueError('Unpinned comparison artifact identity.')
        # Keep restricted deserialization. Only numeric NumPy metadata types
        # are allowed; no arbitrary checkpoint globals or executable callbacks.
        allowed={'numpy.dtype','numpy.core.multiarray.scalar'}
        if not set(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint))<=allowed:
            raise ValueError('Unexpected checkpoint deserialization globals.')
        numeric_globals=[(np._core.multiarray.scalar,'numpy.core.multiarray.scalar'),
                         np.dtype,np.dtypes.Float64DType,np.dtypes.Float32DType]
        with torch.serialization.safe_globals(numeric_globals):
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        configuration=json.loads(plans_path.read_text())['configurations']['2d']
        architecture=configuration['architecture']
        if architecture['network_class_name']!='dynamic_network_architectures.architectures.unet.PlainConvUNet':
            raise ValueError('Unexpected architecture.')
        kwargs=architecture['arch_kwargs'].copy()
        for key in architecture['_kw_requires_import']:
            if kwargs[key] is not None:
                if not kwargs[key].startswith('torch.nn.'):raise ValueError('Unexpected architecture import.')
                kwargs[key]=pydoc.locate(kwargs[key])
        kwargs['deep_supervision']=False
        self.network=PlainConvUNet(input_channels=1,num_classes=13,**kwargs)
        self.network.load_state_dict(state['network_weights'],strict=True)
        del state;gc.collect()
        self.network.eval().to(self.device)
        self.patch=tuple(configuration['patch_size'])
        self.normalize=ZScoreNormalization(use_mask_for_norm=False,intensityproperties={})
        self.gaussian=compute_gaussian;self.steps=compute_steps_for_sliding_window;self.pad=pad_nd_image
        self.evidence={'version':1,'backend':'experimental-felix-m1-per-lead-segmentation-v1',
            'checkpointSha256':CHECKPOINT_SHA256,'plansSha256':PLANS_SHA256,
            'upstreamCommit':'e6f62aa776f105e4c7b04f21669da4d4f0df370b',
            'device':device,'torchVersion':torch.__version__,'threads':2,'fold':0,
            'testTimeMirroring':False,'accumulatorDtype':'float32','patchSize':list(self.patch),
            'tileStepSize':.5,'channels':'13 semantic softmax channels: background, I, II, III, aVR, aVL, aVF, V1-V6',
            'inputConvention':'Upstream RGB-as-slices; global RGB z-score, first (red) output slice',
            'acceptance':'not_evaluated','calibration':'not_established','productionEnabled':False,
            'upstreamReleaseParityClaimed':False}

    def predict(self,bgr: np.ndarray) -> np.ndarray:
        torch=self.torch
        if bgr.ndim!=3 or bgr.shape[2]!=3 or bgr.dtype!=np.uint8 or bgr.shape[0]*bgr.shape[1]>4_000_000:
            raise ValueError('Comparison accepts BGR uint8 images up to four megapixels.')
        # This deliberately follows the upstream NaturalImage2DIO convention,
        # not a guessed RGB-to-grey conversion or a resized network canvas.
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).transpose(2,0,1).astype(np.float32)
        normalized=self.normalize.run(rgb)[0]
        padded,slicer=self.pad(torch.from_numpy(normalized)[None],self.patch,'constant',{'value':0},True)
        h,w=padded.shape[1:];steps=self.steps((h,w),self.patch,.5)
        sums=torch.zeros((13,h,w),dtype=torch.float32)
        counts=torch.zeros((h,w),dtype=torch.float32)
        gaussian=self.gaussian(self.patch,value_scaling_factor=10,dtype=torch.float32,device=torch.device('cpu'))
        with torch.inference_mode():
            for y in steps[0]:
                for x in steps[1]:
                    crop=padded[:,y:y+self.patch[0],x:x+self.patch[1]][None].to(self.device)
                    logits=self.network(crop)[0].float().cpu()
                    if not torch.isfinite(logits).all():raise RuntimeError('Nonfinite learned heatmap.')
                    sums[:,y:y+self.patch[0],x:x+self.patch[1]]+=logits*gaussian
                    counts[y:y+self.patch[0],x:x+self.patch[1]]+=gaussian
                    del crop,logits
        # The input has one channel; its padding slicer must not truncate the
        # thirteen output classes back to that input-channel dimension.
        result=torch.softmax((sums/counts)[(slice(None),*slicer[1:])],dim=0).numpy()
        if result.shape!=(13,*bgr.shape[:2]) or not np.isfinite(result).all():raise RuntimeError('Invalid heatmap shape.')
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--comparison-root',type=Path,required=True)
    parser.add_argument('--plans',type=Path,required=True);parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    model=PretrainedLeadHeatmap(args.comparison_root,args.plans,device=args.device);loaded=time.perf_counter()
    image=cv2.imread(str(args.input));probability=model.predict(image);finished=time.perf_counter()
    np.savez_compressed(args.output/'heatmaps.npz',probability=probability.astype(np.float16))
    cv2.imwrite(str(args.output/'semantic-labels.png'),np.argmax(probability,axis=0).astype(np.uint8))
    evidence={**model.evidence,'sourceSha256':digest_file(args.input),'loadSeconds':loaded-started,
              'inferenceSeconds':finished-loaded,'scriptSha256':digest_file(__file__),
              'foregroundPixels':int(np.count_nonzero(np.argmax(probability,axis=0))),
              'heatmapSha256':digest_file(args.output/'heatmaps.npz')}
    (args.output/'report.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence))


if __name__=='__main__':main()
