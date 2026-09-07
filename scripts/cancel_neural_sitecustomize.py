"""Test-only profile hook: mark real convolution work inside the pinned UNet.

Copied as sitecustomize.py into an owner-controlled temporary directory for two
cancellation tests. No installed/runtime file is edited, no waveform is changed,
and profiling disables itself after the first matching convolution returns.
"""
import json
import os
import sys
import time

marker=os.environ.get('WAVEPARSE_TEST_FORWARD_MARKER')
arguments=getattr(sys,'orig_argv',[])
if marker and 'src.digitize' in arguments:
    def observe(frame,event,arg):
        if event!='return' or frame.f_code.co_name!='forward' or frame.f_globals.get('__name__')!='torch.nn.modules.conv':return
        parent=frame.f_back
        for _ in range(64):
            if parent is None:return
            if parent.f_code.co_name=='forward' and parent.f_globals.get('__name__')=='src.model.unet':
                sys.setprofile(None)
                try:
                    descriptor=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                    with os.fdopen(descriptor,'w') as handle:
                        json.dump({'version':1,'pid':os.getpid(),'wallTime':time.time(),
                                   'boundary':'first_convolution_completed_inside_unet_forward'},handle)
                except FileExistsError:pass
                return
            parent=parent.f_back
    sys.setprofile(observe)
