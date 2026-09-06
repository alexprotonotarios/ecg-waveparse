import unittest
import numpy as np
import cv2
from ecg_pipeline.decoder_backends import TraceCrop, decode_crop
from ecg_pipeline.local_grid_warp import VerticalGridWarp, detect_vertical_grid_warp, apply_vertical_grid_warp
from ecg_pipeline.source_verification import verify_source_trace


class EngineeringExperimentsTests(unittest.TestCase):
    def test_backend_identity_gaps_and_ambiguity_are_preserved(self):
        probability=np.zeros((40,80));probability[20,:]=1;probability[10:21,30]=1;probability[:,40:44]=0
        crop=TraceCrop(probability,probability>0,"a"*64,"I:panel:0",4,8)
        for backend in ("upstream-probability-centroid","waveparse-probability-ridge","experimental-direction-connected-v1"):
            result=decode_crop(crop,backend)
            self.assertTrue(np.isnan(result.y_pixels[40:44]).all())
            self.assertEqual(result.source_sha256,crop.source_sha256)
            self.assertEqual(result.segment_id,crop.segment_id)
            self.assertEqual(result.acceptance,"not_evaluated")
        self.assertTrue(decode_crop(crop,"experimental-direction-connected-v1").ambiguous_columns[30])
        with self.assertRaises(ValueError):decode_crop(TraceCrop(probability,probability>0,"a"*64,"I",0,8),backend)
        with self.assertRaises(ValueError):decode_crop(crop,"unknown")

    def test_warp_refuses_folds_extrapolation_and_unsupported_local_scale(self):
        x=np.linspace(0,100,7);reference=np.array([0.,20.,40.]);observed=reference[:,None]+np.sin(x/100)[None,:]
        warp=VerticalGridWarp(x,observed,reference)
        points=np.column_stack((x,np.full(len(x),20.)))
        self.assertLess(np.max(np.abs(warp.map(warp.map(points,inverse=True))-points)),1e-10)
        with self.assertRaises(ValueError):warp.map(np.array([[101,20]]))
        with self.assertRaises(ValueError):warp.map(np.array([[50,100]]))
        with self.assertRaises(ValueError):VerticalGridWarp(x,observed[::-1],reference)
        with self.assertRaises(ValueError):VerticalGridWarp(x,reference[:,None]+x[None,:],reference)
        changed=observed.copy();changed[1,3]+=5
        with self.assertRaises(ValueError):VerticalGridWarp(x,changed,reference)
        for invalid in (float('nan'),float('inf'),-1,0):
            with self.assertRaises(ValueError):VerticalGridWarp(x,observed,reference,max_slope=invalid)
        for invalid in (float('nan'),float('inf'),-1):
            with self.assertRaises(ValueError):VerticalGridWarp(x,observed,reference,max_row_residual_pixels=invalid)
        with self.assertRaises(ValueError):VerticalGridWarp(x,observed,np.array(1.0))

    def test_verifier_does_not_connect_across_gaps_or_erase_ambiguous_thin_ink(self):
        source=np.full((40,80),255,np.uint8);source[20,:]=0
        points=np.column_stack((np.arange(80),np.full(80,20.)));points[30:50]=np.nan
        region=np.ones(source.shape,bool)
        result=verify_source_trace(source,points,region=region,tolerance_pixels=1)
        self.assertGreater(result.summary['omittedFraction'],.1)
        self.assertEqual(result.summary['unsupportedFraction'],0)
        ambiguous=np.zeros_like(region);ambiguous[20,30:50]=True
        limited=verify_source_trace(source,np.column_stack((np.arange(80),np.full(80,20.))),region=region,ambiguous=ambiguous)
        self.assertEqual(limited.summary['ambiguousPathPoints'],20)
        self.assertEqual(limited.summary['unsupportedFraction'],0)
        self.assertIn('ambiguous_source_overlap',limited.summary['reasons'])
        self.assertFalse(limited.summary['attributionComplete'])
        with self.assertRaises(ValueError):verify_source_trace(source,points,region=region,coordinate_space="cleaned")
        with self.assertRaises(ValueError):verify_source_trace(source,np.array([[np.nan,1]]),region=region)

    def test_detected_grid_mapping_preserves_source_and_refuses_missing_support(self):
        image=np.full((240,600,3),255,np.uint8)
        image[::4,:]=(200,200,255);image[:,::4]=(200,200,255)
        yy,xx=np.mgrid[:240,:600].astype(np.float32)
        displacement=7*np.sin(np.pi*xx/599)**2
        curved=cv2.remap(image,xx,yy-displacement,cv2.INTER_LINEAR,borderValue=(255,255,255))
        original=curved.copy()
        warp=detect_vertical_grid_warp(curved)
        points=np.column_stack((np.arange(600),120+displacement[0]))
        self.assertLess(np.sqrt(np.mean((warp.map(points)[:,1]-120)**2)),.25)
        corrected,support=apply_vertical_grid_warp(curved,warp)
        self.assertTrue(np.array_equal(curved,original))
        self.assertEqual(corrected.shape,curved.shape)
        self.assertTrue(np.all(corrected[~support]==255))
        self.assertTrue(warp.evidence['gridNodesIndependentlyDetected'])
        self.assertFalse(warp.evidence['absoluteTranslationKnown'])
        self.assertEqual(warp.evidence['transform']['resamplingCount'],1)
        gap=curved.copy();gap[:,200:250]=255
        with self.assertRaises(ValueError):detect_vertical_grid_warp(gap)
        gray=cv2.cvtColor(curved,cv2.COLOR_BGR2GRAY)
        with self.assertRaises(ValueError):detect_vertical_grid_warp(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR))


if __name__=='__main__':unittest.main()
