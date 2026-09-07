import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.run_platform_matrix import difference, uncertainty_difference


class PlatformComparisonTests(unittest.TestCase):
    def test_numeric_comparison_keeps_missingness_and_shape_separate(self):
        result=difference({'I':np.array([1.,np.nan,3.])},{'I':np.array([1.,2.,4.])})
        self.assertEqual(result['missingnessDifferences'],1)
        self.assertEqual(result['comparedSamples'],2)
        self.assertEqual(result['maximumAbsoluteDifferenceUv'],1)
        self.assertAlmostEqual(result['rmseUv'],np.sqrt(.5))
        self.assertFalse(difference({'I':np.array([1.])},{'I':np.array([1.,1.])})['shapeMatches'])

    def test_uncertainty_comparison_includes_unmatched_locations_and_candidate_counts(self):
        header='lead,canonical_sample,status,candidate_count,candidate_spread_uv\n'
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'a.csv',Path(directory)/'b.csv'
            a.write_text(header+'I,0,observed,2,1\nI,1,missing,1,0\n')
            b.write_text(header+'I,0,uncertain,3,4\nI,2,missing,1,0\n')
            result=uncertainty_difference(a,b)
            self.assertEqual(result['locationDifferences'],2)
            self.assertEqual(result['statusDifferences'],3)
            self.assertEqual(result['candidateCountDifferences'],3)
            self.assertEqual(result['maximumAbsoluteSpreadDifferenceUv'],3)
            b.write_text(header+'I,0,observed,2,1\nI,0,observed,2,1\n')
            with self.assertRaisesRegex(ValueError,'Duplicate'):uncertainty_difference(a,b)
            b.write_text('lead,status\nI,observed\n')
            with self.assertRaisesRegex(ValueError,'schema'):uncertainty_difference(a,b)


if __name__=='__main__':unittest.main()
