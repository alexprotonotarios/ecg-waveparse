import unittest
from scripts.build_pmcardio_truth import recording_identity,waveform_key,source_acquisition


class PMCardioIdentityTests(unittest.TestCase):
    def test_digital_augmentation_is_not_a_physical_capture(self):
        self.assertEqual(source_acquisition('augmentation_resolution_scale_down_factor_8'),'digital_augmentation')
        self.assertEqual(source_acquisition('digital_data_high_freq_noise_large'),'digital_export')
        self.assertEqual(source_acquisition('photos_scans'),'scan')
        self.assertEqual(source_acquisition('photos_iphone'),'photograph')

    def test_all_noise_variants_share_the_recording_group_before_split(self):
        original='LPAE_123_hr'
        for kind in ('high','low'):
            for level in ('large','middle','low'):
                self.assertEqual(recording_identity(f'{kind}_freq_noise_{level}_{original}'),original)
        self.assertEqual(recording_identity(original),original)
        self.assertEqual(recording_identity('unrecognised_variant_123'),'unrecognised_variant_123')

    def test_explicit_noise_waveform_keys_are_not_prefixed_twice(self):
        original='LPAE_123_hr';noisy=f'high_freq_noise_large_{original}'
        available={original,noisy}
        category='digital_data_high_freq_noise_large'
        self.assertEqual(waveform_key(category,noisy,available),noisy)
        self.assertEqual(waveform_key(category,original,available),noisy)
        with self.assertRaises(KeyError):waveform_key(category,original,{original})


if __name__=='__main__':unittest.main()
