import tempfile
import unittest
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from scripts.prepare_ecg_input import inspect_source,load_working_image,validate_source_header


class InputBoundaryTests(unittest.TestCase):
    def test_multipage_tiff_is_not_silently_reduced_to_first_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'two-pages.tif'
            Image.new('RGB',(32,32)).save(path,save_all=True,append_images=[Image.new('RGB',(32,32),'white')])
            original=path.read_bytes()
            for read in (inspect_source,lambda p:load_working_image(p,100)):
                with self.assertRaisesRegex(ValueError,'multiple_frames'):read(path)
            self.assertEqual(path.read_bytes(),original)

    def test_extreme_headers_fail_before_pixels_are_materialized(self):
        class Header:
            format='PNG';n_frames=1;size=(32769,1)
        with self.assertRaisesRegex(ValueError,'dimensions'):validate_source_header(Header())
        Header.size=(10000,9000)
        with self.assertRaisesRegex(ValueError,'dimensions'):validate_source_header(Header())
        Header.size=(1000,500)
        validate_source_header(Header())
        Header.format='GIF'
        with self.assertRaisesRegex(ValueError,'unsupported'):validate_source_header(Header())

    def test_malformed_raster_does_not_modify_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.png';path.write_bytes(b'not a raster')
            with self.assertRaises(UnidentifiedImageError):inspect_source(path)
            self.assertEqual(path.read_bytes(),b'not a raster')


if __name__=='__main__':unittest.main()
