"""Prepare a candidate raster and report the geometry actually used by Pillow."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
from PIL import Image


def prepare(input_path: Path, output_path: Path, *, mode: str, target: int = 0, crop: list[int] | None = None) -> dict:
    if mode not in ('copy', 'png') or type(target) is not int or target < 0:
        raise ValueError('invalid_candidate_preparation')
    if mode == 'copy' and (target or crop is not None):
        raise ValueError('copy_cannot_change_geometry')
    if input_path.resolve() == output_path.resolve():
        raise ValueError('candidate_must_not_overwrite_input')
    encoded = input_path.read_bytes()
    with Image.open(io.BytesIO(encoded)) as opened:
        image = opened.convert('RGB')
    input_size = list(image.size)
    input_raster = hashlib.sha256(image.tobytes()).hexdigest()
    if crop is not None:
        if (not isinstance(crop, list) or len(crop) != 4 or any(type(v) is not int for v in crop)
                or not (0 <= crop[0] < crop[2] <= image.width and 0 <= crop[1] < crop[3] <= image.height)):
            raise ValueError('candidate_crop_outside_input')
        image = image.crop(tuple(crop))
    cropped_size = list(image.size)
    resized = target > 0 and max(image.size) < target
    if resized:
        scale = target / max(image.size)
        size = tuple(max(1, round(value * scale)) for value in image.size)
        image = image.resize(size, Image.Resampling.LANCZOS)
    if mode == 'copy':
        output_path.write_bytes(encoded)
    else:
        image.save(output_path, format='PNG')
    output = output_path.read_bytes()
    # Read back the published raster, rather than merely reporting requested size.
    with Image.open(io.BytesIO(output)) as saved:
        decoded = saved.convert('RGB')
    if decoded.size != image.size or decoded.tobytes() != image.tobytes():
        raise ValueError('candidate_preparation_readback_mismatch')
    if input_path.read_bytes() != encoded:
        raise ValueError('candidate_preparation_input_changed')
    final_size = list(decoded.size)
    sx, sy = final_size[0] / cropped_size[0], final_size[1] / cropped_size[1]
    left, top = crop[:2] if crop is not None else (0, 0)
    return {'version': 1, 'method': 'observed-candidate-preparation-v1', 'coordinateConvention': 'pixel_edges',
            'mode': mode, 'inputSha256': hashlib.sha256(encoded).hexdigest(),
            'preparedSha256': hashlib.sha256(output).hexdigest(), 'inputSizeWh': input_size,
            'croppedSizeWh': cropped_size, 'preparedSizeWh': final_size, 'cropBox': crop, 'requestedMaxDimension': target,
            'resampling': 'pillow_lanczos' if resized else 'none', 'inputRasterRgbSha256': input_raster,
            'preparedRasterRgbSha256': hashlib.sha256(decoded.tobytes()).hexdigest(),
            'inputToPreparedEdges': [[sx, 0, -sx * left], [0, sy, -sy * top], [0, 0, 1]],
            'inputPreserved': True, 'photometricFilterWeightsExported': False, 'waveformResamplingPerformed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['copy', 'png'], required=True)
    parser.add_argument('--target', type=int, default=0)
    parser.add_argument('--crop', type=int, nargs=4)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output, mode=args.mode, target=args.target, crop=args.crop), allow_nan=False))


if __name__ == '__main__':main()
