"""Observe an untouched source and apply supported timing to a selected export."""
import argparse
import hashlib
import io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from . import observe_source_time
from .export import write_publication_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'coordinates', 'canonical', 'uncertainty', 'segments', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ['expected-source-sha256', 'expected-coordinate-sha256']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    source = args.source.read_bytes(); coordinates = args.coordinates.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != args.expected_source_sha256 or hashlib.sha256(coordinates).hexdigest() != args.expected_coordinate_sha256:
        raise ValueError('source_time_file_identity_mismatch')
    with Image.open(io.BytesIO(source)) as header:
        if min(header.size) < 200 or header.width * header.height > 12_000_000:
            raise ValueError('source_time_source_size_unsupported')
    image = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_COLOR)
    observation = observe_source_time(image, source_sha256=digest)
    result = write_publication_bundle(image, observation, coordinates, args.canonical.read_bytes(),
                                     args.uncertainty.read_bytes(), args.segments.read_bytes(), args.output_dir)
    if args.source.read_bytes() != source: raise ValueError('source_changed_during_time_conversion')
    print(result['state'])


if __name__ == '__main__': main()
