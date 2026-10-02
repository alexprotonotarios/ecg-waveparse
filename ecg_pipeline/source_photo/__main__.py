"""Local source-photo diagnostic candidate; no production selection side effects."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import MAX_SOURCE_PHOTO_PIXELS, observe_source_photo
from .export import write_diagnostic_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a source-photo candidate for explicit diagnostic review.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--original-source", type=Path, help="Optional untouched source whose deterministic working raster is supplied as --input.")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory must be new.")
    encoded = args.input.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if args.expected_source_sha256 is not None and args.expected_source_sha256 != digest:
        parser.error("Source file identity does not match the expected SHA-256.")
    with Image.open(io.BytesIO(encoded)) as header:
        if min(header.size) < 200 or header.width * header.height > MAX_SOURCE_PHOTO_PIXELS:
            parser.error("Source raster is outside the diagnostic image-size bounds.")
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    result = observe_source_photo(image, source_sha256=digest)
    if args.original_source is not None:
        result["source"]["originalFileSha256"] = hashlib.sha256(args.original_source.read_bytes()).hexdigest()
        result["source"]["workingToOriginalTransformVerified"] = False
    candidate = write_diagnostic_bundle(image, result, args.output_dir)
    print(json.dumps(candidate, allow_nan=False))


if __name__ == "__main__":
    main()
