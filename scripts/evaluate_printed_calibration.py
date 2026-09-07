"""Exercise the actual provisioned OCR backend, not a mocked text response."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from detect_ecg_layout import detect_ecg_calibration
from ecg_pipeline.printed_calibration import read_printed_settings, reconcile_printed_settings


def fixture(speed: float, gain: float, printed_speed: float | None, printed_gain: float | None):
    image = np.full((320, 1100, 3), 255, np.uint8)
    for x in range(0, 1100, 4):
        cv2.line(image, (x, 80), (x, 319), (210, 210, 255), 1)
    for y in range(80, 320, 4):
        cv2.line(image, (0, y), (1099, y), (210, 210, 255), 1)
    x0, x1, baseline, top = 40, 40 + round(speed * .2 * 4), 220, 220 - round(gain * 4)
    cv2.polylines(image, [np.array([(10, baseline), (x0, baseline), (x0, top), (x1, top), (x1, baseline), (x1+35, baseline)])], False, (0, 0, 0), 2)
    if printed_speed is not None and printed_gain is not None:
        text = f"{printed_speed:g} mm/s    {printed_gain:g} mm/mV"
        cv2.putText(image, text, (240, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 2, cv2.LINE_AA)
    return image


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    protocol_path=ROOT/"benchmark/protocols/printed-calibration.v1.json"
    protocol=json.loads(protocol_path.read_text())
    cases=[(f"speed_{speed}_gain_{gain}",speed,gain,speed,gain,"corroborated_inference")
           for speed in (25,50) for gain in (5,10,20)]
    cases += [("speed_conflict",25,10,50,10,"conflict"), ("gain_conflict",25,10,25,20,"conflict"),
              ("unsupported_speed",25,10,12.5,10,"unsupported"), ("no_settings",25,10,None,None,"unresolved")]
    results=[]
    for case_id,speed,gain,printed_speed,printed_gain,expected in cases:
        image=fixture(speed,gain,printed_speed,printed_gain)
        source=args.output/f"{case_id}.png"
        if source.exists():
            raise RuntimeError("Refusing to overwrite a previous fixture result; choose a new output directory.")
        assert cv2.imwrite(str(source),image)
        started=time.perf_counter()
        calibration=detect_ecg_calibration(image)["calibration"]
        printed=read_printed_settings(image)
        printed["sourceSha256"]=hashlib.sha256(source.read_bytes()).hexdigest()
        reconciled=reconcile_printed_settings(calibration,printed)
        elapsed=time.perf_counter()-started
        expected_values={"speed": [] if printed_speed is None else [printed_speed], "gain": [] if printed_gain is None else [printed_gain]}
        checks={
            "pulseDetected":calibration["detected"],
            "pulseSpeed":calibration.get("paperSpeedMmPerSecond")==speed,
            "pulseGain":calibration.get("gainMmPerMv")==gain,
            "printedValues":printed["values"]==expected_values,
            "reconciliation":reconciled["reconciliation"]["state"]==expected,
            "quantitativeBlocked":reconciled["reconciliation"]["quantitativeBlocked"]==(expected in {"conflict","unsupported"}),
            "horizontalScale":abs(calibration.get("pixelsPerMmX",0)-4)<=protocol["maximumScaleErrorPixelsPerMm"],
            "verticalScale":abs(calibration.get("pixelsPerMmY",0)-4)<=protocol["maximumScaleErrorPixelsPerMm"],
            "runtime":elapsed<=protocol["maximumSecondsPerFixture"],
        }
        result={"caseId":case_id,"sourceSha256":printed["sourceSha256"],"expectedState":expected,
                "seconds":elapsed,"checks":checks,"passed":all(checks.values()),"calibration":reconciled}
        results.append(result)
        print(json.dumps({"caseId":case_id,"passed":result["passed"],"checks":checks}),flush=True)
    report={"version":1,"clinicalValidationUse":False,"protocolSha256":hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "ocrSelection":os.environ.get("ECG_DIGITIZER_LABEL_OCR_ENGINE","auto"),"attempted":len(results),
            "passed":sum(r["passed"] for r in results),"cases":results}
    (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return 0 if report["passed"]==len(results) else 1


if __name__=="__main__":
    raise SystemExit(main())
