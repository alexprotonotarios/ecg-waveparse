# Installed-package worked examples

## Current installed evidence inspection

The continuation executed the documented JavaScript and Python examples against
three existing physical-setting runs. Both languages returned identical source
availability, reconstruction state and segment counts. This exercises installed
inspection without extra inference or an acceptance action.

| Constructed input | Extraction truth comparison | Installed inspection |
| --- | --- | --- |
| 25 mm/s, 10 mm/mV | 33.686 µV RMSE, 0.963038 correlation, 99.9567% coverage | `needs_review`, original available, twelve segments |
| 25 mm/s, 20 mm/mV | 30.281 µV RMSE, 0.971831 correlation, 99.9667% coverage | `needs_review`, original available, twelve segments |
| 25 mm/s, 5 mm/mV | Abstention: trusted lead identity unavailable | `failed`, original available, zero exported segments |

These examples use the corrected runtime payload
`e84bef376703e8029ac673cfbc1f85d6cbc5fbb1db608367ca221e7e4c22150b`.
The current minimal source rebuild preserves that payload and both executable
interfaces exactly. The complete physical-setting experiment attempted eight
inputs from one controlled family, returned five and abstained on three; it had
no runtime failures. Selecting these three examples does not replace that
denominator. The 20 mm/mV source, diagnostic overlay and paper render were
visually compared. No example was accepted for quantitative or clinical use.

`completion-v8-examples-receipt.json` and `physical-settings-receipt-v1.json` in
the retained continuation directory bind the example script hashes and extraction
evidence. [The continuation report](COMPLETION-WORK-2026-09-06.md) also records
the pre-correction 20 mm/mV failure and the broader device limitations.

## Archived first-campaign examples

Use the synthetic suite and installed-consumer commands in
[REGRESSION.md](REGRESSION.md). Fixture generation writes the original image,
truth CSV and exact construction annotations before extraction. The installed
run writes a diagnostic overlay, canonical/compact signals, uncertainty,
segment map and paper visualization. The source and truth are never replaced.

Three representative examples were executed and inspected in the 6 September
campaign. Final-package Python and JavaScript examples returned identical
evidence summaries for all three, including original-image access on abstention.

| Example | Truth comparison | Evidence outcome |
| --- | --- | --- |
| `smoke__clean` | 16.20953 µV aligned RMSE, 0.9912 correlation, 99.95% coverage | Twelve segments available; strict semantics passed; `needs_review` |
| `smoke__low_resolution` | 26.38297 µV aligned RMSE, 0.9756 correlation, 99.99% coverage | Twelve segments available; narrow V3 feature loss remains visible/documented; `needs_review` |
| `waveparse-12x1__clean` | No numerical reconstruction; absent error is not zero error | Source and abstention reason preserved; original inspection remains available |

These are three examples selected from eight attempts in one deterministic
waveform family. They are not independent patient observations. No example was
accepted for quantitative use. A clean-looking render does not establish
accurate narrow features or clinical validation.

In the retained local campaign directory, `iteration-c/visual-audit/` contains
source/diagnostic/paper composites and the recorded overview inspection for all
eight inputs. Final D source, diagnostic, paper and canonical CSV bytes match
those inspected outputs exactly. `installed-examples-d/` records the executed
Python/JavaScript summaries. These generated artifacts are intentionally absent
from the clean source archive; the documented generator reproduces the example
workflow without downloading patient images.

For downstream measurement, inspect the original and diagnostic output first.
Use CSV values in µV, explicit internal gaps and the segment map's local time
and calibration reference. The 500 Hz export grid does not establish a 500 Hz
source resolution or simultaneous acquisition between panels. A review binds
its decision to specific hashes and retains uncertainty and interval lineage.

The [final regression receipt](verification/2026-09-06-iteration-d.json) binds
example source, package, runtime and scorer identities to the results. The
[public waveform pilot](PUBLIC-PILOT-2026-09-06.md) supplies a separate, less
favourable evidence stratum; the examples must not substitute for it.
