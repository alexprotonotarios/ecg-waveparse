# Source-aware import decision, 6 September 2026

The current input directory contains no PDF, XML or native waveform recording
requiring an additional adapter. Direct PNG/JPEG and explicit WebP/TIFF raster
conversion cover the observed workflow. The installed API refuses other formats.
No vector/native adapter is justified by the supplied engineering review alone.
The PTB reader added for the public pilot prepares benchmark truth. It is not
a native-waveform input route in the installed digitization API.

| Available source | Intended future route | Current state |
| --- | --- | --- |
| Native digital waveform with units, sample rate and recording identity | Read actual samples and preserve acquisition metadata | Export CSVs can be read externally; native recording import is unsupported |
| PDF drawing paths | Inspect actual page operators, transforms and clipping; recover the plotted representation | Unsupported; a PDF extension does not prove vector content |
| Image-only PDF | Explicit page rasterization with document/page hash, resolution and conversion loss | Unsupported; no silent image conversion in the local library |
| Mixed vector/raster page | Route independently verified regions or refuse | Unsupported |
| Encrypted or malformed document | Refuse before content interpretation | Unsupported format at admission |

A future adapter must first demonstrate synthetic path/transform recovery,
correct raster-only classification, malformed/encrypted refusal and the same
source/segment/calibration contract. Recovering a plotted vector path does not
recover every acquisition sample. A new dependency or parser should be admitted
only against a concrete source requirement, including its licence and resource
boundaries. This is the report's demand-led feasibility decision; no PDF/vector
accuracy claim or new import implementation is made.
