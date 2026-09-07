import { createHash } from "node:crypto"
import runtime from "../../../config/waveparse-runtime.json"
import type { DigitizerCandidateSummary } from "@/lib/runs"
import type { LayoutGeometryReport, PreprocessingReport } from "@/lib/digitizer/contracts"
import { candidateCapabilities } from "@/lib/digitizer/candidate-capabilities"

const hash=(value:unknown)=>createHash("sha256").update(JSON.stringify(value)).digest("hex")

export function candidateAncestry(candidate: DigitizerCandidateSummary, preprocessing: PreprocessingReport, calibration?: LayoutGeometryReport["calibration"]) {
  const parameters=candidate.parameters
  const derived=candidateCapabilities(parameters).derived
  const native=parameters.vectorizer==="native-grid-path"
  return { version:1,backendFamily:derived ? "derived" : native ? "native-grid" : "open-ecg-model",
    sourceSha256:preprocessing.sourceSha256,
    engineCommit:native || derived ? null : runtime.engineCommit,
    engineModelHashes:native || derived ? [] : runtime.models.map(model=>model.sha256),
    preprocessingIdentity:hash({inputVariant:parameters.inputVariant ?? "original",workingImage:preprocessing.workingImage,
      annotationMask:preprocessing.annotationMask,preparedImage:preprocessing.preparedImage,cropBox:parameters.cropBox}),
    geometryIdentity:hash(preprocessing.geometryCorrection ?? null),
    calibrationIdentity:hash(calibration ?? null),
    labelRecognitionMethod:parameters.semanticLeadIdentityMethod ?? "not-recorded",
    labelModelIndependence:"not-established",
    parentCandidateIds:[...new Set(Object.values(candidate.leadSources ?? {}).flatMap(value=>value.split("+")))].sort(),
    featureCacheShared:candidate.featureCacheHit ?? false,
    independenceClaim:"not-established; compare shared models, preprocessing, geometry and calibration",
  }
}
