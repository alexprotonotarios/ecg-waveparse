import assert from "node:assert/strict"
import test from "node:test"

import { digitizerTestUtils } from "../src/lib/digitizer"

test("CPU inference bounds the Apple allocation cache only in its own macOS child", () => {
  const inherited = process.env.MallocLargeCache
  assert.equal(digitizerTestUtils.subprocessEnvironment("cpu", "darwin").MallocLargeCache, "0")
  for (const [device, platform] of [["mps", "darwin"], ["cpu", "linux"], [undefined, "darwin"]] as const) {
    assert.equal(digitizerTestUtils.subprocessEnvironment(device, platform).MallocLargeCache, inherited)
  }
  assert.equal(process.env.MallocLargeCache, inherited)
  assert.equal("CUES_ECG_DIGITIZER_WORKER_SECRET" in digitizerTestUtils.subprocessEnvironment("cpu", "darwin"), false)
})

const leadOrder = [
  "I",
  "II",
  "III",
  "aVR",
  "aVL",
  "aVF",
  "V1",
  "V2",
  "V3",
  "V4",
  "V5",
  "V6",
]

test("keeps blank canonical CSV cells missing instead of coercing them to zero", () => {
  assert.ok(Number.isNaN(digitizerTestUtils.parseCanonicalCell("")))
  assert.ok(Number.isNaN(digitizerTestUtils.parseCanonicalCell("  ")))
  assert.ok(Number.isNaN(digitizerTestUtils.parseCanonicalCell("nan")))
  assert.equal(digitizerTestUtils.parseCanonicalCell("0"), 0)
  assert.equal(digitizerTestUtils.parseCanonicalCell("-12.5"), -12.5)
})

test("estimates independent horizontal resolution instead of equating interpolation with 500 Hz", () => {
  const preprocessing = {
    version: 1,
    sourceSha256: "test",
    source: { width: 3000, height: 1368, format: "JPEG" },
    annotationMask: {
      maskedPixels: 0,
      maskedFraction: 0,
      components: [],
    },
    preparedImage: {
      method: "test",
      maskDilationPixels: 1,
      medianWindowPixels: 11,
      morphologyReconstructed: false as const,
    },
  }

  assert.equal(
    digitizerTestUtils.estimateEffectiveSampleRateHz(preprocessing, 1500),
    150
  )
  assert.equal(
    digitizerTestUtils.estimateEffectiveSampleRateHz(preprocessing, 2200),
    220
  )

  assert.equal(
    digitizerTestUtils.estimateEffectiveSampleRateHz(
      {
        ...preprocessing,
        source: { width: 1368, height: 3000, format: "JPEG" },
      },
      2200
    ),
    220
  )
})

test("selects MPS automatically, permits CPU pinning, and rejects unavailable explicit MPS", () => {
  assert.equal(digitizerTestUtils.selectDigitizerDevice(undefined, true), "mps")
  assert.equal(digitizerTestUtils.selectDigitizerDevice("auto", false), "cpu")
  assert.equal(digitizerTestUtils.selectDigitizerDevice("cpu", true), "cpu")
  assert.equal(digitizerTestUtils.selectDigitizerDevice(" MPS ", true), "mps")
  assert.throws(
    () => digitizerTestUtils.selectDigitizerDevice("mps", false),
    /does not expose an available MPS backend/
  )
  assert.throws(
    () => digitizerTestUtils.selectDigitizerDevice("cuda", true),
    /must be one of auto, cpu, or mps/
  )
})

test("configures both Open-ECG-Digitizer neural networks on the selected device", () => {
  const config = digitizerTestUtils.openEcgConfig({
    inputDir: "/tmp/input",
    outputDir: "/tmp/output",
    resampleSize: 1500,
    vectorizer: "probability-centroid",
    device: "mps",
  })

  assert.equal(config.match(/device: 'mps'/g)?.length, 2)
  assert.doesNotMatch(config, /device: 'cpu'/)
})

test("nondefault physical gain reaches neural conversion and stability repeats", () => {
  const options = {inputDir: "/tmp/input", outputDir: "/tmp/output", resampleSize: 1500,
    vectorizer: "probability-centroid" as const, device: "mps" as const}
  const historical = digitizerTestUtils.openEcgConfig(options)
  assert.equal(digitizerTestUtils.openEcgConfig({...options, gainMmPerMv: 10}), historical)
  for (const gainMmPerMv of [5, 20]) {
    assert.match(digitizerTestUtils.openEcgConfig({...options, gainMmPerMv}), new RegExp(`gain_mm_per_mv: ${gainMmPerMv}`))
    const source = {id: "gain-source", label: "Gain source", status: "completed" as const, score: 0,
      parameters: {...options, gainMmPerMv}}
    const repeat = digitizerTestUtils.stabilityCandidateConfig(source, "cpu-confirmation", "cpu")
    assert.equal(repeat.gainMmPerMv, gainMmPerMv)
  }
  for (const gainMmPerMv of [0, -10, NaN, Infinity, 7.5]) {
    assert.throws(() => digitizerTestUtils.openEcgConfig({...options, gainMmPerMv}), /supported physical gain/)
  }
})

test("uses the corrected constrained layout profile for 6x2 pages with a rhythm row", () => {
  const config = digitizerTestUtils.openEcgConfig({
    inputDir: "/tmp/input",
    outputDir: "/tmp/output",
    resampleSize: 2200,
    vectorizer: "probability-centroid",
    device: "mps",
    layoutConstraint: "standard_6x2_with_r1_ignored",
  })

  assert.match(
    config,
    /lead_layout_standard_6x2_with_r1_ignored\.yml/
  )
  assert.doesNotMatch(config, /lead_layouts_reliable\.yml/)
  assert.match(
    config,
    /layout_should_include_substring: "standard_6x2_with_r1_ignored"/
  )
})

test("uses the dedicated geometry-confirmed profile for a plain 3x4 page", () => {
  const config = digitizerTestUtils.openEcgConfig({
    inputDir: "/tmp/input",
    outputDir: "/tmp/output",
    resampleSize: 1500,
    vectorizer: "probability-centroid",
    device: "mps",
    layoutConstraint: "standard_3x4",
  })

  assert.match(config, /lead_layout_standard_3x4\.yml/)
  assert.match(config, /layout_should_include_substring: "standard_3x4"/)
})

test("uses independent fixed-row extraction for constrained six-row panels", () => {
  const centroid = digitizerTestUtils.openEcgConfig({
    inputDir: "/tmp/limb-input",
    outputDir: "/tmp/output",
    resampleSize: 1500,
    vectorizer: "probability-centroid",
    device: "mps",
    layoutConstraint: "standard_6x1_limb_constrained",
  })
  const path = digitizerTestUtils.openEcgConfig({
    inputDir: "/tmp/precordial-input",
    outputDir: "/tmp/output",
    resampleSize: 1500,
    vectorizer: "dynamic-path",
    device: "mps",
    layoutConstraint: "precordial_6x1_constrained",
  })

  assert.match(centroid, /FixedRowCentroidSignalExtractor/)
  assert.match(path, /FixedRowPathSignalExtractor/)
})

test("uses label-aware profiles for Cabrera panels and alternate rhythm layouts", () => {
  const cabrera = digitizerTestUtils.openEcgConfig({
    inputDir: "/tmp/cabrera-input",
    outputDir: "/tmp/output",
    resampleSize: 1500,
    vectorizer: "dynamic-path",
    device: "mps",
    layoutConstraint: "cabrera_6x1_limb_constrained",
  })
  const threeRhythms = digitizerTestUtils.openEcgConfig({
    inputDir: "/tmp/rhythm-input",
    outputDir: "/tmp/output",
    resampleSize: 2200,
    vectorizer: "dynamic-path",
    device: "mps",
    layoutConstraint: "standard_3x4_with_r3",
  })

  assert.match(cabrera, /FixedRowPathSignalExtractor/)
  assert.match(cabrera, /lead_layouts_reliable\.yml/)
  assert.match(threeRhythms, /lead_layouts_reliable\.yml/)
  assert.match(
    threeRhythms,
    /layout_should_include_substring: "standard_3x4_with_r3"/
  )
})

test("maps the cited second V1 arrow to the V1 sample interval", () => {
  const preprocessing = {
    version: 1,
    sourceSha256: "test",
    source: { width: 3000, height: 1368, format: "JPEG" },
    annotationMask: {
      maskedPixels: 1183,
      maskedFraction: 1183 / (3000 * 1368),
      components: [
        {
          x0: 1909,
          y0: 30,
          x1: 1943,
          y1: 108,
          pixels: 1183,
          fillFraction: 0.446,
          dominantColor: "blue" as const,
        },
      ],
    },
    preparedImage: {
      method: "test",
      maskDilationPixels: 1,
      medianWindowPixels: 11,
      morphologyReconstructed: false as const,
    },
  }

  const ranges = digitizerTestUtils.annotationRangesByLead(
    preprocessing,
    "standard_3x4_with_r1",
    5000
  )

  assert.deepEqual(Object.keys(ranges), ["V1"])
  assert.equal(ranges.V1.length, 1)
  assert.ok(ranges.V1[0].start < 3182)
  assert.ok(ranges.V1[0].end > 3182)
  assert.ok(ranges.V1[0].start >= 2500)
  assert.ok(ranges.V1[0].end <= 3750)
  assert.ok(ranges.V1[0].end - ranges.V1[0].start <= 320)
})

test("annotation suppression does not erase a neighboring V3 QRS", () => {
  const preprocessing = {
    version: 1,
    sourceSha256: "test",
    source: { width: 3000, height: 1368, format: "JPEG" },
    annotationMask: {
      maskedPixels: 939,
      maskedFraction: 939 / (3000 * 1368),
      components: [
        {
          x0: 1696,
          y0: 935,
          x1: 1728,
          y1: 1007,
          pixels: 939,
          fillFraction: 0.413,
          dominantColor: "red" as const,
        },
      ],
    },
    preparedImage: {
      method: "test",
      maskDilationPixels: 1,
      medianWindowPixels: 11,
      morphologyReconstructed: false as const,
    },
  }

  const ranges = digitizerTestUtils.annotationRangesByLead(
    preprocessing,
    "standard_3x4_with_r1",
    5000
  )

  assert.deepEqual(Object.keys(ranges), ["V3"])
  assert.ok(ranges.V3[0].start > 2750)
  assert.ok(ranges.V3[0].end - ranges.V3[0].start <= 120)
})

test("keeps annotation ranges aligned after panel timing normalization", () => {
  const preprocessing = {
    version: 1,
    sourceSha256: "test",
    source: { width: 3000, height: 1368, format: "JPEG" },
    annotationMask: {
      maskedPixels: 1183,
      maskedFraction: 1183 / (3000 * 1368),
      components: [
        {
          x0: 1909,
          y0: 30,
          x1: 1943,
          y1: 108,
          pixels: 1183,
          fillFraction: 0.446,
          dominantColor: "blue" as const,
        },
      ],
    },
    preparedImage: {
      method: "test",
      maskDilationPixels: 1,
      medianWindowPixels: 11,
      morphologyReconstructed: false as const,
    },
  }
  const original = digitizerTestUtils.annotationRangesByLead(
    preprocessing,
    "standard_3x4_with_r1",
    5000
  )
  const normalized = digitizerTestUtils.annotationRangesByLead(
    preprocessing,
    "standard_3x4_with_r1",
    5000,
    [
      {
        column: 2,
        sourceStart: 90,
        sourceEnd: 1249,
        scale: 1249 / 1159,
        referenceLeads: ["V1", "V2", "V3"],
      },
    ]
  )

  assert.ok(normalized.V1[0].start < original.V1[0].start)
  assert.ok(normalized.V1[0].end < original.V1[0].end)
  assert.ok(normalized.V1[0].start >= 2500)
  assert.ok(normalized.V1[0].end <= 3750)
})

test("maps source annotations through the selected bounded crop coordinate frame", () => {
  const preprocessing = {
    ...annotatedPreprocessing,
    source: { width: 2000, height: 1000, format: "PNG" },
    workingImage: {
      width: 1000,
      height: 500,
      scaleX: 0.5,
      scaleY: 0.5,
      method: "test",
      maxLongEdgePixels: 1000,
    },
    annotationMask: {
      maskedPixels: 100,
      maskedFraction: 0.001,
      components: [
        {
          x0: 1180,
          y0: 230,
          x1: 1220,
          y1: 270,
          pixels: 100,
          fillFraction: 0.5,
          dominantColor: "blue" as const,
        },
      ],
    },
  }
  const ranges = digitizerTestUtils.annotationRangesByLead(
    preprocessing,
    "standard_6x2",
    100,
    [],
    {
      inputVariant: "annotation-masked",
      cropBox: { left: 500, top: 0, right: 900, bottom: 500 },
    }
  )

  assert.ok(ranges.II?.length)
  assert.equal(ranges.V2, undefined)
})

test("publishes annotation-overlap samples as missing without changing other leads", () => {
  const canonical = {
    leads: leadOrder,
    rows: Array.from({ length: 5000 }, (_, sample) =>
      leadOrder.map((_, leadIndex) => sample + leadIndex)
    ),
  }
  const preprocessing = {
    version: 2,
    sourceSha256: "test",
    source: { width: 3000, height: 1368, format: "JPEG" },
    annotationMask: {
      maskedPixels: 1183,
      maskedFraction: 1183 / (3000 * 1368),
      components: [
        {
          x0: 1909,
          y0: 30,
          x1: 1943,
          y1: 108,
          pixels: 1183,
          fillFraction: 0.446,
          dominantColor: "blue" as const,
        },
      ],
    },
    preparedImage: {
      method: "test",
      maskDilationPixels: 1,
      fillRingPixels: 8,
      fillPercentile: 75,
      morphologyReconstructed: false as const,
    },
  }

  const result = digitizerTestUtils.suppressAnnotatedSamples(
    canonical,
    preprocessing,
    "standard_3x4_with_r1"
  )
  const ranges = digitizerTestUtils.annotationRangesByLead(
    preprocessing,
    "standard_3x4_with_r1",
    canonical.rows.length
  )
  const v1Index = leadOrder.indexOf("V1")
  const v2Index = leadOrder.indexOf("V2")
  const { start, end } = ranges.V1[0]

  assert.ok(Number.isFinite(result.rows[start - 1][v1Index]))
  assert.ok(result.rows.slice(start, end).every((row) => Number.isNaN(row[v1Index])))
  assert.ok(Number.isFinite(result.rows[end][v1Index]))
  assert.ok(result.rows.every((row) => Number.isFinite(row[v2Index])))
  assert.ok(Number.isFinite(canonical.rows[start][v1Index]))
})

test("keeps a review-only estimate for annotation gaps without publishing it as signal", () => {
  const canonical = sixByTwoCanonical()
  const preprocessing = {
    version: 2,
    sourceSha256: "test",
    source: { width: 3000, height: 1368, format: "JPEG" },
    annotationMask: {
      maskedPixels: 1183,
      maskedFraction: 1183 / (3000 * 1368),
      components: [
        {
          x0: 1909,
          y0: 30,
          x1: 1943,
          y1: 108,
          pixels: 1183,
          fillFraction: 0.446,
          dominantColor: "blue" as const,
        },
      ],
    },
    preparedImage: {
      method: "test",
      maskDilationPixels: 1,
      fillRingPixels: 8,
      fillPercentile: 75,
      morphologyReconstructed: false as const,
    },
  }
  const selected = leadFusionCandidate("selected", canonical)
  const published = digitizerTestUtils.suppressAnnotatedSamples(
    canonical,
    preprocessing,
    selected.layout
  )
  const uncertainty = digitizerTestUtils.buildUncertaintyRows(
    selected,
    [selected],
    preprocessing,
    published
  )
  const annotatedRows = uncertainty.filter(
    (row) => row.lead === "V1" && row.annotationOverlap
  )

  assert.ok(annotatedRows.length > 0)
  assert.ok(annotatedRows.every((row) => Number.isNaN(row.valueUv)))
  assert.ok(
    annotatedRows.every((row) => Number.isFinite(row.reviewEstimateUv))
  )
  assert.ok(
    uncertainty
      .filter((row) => !row.annotationOverlap)
      .every((row) => Number.isNaN(row.reviewEstimateUv))
  )
})

test("lead alignment preserves a narrow notch while correcting shift and baseline", () => {
  const reference = Array.from({ length: 200 }, () => 0)
  reference[96] = 40
  reference[97] = 130
  reference[98] = 55
  reference[99] = 170
  reference[100] = 20

  const comparison = Array.from({ length: 200 }, () => 25)
  reference.forEach((value, index) => {
    if (index + 3 < comparison.length) comparison[index + 3] = value + 25
  })

  const aligned = digitizerTestUtils.alignSeries(reference, comparison)
  assert.equal(aligned.shift, 3)
  assert.ok(Math.abs(aligned.offsetUv - 25) < 1e-9)
  assert.ok(aligned.rmseUv < 1e-9)
  assert.deepEqual(aligned.values.slice(96, 101), reference.slice(96, 101))
})

test("lead alignment tolerates modest affine timing drift and isolated artifacts", () => {
  const length = 500
  const center = (length - 1) / 2
  const reference = Array.from({ length }, (_, index) =>
    80 * Math.sin(index / 13) +
    220 * Math.exp(-(((index % 95 - 44) / 5) ** 2))
  )
  const comparison = Array.from({ length }, (_, index) => {
    const source = center + (index - 4 - center) / 1.02
    const lower = Math.max(0, Math.floor(source))
    const upper = Math.min(length - 1, Math.ceil(source))
    const weight = source - lower
    return reference[lower] * (1 - weight) + reference[upper] * weight + 30
  })
  comparison[300] += 1200

  const aligned = digitizerTestUtils.alignSeries(reference, comparison)

  assert.equal(aligned.timeScale, 1.02)
  assert.ok(Math.abs(aligned.shift - 4) <= 1)
  assert.ok(Math.abs(aligned.offsetUv - 30) < 2)
  assert.ok(aligned.rmseUv < 12)
})

test("short gaps require two agreeing peers or one source-verified peer", () => {
  const complete = Array.from({ length: 100 }, (_, index) => index * 2)
  const reference = complete.map((value, index) =>
    index >= 40 && index < 46 ? Number.NaN : value
  )
  const single = digitizerTestUtils.repairShortPeerSupportedGaps(reference, [
    { id: "single", values: complete, sourceVerified: false },
  ])
  assert.equal(single.repairedSamples, 0)

  const consensus = digitizerTestUtils.repairShortPeerSupportedGaps(reference, [
    { id: "a", values: complete, sourceVerified: false },
    {
      id: "b",
      values: complete.map((value) => value + 4),
      sourceVerified: false,
    },
  ])
  assert.equal(consensus.repairedSamples, 6)
  assert.ok(consensus.values.slice(40, 46).every(Number.isFinite))

  const verified = digitizerTestUtils.repairShortPeerSupportedGaps(reference, [
    { id: "verified", values: complete, sourceVerified: true },
  ])
  assert.equal(verified.repairedSamples, 6)
})

test("lead compaction removes absent panels but preserves gaps inside the printed panel", () => {
  const canonical = {
    leads: ["I", "V1"],
    rows: [
      [1, Number.NaN],
      [2, Number.NaN],
      [Number.NaN, Number.NaN],
      [4, Number.NaN],
      [Number.NaN, 10],
      [Number.NaN, 11],
      [Number.NaN, Number.NaN],
      [Number.NaN, 13],
    ],
  }

  assert.deepEqual(
    digitizerTestUtils.canonicalLeadSegment(
      canonical,
      "standard_6x2",
      "I"
    )?.values,
    [1, 2, Number.NaN, 4]
  )
  assert.deepEqual(
    digitizerTestUtils.canonicalLeadSegment(
      canonical,
      "standard_6x2",
      "V1"
    )?.values,
    [10, 11, Number.NaN, 13]
  )
})

test("combines constrained limb and precordial panels without mixing lead columns", () => {
  const left = {
    leads: leadOrder,
    rows: Array.from({ length: 20 }, (_, sample) =>
      leadOrder.map((lead, index) =>
        index < 6 ? sample + index * 100 : Number.NaN
      )
    ),
  }
  const right = {
    leads: leadOrder,
    rows: Array.from({ length: 20 }, (_, sample) =>
      leadOrder.map((lead, index) =>
        index >= 6 ? sample + index * 100 : Number.NaN
      )
    ),
  }

  const combined = digitizerTestUtils.combineSixByTwoPanelCanonicals(
    left,
    right
  )
  const leadI = digitizerTestUtils.canonicalLeadSegment(
    combined,
    "standard_6x2_with_r1_ignored",
    "I"
  )
  const leadV1 = digitizerTestUtils.canonicalLeadSegment(
    combined,
    "standard_6x2_with_r1_ignored",
    "V1"
  )

  assert.equal(combined.rows.length, 5000)
  assert.ok(leadI?.values.every(Number.isFinite))
  assert.ok(leadV1?.values.every(Number.isFinite))
  assert.ok(combined.rows.slice(2500).every((row) => Number.isNaN(row[0])))
  assert.ok(combined.rows.slice(0, 2500).every((row) => Number.isNaN(row[6])))
})

test("combines stacked label-identified panels into a 12x1 review canvas", () => {
  const limb = {
    leads: leadOrder,
    rows: Array.from({ length: 20 }, (_, sample) =>
      leadOrder.map((_, index) => (index < 6 ? sample + index * 100 : Number.NaN))
    ),
  }
  const precordial = {
    leads: leadOrder,
    rows: Array.from({ length: 20 }, (_, sample) =>
      leadOrder.map((_, index) => (index >= 6 ? sample + index * 100 : Number.NaN))
    ),
  }

  const combined = digitizerTestUtils.combineStackedPanelCanonicals(
    limb,
    precordial
  )
  const bounded = digitizerTestUtils.suppressFullWidthPanelBoundaries(combined)

  assert.equal(combined.rows.length, 5000)
  assert.ok(combined.rows.every((row) => row.every(Number.isFinite)))
  assert.ok(bounded.rows.slice(0, 500).every((row) => row.every(Number.isNaN)))
  assert.ok(
    bounded.rows.slice(500, 4850).every((row) => row.every(Number.isFinite))
  )
  assert.ok(bounded.rows.slice(4850).every((row) => row.every(Number.isNaN)))
})

test("maps Cabrera 12x1 canonical leads onto full-width QA segments", () => {
  const canonical = {
    leads: leadOrder,
    rows: Array.from({ length: 50 }, (_, sample) =>
      leadOrder.map((_, index) => sample + index * 100)
    ),
  }

  assert.deepEqual(
    digitizerTestUtils.canonicalLeadSegment(canonical, "cabrera_12x1", "aVR")
      ?.values,
    canonical.rows.map((row) => row[leadOrder.indexOf("aVR")])
  )
})

test("accepts only confident physiologic calibration metadata", () => {
  assert.deepEqual(
    digitizerTestUtils.validatedCalibration({
      calibration: {
        method: "test",
        detected: true,
        paperSpeedMmPerSecond: 50,
        gainMmPerMv: 10,
        confidence: 0.8,
      },
    }),
    {
      method: "test",
      detected: true,
      paperSpeedMmPerSecond: 50,
      gainMmPerMv: 10,
      confidence: 0.8,
    }
  )
  assert.equal(
    digitizerTestUtils.validatedCalibration({
      calibration: {
        method: "test",
        detected: true,
        paperSpeedMmPerSecond: 50,
        gainMmPerMv: 10,
        confidence: 0.2,
      },
    }),
    undefined
  )
})

test("review-only constrained output keeps unsupported panel boundaries missing", () => {
  const canonical = sixByTwoCanonical()
  const result =
    digitizerTestUtils.suppressConstrainedPanelBoundaries(canonical)

  for (const lead of leadOrder) {
    const values = digitizerTestUtils.canonicalLeadSegment(
      result,
      "standard_6x2_with_r1_ignored",
      lead
    )?.values
    assert.ok(values)
    assert.ok(values.slice(0, 5).every(Number.isNaN))
    assert.ok(Number.isFinite(values[5]))
    assert.ok(Number.isFinite(values.at(-3)))
    assert.ok(Number.isNaN(values.at(-1)))
  }
})

test("source-detected panel timing is not masked a second time", () => {
  assert.equal(
    digitizerTestUtils.nativeCanonicalRequiresBoundarySuppression(
      "standard_6x2",
      {
        ...semanticLayoutFidelity,
        sourcePanelTimingDetected: true,
      }
    ),
    false
  )
  assert.equal(
    digitizerTestUtils.nativeCanonicalRequiresBoundarySuppression(
      "standard_6x2",
      semanticLayoutFidelity
    ),
    true
  )
})

test("panel timing normalization removes shared column whitespace without bridging gaps", () => {
  const canonical = {
    leads: ["I", "II", "III", "V1", "V2", "V3"],
    rows: Array.from({ length: 20 }, (_, sample) => {
      const local = sample % 10
      return [
        sample < 10 && local <= 8 ? local : Number.NaN,
        sample < 10 && local <= 8 && local !== 4
          ? local + 10
          : Number.NaN,
        sample < 10 && local <= 8 ? local + 20 : Number.NaN,
        sample >= 10 && local >= 1 ? local + 30 : Number.NaN,
        sample >= 10 && local >= 1 && local !== 5
          ? local + 40
          : Number.NaN,
        sample >= 10 && local >= 1 ? local + 50 : Number.NaN,
      ]
    }),
  }

  const result = digitizerTestUtils.normalizeCanonicalPanelTiming(
    canonical,
    "standard_6x2"
  )

  assert.equal(result.corrections.length, 2)
  assert.ok(Math.abs(result.corrections[0].scale - 9 / 8) < 1e-9)
  assert.ok(Math.abs(result.corrections[1].scale - 9 / 8) < 1e-9)
  const leadI = result.canonical.rows.slice(0, 10).map((row) => row[0])
  const leadII = result.canonical.rows.slice(0, 10).map((row) => row[1])
  const leadV1 = result.canonical.rows.slice(10).map((row) => row[3])
  const leadV2 = result.canonical.rows.slice(10).map((row) => row[4])
  assert.ok(leadI.every(Number.isFinite))
  assert.ok(leadV1.every(Number.isFinite))
  assert.ok(Number.isNaN(leadII[4]))
  assert.ok(Number.isNaN(leadII[5]))
  assert.ok(Number.isNaN(leadV2[4]))
  assert.ok(Number.isNaN(leadV2[5]))
  assert.ok(Number.isFinite(leadII[3]))
  assert.ok(Number.isFinite(leadII[6]))
  assert.ok(Number.isFinite(leadV2[3]))
  assert.ok(Number.isFinite(leadV2[6]))
})

test("panel timing normalization abstains without two consistent complete references", () => {
  const canonical = {
    leads: ["I", "V1"],
    rows: Array.from({ length: 20 }, (_, sample) => {
      const local = sample % 10
      return [
        sample < 10 && local <= 8 ? local : Number.NaN,
        sample >= 10 && local >= 1 ? local : Number.NaN,
      ]
    }),
  }

  const result = digitizerTestUtils.normalizeCanonicalPanelTiming(
    canonical,
    "standard_6x2"
  )

  assert.deepEqual(result.corrections, [])
  assert.deepEqual(result.canonical, canonical)
})

function sixByTwoCanonical(missingLead?: string, truncatedLead?: string) {
  const rows = Array.from({ length: 100 }, (_, sample) =>
    leadOrder.map((lead, leadIndex) => {
      if (lead === missingLead) return Number.NaN
      const isFirstColumn = leadIndex < 6
      const isFirstHalf = sample < 50
      if (isFirstColumn !== isFirstHalf) return Number.NaN
      const leadSample = isFirstHalf ? sample : sample - 50
      if (lead === truncatedLead && leadSample >= 30) return Number.NaN
      return sample + leadIndex
    })
  )
  return { leads: leadOrder, rows }
}

function threeByFourCanonical({
  gapLead,
  truncatedLead,
}: {
  gapLead?: string
  truncatedLead?: string
} = {}) {
  const columns: Record<string, number> = {
    I: 0,
    II: 0,
    III: 0,
    aVR: 1,
    aVL: 1,
    aVF: 1,
    V1: 2,
    V2: 2,
    V3: 2,
    V4: 3,
    V5: 3,
    V6: 3,
  }
  const rows = Array.from({ length: 400 }, (_, sample) =>
    leadOrder.map((lead, leadIndex) => {
      const column = Math.floor(sample / 100)
      if (columns[lead] !== column) return Number.NaN
      const leadSample = sample % 100
      if (lead === gapLead && leadSample >= 45 && leadSample < 57) {
        return Number.NaN
      }
      if (lead === truncatedLead && leadSample >= 72) {
        return Number.NaN
      }
      return sample + leadIndex
    })
  )
  return { leads: leadOrder, rows }
}

function candidate(
  id: string,
  inputVariant: "original" | "annotation-masked",
  missingLead?: string
) {
  const canonical = sixByTwoCanonical(missingLead)
  const qa = digitizerTestUtils.evaluateQa(
    canonical,
    "standard_6x2"
  )
  return {
    id,
    label: id,
    localPath: id,
    status: "completed" as const,
    parameters: {
      resampleSize: 1500,
      semanticLeadIdentityConfirmed: true,
      semanticLeadIdentityMethod: "test-exact-page-label-recognizer",
      semanticLeadIdentityOrder: "standard" as const,
      inputVariant,
      vectorizer: "probability-centroid" as const,
    },
    layout: "standard_6x2",
    score: inputVariant === "annotation-masked" ? 0 : 100,
    qa,
    canonical,
    sourceFidelity: semanticLayoutFidelity,
  }
}

const annotatedPreprocessing = {
  version: 1,
  sourceSha256: "test",
  source: { width: 3000, height: 1368, format: "PNG" },
  annotationMask: {
    maskedPixels: 100,
    maskedFraction: 100 / (3000 * 1368),
    components: [],
  },
  preparedImage: {
    method: "test",
    maskDilationPixels: 1,
    medianWindowPixels: 11,
    morphologyReconstructed: false as const,
  },
}

const passedSourceFidelity = {
  passed: true,
  method: "native-grid-residual-viterbi-v1",
  pixelsPerMm: 3,
  layoutConfidence: 0.75,
  evidenceMedian: 0.88,
  evidenceP10: 0.62,
  maxNativeJumpPixels: 20,
  p95NativeJumpPixels: 8,
  minimumCoverage: 0.9,
  leadOrderValidation: {
    passed: true,
    selectedOrder: "standard",
    method: "test-label-order",
  },
  leadLabelValidation: {
    passed: true,
    order: "standard",
    method: "test-lead-labels",
    semanticIdentityConfirmed: true,
  },
  precordialLabelValidation: {
    passed: true,
    method: "test-precordial-labels",
    semanticIdentityConfirmed: true,
  },
  limbLabelValidation: {
    passed: true,
    order: "standard",
    method: "test-limb-labels",
    semanticIdentityConfirmed: true,
  },
  leadMetrics: {},
}

const semanticLayoutFidelity = {
  ...passedSourceFidelity,
  passed: false,
}

test("preprocessed candidates require adaptive eligibility and native source fidelity", () => {
  const modelTrial = {
    ...leadFusionCandidate("preprocessed-path-2200"),
    parameters: {
      ...leadFusionCandidate("preprocessed-path-2200").parameters,
      kind: "adaptive-preprocessed" as const,
      inputVariant: "preprocessed" as const,
      vectorizer: "dynamic-path" as const,
      adaptivePreprocessingEligible: false,
    },
  }
  const nativeTrial = {
    ...leadFusionCandidate("preprocessed-native-grid-path"),
    parameters: {
      ...leadFusionCandidate("preprocessed-native-grid-path").parameters,
      inputVariant: "preprocessed" as const,
      vectorizer: "native-grid-path" as const,
      adaptivePreprocessingEligible: true,
    },
    sourceFidelity: {
      ...passedSourceFidelity,
      passed: false,
    },
  }

  assert.equal(
    digitizerTestUtils.candidateEligibleForSelection(modelTrial),
    false
  )
  assert.equal(
    digitizerTestUtils.candidateEligibleForSelection({
      ...modelTrial,
      parameters: {
        ...modelTrial.parameters,
        adaptivePreprocessingEligible: true,
      },
    }),
    true
  )
  assert.equal(
    digitizerTestUtils.candidateEligibleForSelection(nativeTrial),
    false
  )
  assert.equal(
    digitizerTestUtils.candidateEligibleForSelection({
      ...nativeTrial,
      sourceFidelity: passedSourceFidelity,
    }),
    true
  )

  const eligibleModel = {
    ...modelTrial,
    parameters: {
      ...modelTrial.parameters,
      adaptivePreprocessingEligible: true,
    },
  }
  const corroboratingPeerBase = leadFusionCandidate("original-native-peer")
  const corroboratingPeer = {
    ...corroboratingPeerBase,
    parameters: {
      ...corroboratingPeerBase.parameters,
      kind: "native-grid" as const,
      vectorizer: "native-grid-path" as const,
    },
    sourceFidelity: passedSourceFidelity,
  }
  assert.equal(
    digitizerTestUtils.selectAdaptivePreprocessedCandidate([
      eligibleModel,
      corroboratingPeer,
    ])?.id,
    "preprocessed-path-2200"
  )
  assert.equal(
    digitizerTestUtils.selectAdaptivePreprocessedCandidate([
      modelTrial,
      corroboratingPeer,
    ]),
    undefined
  )
})

test("native 3x4 and sequential outputs require a verified waveform time origin", () => {
  for (const layout of ["standard_3x4", "standard_12x1"]) {
    for (const inputVariant of ["original", "geometry-corrected"] as const) {
      const base = leadFusionCandidate(`timing-${layout}-${inputVariant}`)
      const native = {
        ...base,
        layout,
        parameters: {
          ...base.parameters,
          kind: "native-grid" as const,
          vectorizer: "native-grid-path" as const,
          inputVariant,
          geometryConfirmedLayout: true,
        },
        sourceFidelity: {
          ...passedSourceFidelity,
          sourcePanelTimingDetected: false,
          rowLocalSourceTimingDetected: false,
        },
      }
      assert.equal(digitizerTestUtils.candidateEligibleForSelection(native), false)
      assert.equal(digitizerTestUtils.candidateEligibleForSelection({
        ...native,
        sourceFidelity: { ...native.sourceFidelity, sourcePanelTimingDetected: true },
      }), true)
      assert.equal(digitizerTestUtils.candidateEligibleForSelection({
        ...native,
        sourceFidelity: { ...native.sourceFidelity, rowLocalSourceTimingDetected: true },
      }), true)
    }
  }
})

test("low-resolution native review output requires untouched-source trace corroboration", () => {
  const lowResolutionPreprocessing = {
    ...annotatedPreprocessing,
    source: { width: 420, height: 228, format: "JPEG" },
    annotationMask: {
      maskedPixels: 0,
      maskedFraction: 0,
      components: [],
    },
  }
  const canonical = sixByTwoCanonical()
  const nativeFidelity = {
    ...passedSourceFidelity,
    passed: false,
    layoutConfidence: 0.38,
    evidenceMedian: 0.42,
    evidenceP10: 0,
    minimumCoverage: 0.93,
    minimumRequiredCoverage: 0.8,
    minimumHomeRowFraction: 0.9,
    unsupportedLargeJumpCount: 6,
    maximumUnsupportedLargeJumps: 4,
  }
  const original = {
    ...leadFusionCandidate("native-original", canonical),
    parameters: {
      ...leadFusionCandidate("native-original", canonical).parameters,
      layoutConstraint: "row_local_labeled_6x2",
      inputVariant: "original" as const,
      vectorizer: "native-grid-path" as const,
    },
    sourceFidelity: {
      ...nativeFidelity,
      unsupportedLargeJumpCount: 4,
    },
  }
  const preprocessed = {
    ...leadFusionCandidate("preprocessed-native-grid-path", canonical),
    parameters: {
      ...leadFusionCandidate("preprocessed-native-grid-path", canonical)
        .parameters,
      layoutConstraint: "row_local_labeled_6x2",
      inputVariant: "preprocessed" as const,
      vectorizer: "native-grid-path" as const,
      adaptivePreprocessingEligible: true,
    },
    sourceFidelity: nativeFidelity,
  }
  const upscaledPreprocessed = {
    ...preprocessed,
    id: "preprocessed-native-grid-path-upscaled-1200",
    parameters: {
      ...preprocessed.parameters,
      upscaleToMaxDimension: 1_200,
    },
    sourceFidelity: {
      ...nativeFidelity,
      unsupportedLargeJumpCount: 4,
    },
  }

  assert.equal(
    digitizerTestUtils.hasLowResolutionNativeTraceCorroboration(
      preprocessed,
      [original, preprocessed],
      lowResolutionPreprocessing
    ),
    true
  )
  assert.equal(
    digitizerTestUtils.hasLowResolutionNativeTraceCorroboration(
      preprocessed,
      [original, preprocessed, upscaledPreprocessed],
      lowResolutionPreprocessing
    ),
    true
  )
  assert.equal(
    digitizerTestUtils.hasLowResolutionNativeTraceCorroboration(
      preprocessed,
      [original, preprocessed, upscaledPreprocessed],
      {
        ...lowResolutionPreprocessing,
        source: { width: 1_200, height: 760, format: "JPEG" },
      }
    ),
    false
  )

  const conflictingCanonical = {
    ...canonical,
    rows: canonical.rows.map((row, sample) =>
      row.map((value, leadIndex) =>
        Number.isFinite(value) && leadIndex < 8
          ? value + 300 * Math.sin(sample / 3 + leadIndex)
          : value
      )
    ),
  }
  const conflictingOriginal = {
    ...original,
    canonical: conflictingCanonical,
    qa: digitizerTestUtils.evaluateQa(
      conflictingCanonical,
      "standard_6x2"
    ),
  }
  assert.equal(
    digitizerTestUtils.hasLowResolutionNativeTraceCorroboration(
      preprocessed,
      [conflictingOriginal, preprocessed, upscaledPreprocessed],
      lowResolutionPreprocessing
    ),
    false
  )
  assert.equal(
    digitizerTestUtils.hasLowResolutionNativeTraceCorroboration(
      {
        ...preprocessed,
        sourceFidelity: {
          ...nativeFidelity,
          unsupportedLargeJumpCount: 7,
        },
      },
      [original, preprocessed, upscaledPreprocessed],
      lowResolutionPreprocessing
    ),
    false
  )
})

test("compound panel components remain audit-only until a composite validates", () => {
  const component = {
    ...leadFusionCandidate("compound-side_by_side-standard-limb"),
    parameters: {
      ...leadFusionCandidate("compound-side_by_side-standard-limb").parameters,
      layoutConstraint: "standard_6x1_limb_constrained",
      selectionEligible: false,
    },
  }

  assert.equal(
    digitizerTestUtils.candidateEligibleForSelection(component),
    false
  )
})

test("lead QA rejects a canonical candidate with an entirely missing lead", () => {
  const qa = digitizerTestUtils.evaluateQa(
    sixByTwoCanonical("aVF"),
    "standard_6x2"
  )

  assert.equal(qa.passed, false)
  assert.equal(qa.leads.aVF.finiteSamples, 0)
  assert.ok(
    qa.warnings.some(
      (warning) => warning.lead === "aVF" && warning.severity === "error"
    )
  )
})

test("lead QA rejects a lead with less than 80 percent coverage", () => {
  const qa = digitizerTestUtils.evaluateQa(
    sixByTwoCanonical(undefined, "V5"),
    "standard_6x2"
  )

  assert.equal(qa.leads.V5.finiteSamples, 30)
  assert.ok(
    qa.warnings.some(
      (warning) =>
        warning.lead === "V5" &&
        warning.severity === "error" &&
        warning.message.includes("80%")
    )
  )
  assert.equal(
    digitizerTestUtils.candidateHasAllScorableLeads({
      ...candidate("truncated", "original"),
      qa,
    }),
    false
  )
})

test("lead QA rejects a fully populated near-flat placeholder trace", () => {
  const canonical = sixByTwoCanonical()
  const leadIndex = leadOrder.indexOf("V5")
  for (let sample = 50; sample < 100; sample += 1) {
    canonical.rows[sample][leadIndex] = 4 + (sample % 5) * 0.2
  }
  const qa = digitizerTestUtils.evaluateQa(canonical, "standard_6x2")
  const result = {
    ...candidate("near-flat", "original"),
    qa,
  }

  assert.equal(qa.leads.V5.finiteSamples, 50)
  assert.ok(
    qa.warnings.some(
      (warning) =>
        warning.lead === "V5" && warning.message.includes("near-flat")
    )
  )
  assert.equal(digitizerTestUtils.candidateHasAllScorableLeads(result), false)
  const published = digitizerTestUtils.suppressNearFlatLeads(
    canonical,
    "standard_6x2"
  )
  assert.ok(
    published.rows.every((row) => !Number.isFinite(row[leadIndex]))
  )
})

test("publication rejects a long internal gap despite greater than 95 percent coverage", () => {
  const canonical = {
    leads: leadOrder,
    rows: Array.from({ length: 5000 }, (_, sample) =>
      leadOrder.map((lead, leadIndex) => {
        const isFirstColumn = leadIndex < 6
        const isFirstHalf = sample < 2500
        if (isFirstColumn !== isFirstHalf) return Number.NaN
        const leadSample = isFirstHalf ? sample : sample - 2500
        if (lead === "V3" && leadSample >= 1200 && leadSample < 1220) {
          return Number.NaN
        }
        return sample + leadIndex
      })
    ),
  }
  const result = leadFusionCandidate("internal-gap", canonical)

  assert.equal(result.qa.leads.V3.finiteSamples, 2480)
  assert.equal(
    digitizerTestUtils.candidateHasStructurallyCompleteLead(result, "V3"),
    false
  )
})

test("source-verified boundary exclusions remain publishable without inventing edge samples", () => {
  const canonical = sixByTwoCanonical()
  for (const lead of leadOrder) {
    const leadIndex = leadOrder.indexOf(lead)
    const segmentStart = leadIndex < 6 ? 0 : 50
    for (let sample = 0; sample < 5; sample += 1) {
      canonical.rows[segmentStart + sample][leadIndex] = Number.NaN
    }
    for (let sample = 48; sample < 50; sample += 1) {
      canonical.rows[segmentStart + sample][leadIndex] = Number.NaN
    }
  }
  const result = {
    ...leadFusionCandidate("source-verified-boundaries", canonical),
    sourceFidelity: {
      ...passedSourceFidelity,
      method: "row-local-calibration-anchored-six-by-two-v1",
      layoutConfidence: 0.8,
      minimumCoverage: 0.87,
    },
  }

  assert.equal(
    digitizerTestUtils.candidateHasStructurallyCompleteLead(result, "V3"),
    false
  )
  assert.equal(
    digitizerTestUtils.candidateHasSourceVerifiedBoundaryLead(result, "V3"),
    true
  )
  assert.equal(
    digitizerTestUtils.candidateHasPublishableLead(result, "V3"),
    true
  )
  assert.equal(
    digitizerTestUtils.candidateHasSourceVerifiedBoundaryLead(
      {
        ...result,
        sourceFidelity: passedSourceFidelity,
      },
      "V3"
    ),
    false
  )
  assert.ok(Number.isNaN(canonical.rows[50][leadOrder.indexOf("V3")]))
})

test("sequential 12x1 source-fidelity revisions retain validated boundary eligibility", () => {
  const candidate = leadFusionCandidate("sequential-boundary-layout")
  const sequentialFidelity = {
    ...passedSourceFidelity,
    method: "row-local-sequential-label-validated-v3",
    leadOrderValidation: {
      passed: true,
      selectedOrder: "standard" as const,
      method: "sequential-lead-label-anchors-v2",
    },
    leadLabelValidation: {
      passed: true,
      order: "standard" as const,
      method: "test-value-aware-sequential-label-recognizer",
      semanticIdentityConfirmed: true,
    },
  }

  assert.equal(
    digitizerTestUtils.candidateHasValidatedBoundaryLayout({
      ...candidate,
      sourceFidelity: sequentialFidelity,
    }),
    true
  )
  assert.equal(
    digitizerTestUtils.candidateHasValidatedBoundaryLayout({
      ...candidate,
      sourceFidelity: {
        ...sequentialFidelity,
        method: "row-local-sequential-label-validated-v2",
      },
    }),
    true
  )
})

test("source verification never converts an internal gap into a publishable lead", () => {
  const canonical = sixByTwoCanonical()
  const v3Index = leadOrder.indexOf("V3")
  for (let sample = 50; sample < 55; sample += 1) {
    canonical.rows[sample][v3Index] = Number.NaN
  }
  for (let sample = 98; sample < 100; sample += 1) {
    canonical.rows[sample][v3Index] = Number.NaN
  }
  canonical.rows[70][v3Index] = Number.NaN
  const result = {
    ...leadFusionCandidate("source-verified-internal-gap", canonical),
    sourceFidelity: {
      ...passedSourceFidelity,
      method: "row-local-calibration-anchored-six-by-two-v1",
      minimumCoverage: 0.87,
    },
  }

  assert.equal(
    digitizerTestUtils.candidateHasSourceVerifiedBoundaryLead(result, "V3"),
    false
  )
  assert.equal(
    digitizerTestUtils.candidateHasPublishableLead(result, "V3"),
    false
  )
})

test("source-verified boundary publication rejects unsafe excursions and sub-scorable coverage", () => {
  const canonical = sixByTwoCanonical()
  const v3Index = leadOrder.indexOf("V3")
  for (let sample = 50; sample < 55; sample += 1) {
    canonical.rows[sample][v3Index] = Number.NaN
  }
  const base = leadFusionCandidate("source-verified-unsafe", canonical)
  const unsafe = {
    ...base,
    sourceFidelity: {
      ...passedSourceFidelity,
      method: "row-local-calibration-anchored-six-by-two-v1",
      minimumCoverage: 0.87,
      unsafeExcursionCount: 1,
    },
  }
  const insufficient = {
    ...base,
    sourceFidelity: {
      ...passedSourceFidelity,
      method: "row-local-calibration-anchored-six-by-two-v1",
      minimumCoverage: 0.79,
    },
  }

  assert.equal(
    digitizerTestUtils.candidateHasSourceVerifiedBoundaryLead(unsafe, "V3"),
    false
  )
  assert.equal(
    digitizerTestUtils.candidateHasSourceVerifiedBoundaryLead(
      insufficient,
      "V3"
    ),
    false
  )
})

test("publication requires a structurally complete agreeing peer per lead", () => {
  const reference = leadFusionCandidate("reference")
  const agreeingBase = leadFusionCandidate("agreeing", sixByTwoCanonical())
  const agreeing = {
    ...agreeingBase,
    parameters: {
      ...agreeingBase.parameters,
      kind: "native-grid" as const,
      vectorizer: "native-grid-path" as const,
    },
    sourceFidelity: passedSourceFidelity,
  }
  const outlierCanonical = sixByTwoCanonical()
  for (let sample = 50; sample < 100; sample += 1) {
    outlierCanonical.rows[sample][leadOrder.indexOf("V3")] +=
      sample % 2 === 0 ? 500 : -500
  }
  const outlierBase = leadFusionCandidate("outlier", outlierCanonical, {
    vectorizer: "dynamic-path",
  })
  const outlier = {
    ...outlierBase,
    parameters: {
      ...outlierBase.parameters,
      darkInkEnhancement: true,
    },
  }

  assert.equal(
    digitizerTestUtils.candidateHasTrustedLead(
      reference,
      [reference, agreeing],
      "V3"
    ),
    true
  )
  assert.equal(
    digitizerTestUtils.candidateHasTrustedLead(
      outlier,
      [outlier, reference],
      "V3"
    ),
    false
  )
})

test("two agreeing unlabeled neural paths cannot establish lead identity", () => {
  const unlabeled = (id: string, dynamicPath = false) => {
    const base = leadFusionCandidate(id, sixByTwoCanonical(), {
      vectorizer: dynamicPath ? "dynamic-path" : "probability-centroid",
    })
    return {
      ...base,
      sourceFidelity: undefined,
      parameters: {
        ...base.parameters,
        semanticLeadIdentityConfirmed: undefined,
        semanticLeadIdentityMethod: undefined,
        semanticLeadIdentityOrder: undefined,
        ...(dynamicPath ? { darkInkEnhancement: true } : {}),
      },
    }
  }
  const first = unlabeled("unlabeled-centroid")
  const second = unlabeled("unlabeled-path", true)

  assert.equal(
    digitizerTestUtils.candidateHasTrustedLead(
      first,
      [first, second],
      "V3"
    ),
    false
  )
})

test("explicit pre-extraction label identity can anchor an agreeing 3x4 peer", () => {
  const anchorBase = leadFusionCandidate(
    "label-grid-centroid",
    threeByFourCanonical(),
    { vectorizer: "probability-centroid" }
  )
  const anchor = {
    ...anchorBase,
    layout: "standard_3x4",
    qa: digitizerTestUtils.evaluateQa(
      anchorBase.canonical,
      "standard_3x4"
    ),
    sourceFidelity: undefined,
    parameters: {
      ...anchorBase.parameters,
      geometryConfirmedLayout: true,
      semanticLeadIdentityConfirmed: true,
      semanticLeadIdentityMethod: "standard-three-by-four-label-grid-v1",
      semanticLeadIdentityOrder: "standard" as const,
    },
  }
  const peerBase = leadFusionCandidate(
    "label-grid-path",
    threeByFourCanonical(),
    { vectorizer: "dynamic-path" }
  )
  const peer = {
    ...peerBase,
    layout: "standard_3x4",
    qa: digitizerTestUtils.evaluateQa(
      peerBase.canonical,
      "standard_3x4"
    ),
    sourceFidelity: {
      ...passedSourceFidelity,
      leadOrderValidation: undefined,
      leadLabelValidation: undefined,
      precordialLabelValidation: undefined,
      limbLabelValidation: undefined,
    },
    parameters: {
      ...peerBase.parameters,
      kind: "native-grid" as const,
      vectorizer: "native-grid-path" as const,
      semanticLeadIdentityConfirmed: undefined,
      semanticLeadIdentityMethod: undefined,
      semanticLeadIdentityOrder: undefined,
    },
  }

  assert.equal(
    digitizerTestUtils.candidateHasTrustedLead(
      anchor,
      [anchor, peer],
      "V3"
    ),
    true
  )
  assert.equal(
    digitizerTestUtils.candidateHasTrustedLead(
      {
        ...peer,
        sourceFidelity: {
          ...peer.sourceFidelity,
          passed: false,
        },
      },
      [anchor, peer],
      "V3"
    ),
    false
  )
  assert.equal(
    digitizerTestUtils.candidateHasTrustedLead(
      {
        ...anchor,
        parameters: {
          ...anchor.parameters,
          semanticLeadIdentityOrder: "cabrera" as const,
        },
      },
      [peer],
      "V3"
    ),
    false
  )
  assert.equal(
    digitizerTestUtils.candidateHasExplicitSemanticLayoutEvidence({
      ...anchor,
      parameters: {
        ...anchor.parameters,
        semanticLeadIdentityLayout: "standard_6x2",
      },
    }),
    false
  )
  assert.equal(
    digitizerTestUtils.candidateHasExplicitSemanticLayoutEvidence({
      ...anchor,
      parameters: {
        ...anchor.parameters,
        semanticLeadIdentityLayout: "standard_3x4",
      },
    }),
    true
  )
})

test("resampling the same vectorizer is not independent morphology evidence", () => {
  const nativeResolution = leadFusionCandidate("centroid-1500")
  const largerResolution = leadFusionCandidate(
    "centroid-2200",
    sixByTwoCanonical(),
    { resampleSize: 2200 }
  )

  assert.equal(
    digitizerTestUtils.candidateHasTrustedLead(
      nativeResolution,
      [nativeResolution, largerResolution],
      "V3"
    ),
    false
  )
})

test("a permissive neural trace can corroborate only a conservative native source", () => {
  const conservativeBase = leadFusionCandidate("conservative")
  const conservative = {
    ...conservativeBase,
    parameters: {
      ...conservativeBase.parameters,
      kind: "native-grid" as const,
      vectorizer: "native-grid-path" as const,
    },
    sourceFidelity: passedSourceFidelity,
  }
  const permissiveABase = leadFusionCandidate(
    "permissive-a",
    sixByTwoCanonical(),
    { labelThresh: 0.05, vectorizer: "dynamic-path" }
  )
  const permissiveA = {
    ...permissiveABase,
    parameters: {
      ...permissiveABase.parameters,
      darkInkEnhancement: true,
    },
  }
  const permissiveB = leadFusionCandidate(
    "permissive-b",
    sixByTwoCanonical(),
    { labelThresh: 0.02, vectorizer: "dynamic-path" }
  )

  assert.equal(
    digitizerTestUtils.candidateHasTrustedLead(
      conservative,
      [conservative, permissiveA],
      "V3"
    ),
    true
  )
  assert.equal(
    digitizerTestUtils.candidateHasTrustedLead(
      permissiveA,
      [permissiveA, permissiveB],
      "V3"
    ),
    false
  )
})

test("selector cannot prefer annotation safety over 12-lead scorability", () => {
  const original = candidate("default", "original")
  const agreeing = {
    ...candidate("agreeing", "original"),
    parameters: {
      ...candidate("agreeing", "original").parameters,
      kind: "native-grid" as const,
      vectorizer: "native-grid-path" as const,
    },
    sourceFidelity: passedSourceFidelity,
    score: 200,
  }
  const masked = candidate("annotation-masked", "annotation-masked", "aVF")

  const selected = digitizerTestUtils.selectSafestCandidate(
    [original, agreeing, masked],
    annotatedPreprocessing
  )

  assert.equal(selected?.id, "default")
})

test("selector returns no candidate when every result has a non-scorable lead", () => {
  const original = candidate("default", "original", "V6")
  const masked = candidate("annotation-masked", "annotation-masked", "aVF")

  assert.equal(
    digitizerTestUtils.selectSafestCandidate(
      [original, masked],
      annotatedPreprocessing
    ),
    undefined
  )
})

test("review-only selection is restricted to complete constrained-layout peers", () => {
  const conservative = {
    ...leadFusionCandidate("conservative"),
    parameters: {
      ...leadFusionCandidate("conservative").parameters,
      layoutConstraint: "standard_6x2_with_r1_ignored",
    },
    layout: "standard_6x2_with_r1_ignored",
  }
  const recovery = {
    ...leadFusionCandidate("recovery"),
    parameters: {
      ...leadFusionCandidate("recovery").parameters,
      labelThresh: 0.02,
      layoutConstraint: "standard_6x2_with_r1_ignored",
    },
    layout: "standard_6x2_with_r1_ignored",
  }

  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      conservative,
      recovery,
    ])?.id,
    "conservative"
  )
  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      {
        ...conservative,
        parameters: {
          ...conservative.parameters,
          layoutConstraint: undefined,
        },
      },
      recovery,
    ]),
    undefined
  )

})

test("review-only selection accepts two conservative candidates with the same supported layout", () => {
  const first = leadFusionCandidate("first")
  const second = leadFusionCandidate("second", sixByTwoCanonical(), {
    vectorizer: "dynamic-path",
  })

  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([first, second])?.id,
    "first"
  )
})

test("review-only selection down-weights warnings after structural gates pass", () => {
  const warningHeavy = {
    ...leadFusionCandidate("warning-heavy"),
    score: 10_000,
    scoreBreakdown: {
      errorPenalty: 0,
      warningPenalty: 10_000,
      missingPenalty: 0,
      gapPenalty: 0,
      layoutPenalty: 0,
      annotationRiskPenalty: 0,
      permissiveThresholdPenalty: 0,
      candidateDisagreementPenalty: 0,
      layoutCalibrationPenalty: 0,
    },
  }
  const calibratedRiskier = {
    ...leadFusionCandidate("calibrated-riskier"),
    score: 4_000,
    scoreBreakdown: {
      errorPenalty: 0,
      warningPenalty: 0,
      missingPenalty: 0,
      gapPenalty: 0,
      layoutPenalty: 0,
      annotationRiskPenalty: 0,
      permissiveThresholdPenalty: 0,
      candidateDisagreementPenalty: 0,
      layoutCalibrationPenalty: 4_000,
    },
  }

  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      warningHeavy,
      calibratedRiskier,
    ])?.id,
    "warning-heavy"
  )
})

test("review-only peers must agree on every expected lead", () => {
  const first = leadFusionCandidate("first")
  const disagreeingCanonical = sixByTwoCanonical()
  const v5Index = leadOrder.indexOf("V5")
  disagreeingCanonical.rows.forEach((row) => {
    if (Number.isFinite(row[v5Index])) {
      row[v5Index] *= 10
    }
  })
  const second = leadFusionCandidate(
    "second",
    disagreeingCanonical,
    { vectorizer: "dynamic-path" }
  )

  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([first, second]),
    undefined
  )
})

test("complete diagnostic peer consensus outranks a lone source-supported outlier", () => {
  const consensusCanonical = sixByTwoCanonical()
  const peerCanonical = {
    ...consensusCanonical,
    rows: consensusCanonical.rows.map((row) =>
      row.map((value) =>
        Number.isFinite(value) ? value + 4 : value
      )
    ),
  }
  const outlierCanonical = {
    ...consensusCanonical,
    rows: consensusCanonical.rows.map((row, sample) =>
      row.map((value, leadIndex) =>
        Number.isFinite(value) && [6, 8, 11].includes(leadIndex)
          ? value + (sample % 2 === 0 ? 500 : -500)
          : value
      )
    ),
  }
  const consensus = {
    ...leadFusionCandidate("consensus", consensusCanonical, {
      vectorizer: "dynamic-path",
    }),
    score: 10_000,
  }
  const peer = {
    ...leadFusionCandidate("peer", peerCanonical),
    score: 20_000,
  }
  const nativeBase = leadFusionCandidate("native-outlier", outlierCanonical)
  const native = {
    ...nativeBase,
    parameters: {
      ...nativeBase.parameters,
      kind: "native-grid" as const,
      vectorizer: "native-grid-path" as const,
    },
    sourceFidelity: passedSourceFidelity,
    score: 0,
  }

  assert.equal(
    digitizerTestUtils.candidateHasAllTrustedLeads(
      consensus,
      [consensus, peer, native]
    ),
    false
  )
  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      consensus,
      peer,
      native,
    ])?.id,
    "consensus"
  )
})

test("source-supported diagnostic remains preferred without full peer consensus", () => {
  const first = leadFusionCandidate("first")
  const disagreeingCanonical = sixByTwoCanonical()
  const v5Index = leadOrder.indexOf("V5")
  disagreeingCanonical.rows.forEach((row, sample) => {
    if (Number.isFinite(row[v5Index])) {
      row[v5Index] += sample % 2 === 0 ? 500 : -500
    }
  })
  const second = leadFusionCandidate(
    "second",
    disagreeingCanonical,
    { vectorizer: "dynamic-path" }
  )
  const nativeBase = leadFusionCandidate("native")
  const native = {
    ...nativeBase,
    parameters: {
      ...nativeBase.parameters,
      kind: "native-grid" as const,
      vectorizer: "native-grid-path" as const,
    },
    sourceFidelity: passedSourceFidelity,
    score: -1,
  }

  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      first,
      second,
      native,
    ])?.id,
    "native"
  )
})

test("diagnostic-only source timing cannot establish quantitative trust", () => {
  const neural = leadFusionCandidate(
    "neural",
    sixByTwoCanonical(),
    { vectorizer: "dynamic-path" }
  )
  const nativeBase = leadFusionCandidate("native-timing-inference")
  const native = {
    ...nativeBase,
    parameters: {
      ...nativeBase.parameters,
      kind: "native-grid" as const,
      vectorizer: "native-grid-path" as const,
    },
    sourceFidelity: {
      ...passedSourceFidelity,
      sourceTimingInference: {
        method: "row-local-source-span-plus-grid-ratio-v1",
        inferredPixelsPerMm: 6,
        rawGridPeriodPixels: 30,
        inferredGridScaleMm: 5,
        gridScaleErrorFraction: 0,
        quantitativeCalibrationConfirmed: false as const,
      },
    },
  }

  assert.equal(
    digitizerTestUtils.candidateHasAllTrustedLeads(
      neural,
      [neural, native]
    ),
    false
  )
  assert.equal(
    digitizerTestUtils.candidateHasAllTrustedLeads(
      native,
      [native, neural]
    ),
    false
  )
  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      neural,
      native,
    ])?.id,
    "native-timing-inference"
  )
})

test("different automatic crop regions cannot corroborate one another", () => {
  const first = {
    ...leadFusionCandidate("first-crop"),
    parameters: {
      ...leadFusionCandidate("first-crop").parameters,
      cropBox: { left: 0, top: 0, right: 800, bottom: 400 },
    },
  }
  const second = {
    ...leadFusionCandidate("second-crop"),
    parameters: {
      ...leadFusionCandidate("second-crop").parameters,
      cropBox: { left: 0, top: 400, right: 800, bottom: 800 },
    },
  }

  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([first, second]),
    undefined
  )
})

test("partial precordial layouts publish only their six expected leads", () => {
  const canonical = {
    leads: leadOrder,
    rows: Array.from({ length: 100 }, (_, sample) =>
      leadOrder.map((lead, leadIndex) =>
        lead.startsWith("V") ? sample + leadIndex : 999
      )
    ),
  }
  const partial = (id: string, vectorizer: "probability-centroid" | "dynamic-path") => ({
    id,
    label: id,
    localPath: id,
    status: "completed" as const,
    parameters: {
      resampleSize: 1500,
      semanticLeadIdentityConfirmed: true,
      semanticLeadIdentityMethod: "test-exact-precordial-recognizer",
      semanticLeadIdentityOrder: "standard" as const,
      inputVariant: "original" as const,
      vectorizer,
      device: "cpu" as const,
    },
    layout: "precordial_6x1",
    score: 0,
    qa: digitizerTestUtils.evaluateQa(canonical, "precordial_6x1"),
    canonical,
    sourceFidelity: {
      ...semanticLayoutFidelity,
      leadLabelValidation: undefined,
      leadOrderValidation: undefined,
      limbLabelValidation: undefined,
      precordialLabelValidation: {
        passed: true,
        method: "test-precordial-labels",
        semanticIdentityConfirmed: true,
      },
    },
  })

  const selected = digitizerTestUtils.selectPartialLeadCandidate([
    partial("centroid", "probability-centroid"),
    partial("path", "dynamic-path"),
  ])
  assert.equal(selected?.id, "centroid")
  const sanitized = digitizerTestUtils.sanitizeCanonicalForLayout(
    canonical,
    "precordial_6x1"
  )
  assert.ok(
    sanitized.rows.every((row) =>
      leadOrder.slice(0, 6).every((_, index) => Number.isNaN(row[index]))
    )
  )
  assert.ok(
    sanitized.rows.every((row) =>
      leadOrder.slice(6).every((_, index) => Number.isFinite(row[index + 6]))
    )
  )
})

test("crop proposals select content crops but retain composite splits for audit only", () => {
  const proposals = digitizerTestUtils.deterministicCropProposals(
    {
      ...annotatedPreprocessing,
      source: { width: 1000, height: 1000, format: "JPEG" },
    },
    {
      contentBox: { left: 100, top: 100, right: 900, bottom: 900 },
      contentConfidence: 0.9,
    }
  )

  assert.equal(proposals.find((proposal) => proposal.id === "content")?.selectionEligible, true)
  assert.equal(proposals.find((proposal) => proposal.id === "upper-panel")?.selectionEligible, false)
  assert.equal(proposals.find((proposal) => proposal.id === "lower-panel")?.selectionEligible, false)
})

test("review-only constrained selection can retain explicit gaps below the publication threshold", () => {
  const reviewCanonical = sixByTwoCanonical()
  const v3Index = leadOrder.indexOf("V3")
  for (let sample = 88; sample < 100; sample += 1) {
    reviewCanonical.rows[sample][v3Index] = Number.NaN
  }
  const conservative = {
    ...leadFusionCandidate("conservative-review", reviewCanonical),
    parameters: {
      ...leadFusionCandidate("conservative-review").parameters,
      layoutConstraint: "standard_6x2_with_r1_ignored",
    },
    layout: "standard_6x2_with_r1_ignored",
  }
  const recovery = {
    ...conservative,
    id: "recovery-review",
    parameters: {
      ...conservative.parameters,
      labelThresh: 0.02,
    },
  }

  assert.equal(conservative.qa.leads.V3.finiteSamples, 38)
  assert.equal(conservative.qa.passed, false)
  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      conservative,
      recovery,
    ])?.id,
    "conservative-review"
  )
})

test("review-only selection retains a complete 3x4 trace when native layout evidence is strong", () => {
  const reviewCanonical = threeByFourCanonical({ gapLead: "V1" })
  const nativeCanonical = threeByFourCanonical({ truncatedLead: "V5" })
  const review = {
    id: "ink-path-2200",
    label: "Source-ink fidelity extraction",
    localPath: "ink-path-2200",
    status: "completed" as const,
    parameters: {
      resampleSize: 2200,
      semanticLeadIdentityConfirmed: true,
      semanticLeadIdentityMethod: "test-exact-page-label-recognizer",
      semanticLeadIdentityOrder: "standard" as const,
      inputVariant: "original" as const,
      vectorizer: "dynamic-path" as const,
      device: "cpu" as const,
    },
    layout: "standard_3x4_with_r1",
    score: 100,
    qa: digitizerTestUtils.evaluateQa(
      reviewCanonical,
      "standard_3x4_with_r1"
    ),
    canonical: reviewCanonical,
  }
  const native = {
    id: "three-by-four-rhythm-native-grid-path",
    label: "Native grid-aware deterministic extraction",
    localPath: "three-by-four-rhythm-native-grid-path",
    status: "completed" as const,
    parameters: {
      resampleSize: 750,
      layoutConstraint: "standard_3x4_with_r1",
      semanticLeadIdentityConfirmed: true,
      semanticLeadIdentityMethod: "test-exact-page-label-recognizer",
      semanticLeadIdentityOrder: "standard" as const,
      inputVariant: "original" as const,
      vectorizer: "native-grid-path" as const,
      device: "cpu" as const,
    },
    layout: "standard_3x4_with_r1",
    score: 200,
    qa: digitizerTestUtils.evaluateQa(
      nativeCanonical,
      "standard_3x4_with_r1"
    ),
    canonical: nativeCanonical,
    sourceFidelity: {
      ...passedSourceFidelity,
      passed: false,
      layoutConfidence: 0.81,
      minimumCoverage: 0.72,
      minimumRequiredCoverage: 0.8,
    },
  }

  assert.equal(review.qa.passed, false)
  assert.equal(native.qa.leads.V5.finiteSamples, 72)
  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      native,
      review,
    ])?.id,
    "ink-path-2200"
  )
  assert.ok(
    Number.isNaN(
      review.canonical.rows[245][leadOrder.indexOf("V1")]
    )
  )
  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      {
        ...native,
        sourceFidelity: {
          ...native.sourceFidelity,
          layoutConfidence: 0.7,
        },
      },
      review,
    ]),
    undefined
  )

})

test("source-pixel fidelity can corroborate a lone constrained review candidate", () => {
  const verified = {
    ...leadFusionCandidate("native-grid"),
    parameters: {
      ...leadFusionCandidate("native-grid").parameters,
      layoutConstraint: "standard_6x2_with_r1_ignored",
      vectorizer: "native-grid-path" as const,
    },
    layout: "standard_6x2_with_r1_ignored",
    sourceFidelity: passedSourceFidelity,
  }

  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([verified])?.id,
    "native-grid"
  )
})

test("review-only selection retains a rhythm-corroborated stable 3x4 trace over grayscale grid lines", () => {
  const canonical = threeByFourCanonical()
  const leadIiIndex = leadOrder.indexOf("II")
  canonical.rows.forEach((row, sample) => {
    row[leadIiIndex] = sample
  })
  const qa = digitizerTestUtils.evaluateQa(
    canonical,
    "standard_3x4_with_r1"
  )
  const leadMetrics = Object.fromEntries(
    leadOrder.map((lead) => [
      lead,
      {
        evidenceMedian: 1,
        evidenceP10: lead === "I" ? 0.02 : 1,
        maxNativeJumpPixels: 40,
        p95NativeJumpPixels: 7,
        coverage: 0.94,
        sourceStartPixel: 20,
        sourceEndPixel: 550,
        homeRowFraction: 1,
        unsupportedLargeJumpCount: 0,
        unsafeExcursionCount: 0,
      },
    ])
  )
  const native = {
    ...leadFusionCandidate("stable-rhythm-native", canonical),
    parameters: {
      ...leadFusionCandidate("stable-rhythm-native", canonical).parameters,
      layoutConstraint: "standard_3x4_with_r1",
      vectorizer: "native-grid-path" as const,
    },
    layout: "standard_3x4_with_r1",
    qa,
    sourceFidelity: {
      ...passedSourceFidelity,
      passed: false,
      method: "rhythm-anchored-connected-multievidence-native-path-v6",
      layoutConfidence: 0.96,
      evidenceMedian: 1,
      evidenceP10: 0.02,
      minimumCoverage: 0.94,
      minimumHomeRowFraction: 1,
      rhythmAnchorCount: 9,
      rhythmTracePassed: true,
      unsafeExcursionCount: 0,
      unsupportedLargeJumpCount: 0,
      maximumUnsupportedLargeJumps: 2,
      leadMetrics,
    },
  }

  assert.equal(qa.passed, true)
  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([native])?.id,
    "stable-rhythm-native"
  )
  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      {
        ...native,
        sourceFidelity: {
          ...native.sourceFidelity,
          rhythmTracePassed: false,
        },
      },
    ]),
    undefined
  )

  const labelValidatedForPublication = {
    ...native,
    sourceFidelity: {
      ...native.sourceFidelity,
      passed: true,
      method: "rhythm-anchored-label-validated-three-by-four-v1",
      layoutConfidence: 0.42,
      leadLabelValidation: {
        passed: true,
        order: "standard" as const,
        method: "test-value-aware-lead-label-recognizer",
        semanticIdentityConfirmed: true,
      },
    },
  }
  assert.equal(
    digitizerTestUtils.candidateHasSourceVerifiedBoundaryLead(
      labelValidatedForPublication,
      "I"
    ),
    true
  )
  assert.equal(
    digitizerTestUtils.candidateHasSourceVerifiedBoundaryLead(
      {
        ...labelValidatedForPublication,
        sourceFidelity: {
          ...labelValidatedForPublication.sourceFidelity,
          leadLabelValidation: {
            ...labelValidatedForPublication.sourceFidelity.leadLabelValidation,
            passed: false,
          },
        },
      },
      "I"
    ),
    false
  )
})

test("review-only native layout constraints still require credible source evidence", () => {
  const candidate = {
    ...leadFusionCandidate("native-grid-review"),
    parameters: {
      ...leadFusionCandidate("native-grid-review").parameters,
      layoutConstraint: "standard_6x2",
      vectorizer: "native-grid-path" as const,
    },
    layout: "standard_6x2",
    sourceFidelity: {
      ...passedSourceFidelity,
      passed: false,
      layoutConfidence: 0.36,
      evidenceMedian: 0.82,
      evidenceP10: 0.37,
      minimumHomeRowFraction: 0.82,
    },
  }

  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([candidate])?.id,
    "native-grid-review"
  )
  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      {
        ...candidate,
        sourceFidelity: {
          ...candidate.sourceFidelity,
          layoutConfidence: 0.25,
          evidenceMedian: 0.38,
          evidenceP10: 0.005,
        },
      },
    ]),
    undefined
  )
})

test("label-anchored twelve-row recovery tolerates one weak but stable trace", () => {
  const canonical = {
    leads: leadOrder,
    rows: Array.from({ length: 100 }, (_, sample) =>
      leadOrder.map((_, leadIndex) => sample + leadIndex)
    ),
  }
  const strongLead = {
    evidenceMedian: 0.95,
    evidenceP10: 0.7,
    maxNativeJumpPixels: 28,
    p95NativeJumpPixels: 7,
    coverage: 0.96,
    sourceStartPixel: 40,
    sourceEndPixel: 760,
    homeRowFraction: 0.9,
    unsupportedLargeJumpCount: 1,
  }
  const weakLead = {
    ...strongLead,
    evidenceMedian: 0.12,
    evidenceP10: 0.02,
    p95NativeJumpPixels: 6,
    homeRowFraction: 0.99,
    unsupportedLargeJumpCount: 0,
  }
  const leadMetrics = Object.fromEntries(
    leadOrder.map((lead) => [lead, lead === "V4" ? weakLead : strongLead])
  )
  const legacyCandidate = {
    ...leadFusionCandidate("label-anchored-twelve", canonical),
    parameters: {
      ...leadFusionCandidate("label-anchored-twelve", canonical).parameters,
      layoutConstraint: "standard_12x1",
      vectorizer: "native-grid-path" as const,
      semanticLeadIdentityConfirmed: undefined,
      semanticLeadIdentityMethod: undefined,
      semanticLeadIdentityOrder: undefined,
    },
    layout: "standard_12x1",
    qa: digitizerTestUtils.evaluateQa(canonical, "standard_12x1"),
    sourceFidelity: {
      ...passedSourceFidelity,
      passed: false,
      method: "row-local-sequential-label-validated-v3",
      layoutConfidence: 0.84,
      evidenceMedian: 0.95,
      evidenceP10: 0.02,
      minimumCoverage: 0.9,
      minimumHomeRowFraction: 0.82,
      unsupportedLargeJumpCount: 12,
      maximumUnsupportedLargeJumps: 95,
      leadOrderValidation: {
        passed: true,
        selectedOrder: "standard",
        method: "sequential-lead-label-anchors-v2",
      },
      leadLabelValidation: {
        passed: true,
        order: "standard",
        method: "test-value-aware-sequential-label-recognizer",
        semanticIdentityConfirmed: true,
      },
      leadMetrics,
    },
  }

  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      legacyCandidate,
    ]),
    undefined
  )
  const provisionalLayoutCandidate = {
    ...legacyCandidate,
    parameters: {
      ...legacyCandidate.parameters,
      geometryConfirmedLayout: true,
      geometryLayoutConfidence: 0.91,
      inputVariant: "geometry-corrected" as const,
    },
    sourceFidelity: {
      ...passedSourceFidelity,
      layoutConfidence: 0.91,
    },
  }
  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      provisionalLayoutCandidate,
    ])?.id,
    "label-anchored-twelve"
  )
  assert.equal(
    digitizerTestUtils.candidateHasExplicitSemanticLayoutEvidence(
      provisionalLayoutCandidate
    ),
    false
  )
  assert.equal(
    digitizerTestUtils.candidateHasProvisionalLayoutLeadIdentity(
      provisionalLayoutCandidate
    ),
    true
  )
  const geometryDerivedModel = {
    ...provisionalLayoutCandidate,
    id: "geometry-derived-model",
    parameters: {
      ...provisionalLayoutCandidate.parameters,
      kind: "adaptive-preprocessed" as const,
      layoutConstraint: undefined,
      geometryConfirmedLayout: undefined,
      geometryLayoutConfidence: undefined,
      inputVariant: "artifact-preprocessed" as const,
      vectorizer: "probability-centroid" as const,
      artifactPreprocessingEligible: true,
    },
    sourceFidelity: undefined,
  }
  assert.equal(
    digitizerTestUtils.candidateInheritsProvisionalGeometryLeadIdentity(
      geometryDerivedModel,
      [geometryDerivedModel, provisionalLayoutCandidate]
    ),
    true
  )
  assert.equal(
    digitizerTestUtils.candidateHasSemanticLeadIdentity(
      geometryDerivedModel,
      [geometryDerivedModel, provisionalLayoutCandidate],
      "V4"
    ),
    true
  )
  assert.equal(
    digitizerTestUtils.candidateInheritsProvisionalGeometryLeadIdentity(
      {
        ...geometryDerivedModel,
        parameters: {
          ...geometryDerivedModel.parameters,
          inputVariant: "original" as const,
        },
      },
      [geometryDerivedModel, provisionalLayoutCandidate]
    ),
    false
  )
  assert.equal(
    digitizerTestUtils.candidateInheritsProvisionalGeometryLeadIdentity(
      {
        ...geometryDerivedModel,
        parameters: {
          ...geometryDerivedModel.parameters,
          inputVariant: "preprocessed" as const,
        },
      },
      [geometryDerivedModel, provisionalLayoutCandidate]
    ),
    true
  )
  const candidate = {
    ...legacyCandidate,
    parameters: {
      ...legacyCandidate.parameters,
      semanticLeadIdentityConfirmed: true,
      semanticLeadIdentityMethod: "test-exact-sequential-recognizer",
      semanticLeadIdentityOrder: "standard" as const,
    },
  }

  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([candidate])?.id,
    "label-anchored-twelve"
  )
  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      {
        ...candidate,
        sourceFidelity: {
          ...candidate.sourceFidelity,
          method: "row-local-sequential-label-validated-v2",
        },
      },
    ])?.id,
    "label-anchored-twelve"
  )
  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      {
        ...candidate,
        sourceFidelity: {
          ...candidate.sourceFidelity,
          leadMetrics: {
            ...leadMetrics,
            V4: { ...weakLead, unsupportedLargeJumpCount: 2 },
          },
        },
      },
    ]),
    undefined
  )
  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      {
        ...candidate,
        sourceFidelity: {
          ...candidate.sourceFidelity,
          leadMetrics: {
            ...leadMetrics,
            V3: weakLead,
          },
        },
      },
    ]),
    undefined
  )
})

test("source-verified output is not marked uncertain by unverified model traces", () => {
  const verified = {
    ...leadFusionCandidate("native-grid"),
    sourceFidelity: passedSourceFidelity,
  }
  const conflictingCanonical = {
    ...sixByTwoCanonical(),
    rows: sixByTwoCanonical().rows.map((row) =>
      row.map((value) => (Number.isFinite(value) ? value + 500 : value))
    ),
  }
  const conflicting = leadFusionCandidate("model", conflictingCanonical)
  const uncertainty = digitizerTestUtils.buildUncertaintyRows(
    verified,
    [verified, conflicting],
    {
      ...annotatedPreprocessing,
      annotationMask: {
        maskedPixels: 0,
        maskedFraction: 0,
        components: [],
      },
    }
  )

  assert.ok(uncertainty.every((row) => row.candidateCount === 1))
  assert.ok(
    uncertainty.every(
      (row) => row.status !== "uncertain_candidate_disagreement"
    )
  )
})

test("structurally incomplete peer leads do not create disagreement uncertainty", () => {
  const selected = leadFusionCandidate("selected")
  const incompleteCanonical = sixByTwoCanonical(undefined, "V3")
  const leadIndex = incompleteCanonical.leads.indexOf("V3")
  incompleteCanonical.rows = incompleteCanonical.rows.map((row) =>
    row.map((value, index) => {
      if (index !== leadIndex || !Number.isFinite(value)) return value
      return value + 500
    })
  )
  const incompletePeer = leadFusionCandidate(
    "incomplete-peer",
    incompleteCanonical
  )

  assert.equal(
    digitizerTestUtils.candidateHasStructurallyCompleteLead(
      incompletePeer,
      "V3"
    ),
    false
  )
  const uncertainty = digitizerTestUtils.buildUncertaintyRows(
    selected,
    [selected, incompletePeer],
    {
      ...annotatedPreprocessing,
      annotationMask: {
        maskedPixels: 0,
        maskedFraction: 0,
        components: [],
      },
    }
  )
  const v3Rows = uncertainty.filter((row) => row.lead === "V3")
  assert.ok(v3Rows.length > 0)
  assert.ok(v3Rows.every((row) => row.candidateCount === 1))
  assert.ok(
    v3Rows.every(
      (row) => row.status !== "uncertain_candidate_disagreement"
    )
  )
})

function leadFusionCandidate(
  id: string,
  canonical = sixByTwoCanonical(),
  parameters: {
    resampleSize?: number
    upscaleToMaxDimension?: number
    labelThresh?: number
    vectorizer?: "probability-centroid" | "dynamic-path"
  } = {}
) {
  return {
    id,
    label: id,
    localPath: id,
    status: "completed" as const,
    parameters: {
      resampleSize: parameters.resampleSize ?? 1500,
      ...(parameters.upscaleToMaxDimension
        ? { upscaleToMaxDimension: parameters.upscaleToMaxDimension }
        : {}),
      ...(parameters.labelThresh
        ? { labelThresh: parameters.labelThresh }
        : {}),
      semanticLeadIdentityConfirmed: true,
      semanticLeadIdentityMethod: "test-exact-page-label-recognizer",
      semanticLeadIdentityOrder: "standard" as const,
      inputVariant: "original" as const,
      vectorizer: parameters.vectorizer ?? "probability-centroid",
      device: "cpu" as const,
    },
    layout: "standard_6x2",
    effectiveSampleRateHz: 150,
    score: 0,
    qa: digitizerTestUtils.evaluateQa(canonical, "standard_6x2"),
    canonical,
    sourceFidelity: semanticLayoutFidelity,
  }
}

test("agreement requires genuinely distinct extraction evidence", () => {
  const centroid = leadFusionCandidate("centroid")
  const correlatedPath = leadFusionCandidate(
    "correlated-path",
    sixByTwoCanonical(),
    { vectorizer: "dynamic-path" }
  )
  const sourceInkPath = {
    ...correlatedPath,
    parameters: {
      ...correlatedPath.parameters,
      darkInkEnhancement: true,
    },
  }
  const preparedCentroid = {
    ...centroid,
    parameters: {
      ...centroid.parameters,
      inputVariant: "preprocessed" as const,
    },
  }

  assert.equal(
    digitizerTestUtils.candidatePathsAreIndependent(
      centroid,
      correlatedPath
    ),
    false
  )
  assert.equal(
    digitizerTestUtils.candidatePathsAreIndependent(centroid, sourceInkPath),
    false
  )
  assert.equal(
    digitizerTestUtils.candidatePathsAreIndependent(
      centroid,
      preparedCentroid
    ),
    false
  )

  const nativeGrid = {
    ...centroid,
    parameters: {
      ...centroid.parameters,
      kind: "native-grid" as const,
      vectorizer: "native-grid-path" as const,
    },
  }
  assert.equal(
    digitizerTestUtils.candidatePathsAreIndependent(centroid, nativeGrid),
    true
  )
})

test("per-lead consensus prefers native centroid morphology over an upscaled trace", () => {
  const native = leadFusionCandidate("native")
  const upscaled = leadFusionCandidate("upscaled", sixByTwoCanonical(), {
    resampleSize: 2200,
    upscaleToMaxDimension: 2200,
  })

  const selected = digitizerTestUtils.selectLeadSourceCandidate(
    [native, upscaled],
    { ...annotatedPreprocessing, annotationMask: { ...annotatedPreprocessing.annotationMask, maskedPixels: 0 } },
    "standard_6x2",
    "V3"
  )

  assert.equal(selected?.id, "native")
})

test("review selection preserves strict source timing against agreeing neural peers", () => {
  const sourceCanonical = sixByTwoCanonical()
  sourceCanonical.rows = sourceCanonical.rows.map((row, sample) =>
    row.map((value, lead) => Number.isFinite(value)
      ? 200 * Math.sin(sample * 0.33) + lead * 20
      : Number.NaN)
  )
  const neuralCanonical = {
    leads: leadOrder,
    rows: sourceCanonical.rows.map((row) => row.map((value) => value * -4)),
  }
  const nativeBase = leadFusionCandidate("strict-source", sourceCanonical)
  const native = {
    ...nativeBase,
    parameters: {
      ...nativeBase.parameters,
      kind: "native-grid" as const,
      vectorizer: "native-grid-path" as const,
      layoutConstraint: "standard_6x2",
    },
    sourceFidelity: {
      ...passedSourceFidelity,
      method: "rhythm-anchored-connected-multievidence-native-path-v6",
      sourcePanelTimingDetected: true,
    },
  }
  const model = { ...leadFusionCandidate("model", neuralCanonical), score: 4_000 }
  const peer = { ...leadFusionCandidate("peer", neuralCanonical), score: 5_000 }
  const choose = (candidate: typeof native) =>
    digitizerTestUtils.selectReviewableConstrainedCandidate([candidate, model, peer])?.id

  assert.equal(choose(native), "strict-source")
  assert.equal(choose({ ...native, sourceFidelity: {
    ...native.sourceFidelity, sourcePanelTimingDetected: false,
  } }), "model")
  assert.equal(choose({ ...native, sourceFidelity: {
    ...native.sourceFidelity, passed: false,
  } }), "model")
  assert.equal(choose({ ...native, sourceFidelity: {
    ...native.sourceFidelity, method: "rhythm-anchored-label-validated-three-by-four-v1",
  } }), "model")
  assert.equal(choose({ ...native, qa: { ...native.qa, passed: false } }), "model")
})

test("selection keeps an abrupt 12x1 native path as corroboration when a model is available", () => {
  const canonical = {
    leads: leadOrder,
    rows: Array.from({ length: 100 }, (_, sample) =>
      leadOrder.map((_, leadIndex) => sample + leadIndex)
    ),
  }
  const nativeBase = leadFusionCandidate("native-raster-column", canonical)
  const native = {
    ...nativeBase,
    parameters: {
      ...nativeBase.parameters,
      kind: "native-grid" as const,
      resampleSize: 1644,
      vectorizer: "native-grid-path" as const,
    },
    layout: "standard_12x1",
    qa: digitizerTestUtils.evaluateQa(canonical, "standard_12x1"),
    sourceFidelity: {
      ...passedSourceFidelity,
      method: "row-local-sequential-label-validated-v3",
      inkConnectedTransitionRecovery: {
        method: "near-continuous-vertical-source-ink-v1",
        transitionScale: 2,
        displacementRewardPerPixel: 0.015,
        maximumMissingInkFraction: 0.08,
        maximumMissingInkPixelsFloor: 2,
      },
      leadMetrics: {
        V3: {
          evidenceMedian: 0.99,
          evidenceP10: 0.9,
          maxNativeJumpPixels: 40,
          p95NativeJumpPixels: 4,
          coverage: 1,
          sourceStartPixel: 80,
          sourceEndPixel: 1580,
          largeJumpCount: 1,
          unsupportedLargeJumpCount: 0,
          minimumLargeJumpSupport: 1,
        },
      },
    },
  }
  const modelBase = leadFusionCandidate("corroborated-model", canonical, {
    resampleSize: 2200,
    upscaleToMaxDimension: 2200,
    vectorizer: "dynamic-path",
  })
  const model = {
    ...modelBase,
    layout: "standard_12x1",
    qa: digitizerTestUtils.evaluateQa(canonical, "standard_12x1"),
    score: 2_000,
  }

  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([native, model])?.id,
    "corroborated-model"
  )
  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([native])?.id,
    "native-raster-column"
  )
  const incompleteModel = {
    ...model,
    qa: {
      ...model.qa,
      leads: {
        ...model.qa.leads,
        V3: { ...model.qa.leads.V3, finiteSamples: 0, missingSamples: 100 },
      },
    },
  }
  assert.equal(
    digitizerTestUtils.selectReviewableConstrainedCandidate([
      native,
      incompleteModel,
    ])?.id,
    "native-raster-column"
  )

  assert.equal(
    digitizerTestUtils.candidateHasNativeRasterColumnSamplingRisk(native, "V3"),
    true
  )
  assert.equal(
    digitizerTestUtils.selectLeadSourceCandidate(
      [native, model],
      {
        ...annotatedPreprocessing,
        source: { width: 1644, height: 1380, format: "PNG" },
        annotationMask: {
          ...annotatedPreprocessing.annotationMask,
          maskedPixels: 0,
        },
      },
      "standard_12x1",
      "V3"
    )?.id,
    "corroborated-model"
  )
  assert.equal(
    digitizerTestUtils.candidateHasNativeRasterColumnSamplingRisk(
      {
        ...native,
        sourceFidelity: {
          ...native.sourceFidelity,
          leadMetrics: {
            V3: {
              ...native.sourceFidelity.leadMetrics.V3,
              largeJumpCount: 0,
            },
          },
        },
      },
      "V3"
    ),
    false
  )
})

test("per-lead consensus uses deterministic upscale only when native support is insufficient", () => {
  const native = leadFusionCandidate(
    "native",
    sixByTwoCanonical("aVL")
  )
  const upscaled = leadFusionCandidate("upscaled", sixByTwoCanonical(), {
    resampleSize: 2200,
    upscaleToMaxDimension: 2200,
  })

  const selected = digitizerTestUtils.selectLeadSourceCandidate(
    [native, upscaled],
    { ...annotatedPreprocessing, annotationMask: { ...annotatedPreprocessing.annotationMask, maskedPixels: 0 } },
    "standard_6x2",
    "aVL"
  )

  assert.equal(selected?.id, "upscaled")
})

test("per-lead consensus uses a complete corroborated trace over a boundary-limited native anchor", () => {
  const boundaryLimitedCanonical = sixByTwoCanonical()
  const v3Index = leadOrder.indexOf("V3")
  for (let sample = 50; sample < 59; sample += 1) {
    boundaryLimitedCanonical.rows[sample][v3Index] = Number.NaN
  }
  const native = leadFusionCandidate("native-boundary-limited", boundaryLimitedCanonical)
  const complete = leadFusionCandidate("complete-corroborated", sixByTwoCanonical(), {
    resampleSize: 2200,
    upscaleToMaxDimension: 2200,
    vectorizer: "dynamic-path",
  })

  const selected = digitizerTestUtils.selectLeadSourceCandidate(
    [native, complete],
    {
      ...annotatedPreprocessing,
      annotationMask: {
        ...annotatedPreprocessing.annotationMask,
        maskedPixels: 0,
      },
    },
    "standard_6x2",
    "V3"
  )

  assert.equal(selected?.id, "complete-corroborated")
})

test("per-lead consensus treats a genuinely low-resolution source as needing model-input upscale", () => {
  const native = leadFusionCandidate("native")
  const upscaled = leadFusionCandidate("upscaled", sixByTwoCanonical(), {
    resampleSize: 2200,
    upscaleToMaxDimension: 2200,
  })

  const selected = digitizerTestUtils.selectLeadSourceCandidate(
    [native, upscaled],
    {
      ...annotatedPreprocessing,
      source: { width: 900, height: 410, format: "PNG" },
      annotationMask: {
        ...annotatedPreprocessing.annotationMask,
        maskedPixels: 0,
      },
    },
    "standard_6x2",
    "V3"
  )

  assert.equal(selected?.id, "upscaled")
})

test("compatible layout selection uses completeness and layout cost before ablation count", () => {
  const correct = [
    leadFusionCandidate("upscaled-a"),
    leadFusionCandidate("upscaled-b"),
  ].map((result) => ({
    ...result,
    layout: "standard_6x2",
    layoutCost: 0.01,
  }))
  const duplicatedWrongLayout = ["native-a", "native-b", "native-c"].map(
    (id) => ({
      ...leadFusionCandidate(id),
      layout: "standard_3x4",
      layoutCost: 0.26,
    })
  )

  const selected = digitizerTestUtils.largestCompatibleCandidateGroup([
    ...duplicatedWrongLayout,
    ...correct,
  ])

  assert.deepEqual(
    selected.map((candidate) => candidate.id),
    ["upscaled-a", "upscaled-b"]
  )
})

test("lead fusion copies complete observed lead segments without bridging or averaging", () => {
  const nativeCanonical = sixByTwoCanonical()
  nativeCanonical.rows[70][leadOrder.indexOf("V3")] = 777
  const upscaledCanonical = sixByTwoCanonical()
  upscaledCanonical.rows[20][leadOrder.indexOf("aVL")] = -333
  const native = leadFusionCandidate("native", nativeCanonical)
  const upscaled = leadFusionCandidate("upscaled", upscaledCanonical, {
    resampleSize: 2200,
    upscaleToMaxDimension: 2200,
  })
  const sources = Object.fromEntries(
    leadOrder.map((lead) => [lead, lead === "aVL" ? upscaled : native])
  )

  const fused = digitizerTestUtils.fuseLeadCandidates(
    nativeCanonical,
    "standard_6x2",
    sources
  )

  assert.equal(fused.rows[70][leadOrder.indexOf("V3")], 777)
  assert.equal(fused.rows[20][leadOrder.indexOf("aVL")], -333)
  assert.ok(Number.isNaN(fused.rows[70][leadOrder.indexOf("aVL")]))
})

test("lead-fusion selection cost scales with explicit disagreement samples", () => {
  assert.equal(
    digitizerTestUtils.fusionDisagreementPenalty([
      { status: "observed" },
      { status: "uncertain_annotation" },
      { status: "uncertain_candidate_disagreement" },
      { status: "uncertain_annotation_and_disagreement" },
    ]),
    2
  )
})

test("lead fusion preserves a full-width rhythm trace only with independent corroboration", () => {
  const rhythmCanonical = threeByFourCanonical()
  const leadIiIndex = leadOrder.indexOf("II")
  rhythmCanonical.rows.forEach((row, sample) => {
    row[leadIiIndex] = sample * 2
  })
  const neural = {
    ...leadFusionCandidate("rhythm-neural", rhythmCanonical),
    layout: "standard_3x4_with_r1",
    qa: digitizerTestUtils.evaluateQa(
      rhythmCanonical,
      "standard_3x4_with_r1"
    ),
  }
  const native = {
    ...leadFusionCandidate("rhythm-native", rhythmCanonical),
    parameters: {
      ...leadFusionCandidate("rhythm-native", rhythmCanonical).parameters,
      layoutConstraint: "standard_3x4_with_r1",
      vectorizer: "native-grid-path" as const,
    },
    layout: "standard_3x4_with_r1",
    qa: digitizerTestUtils.evaluateQa(
      rhythmCanonical,
      "standard_3x4_with_r1"
    ),
    sourceFidelity: passedSourceFidelity,
  }
  const candidates = [neural, native]
  assert.equal(
    digitizerTestUtils.candidateHasCorroboratedFullWidthRhythmLead(
      neural,
      candidates,
      "II"
    ),
    true
  )
  assert.equal(
    digitizerTestUtils.candidateHasCorroboratedFullWidthRhythmLead(
      neural,
      [neural],
      "II"
    ),
    false
  )

  const sources = Object.fromEntries(
    leadOrder.map((lead) => [lead, neural])
  )
  const panelOnly = digitizerTestUtils.fuseLeadCandidates(
    rhythmCanonical,
    "standard_3x4_with_r1",
    sources
  )
  const rhythmPreserved = digitizerTestUtils.fuseLeadCandidates(
    rhythmCanonical,
    "standard_3x4_with_r1",
    sources,
    new Set(["II"])
  )

  assert.ok(Number.isNaN(panelOnly.rows[350][leadIiIndex]))
  assert.equal(rhythmPreserved.rows[350][leadIiIndex], 700)
  assert.ok(Number.isNaN(rhythmPreserved.rows[350][leadOrder.indexOf("I")]))
})

test("borderline MPS confirmation assigns every selected lead to a repeatable path", () => {
  const selected = {
    ...leadFusionCandidate("mps-selected"),
    parameters: {
      ...leadFusionCandidate("mps-selected").parameters,
      device: "mps" as const,
    },
  }

  const assignments = digitizerTestUtils.mpsConfirmationAssignments(
    selected,
    [selected]
  )

  assert.equal(assignments.size, 1)
  assert.deepEqual(assignments.get(selected), leadOrder)
})

test("CPU stability confirmation preserves the extraction recipe but is audit-only", () => {
  const source = {
    ...leadFusionCandidate("artifact-source"),
    parameters: {
      ...leadFusionCandidate("artifact-source").parameters,
      inputVariant: "artifact-preprocessed" as const,
      artifactPreprocessingEligible: true,
      device: "mps" as const,
    },
  }

  const confirmation = digitizerTestUtils.stabilityCandidateConfig(
    source,
    "cpu-confirmation",
    "cpu"
  )

  assert.equal(confirmation.device, "cpu")
  assert.equal(confirmation.selectionEligible, false)
  assert.equal(confirmation.inputVariant, "artifact-preprocessed")
  assert.equal(confirmation.artifactPreprocessingEligible, true)
  assert.equal(confirmation.resampleSize, source.parameters.resampleSize)
})

test("neural stability cannot erase an explicitly partial review extraction", () => {
  assert.equal(
    digitizerTestUtils.stabilityRequiredForPublicationOutcome("partial"),
    false
  )
  assert.equal(
    digitizerTestUtils.stabilityRequiredForPublicationOutcome("needs_review"),
    true
  )
})

test("publication ranking does not let duplicate boundary warnings overwhelm paired-truth quality", () => {
  const boundaryWarningCandidate = {
    score: 41_578,
    scoreBreakdown: {
      errorPenalty: 0,
      warningPenalty: 40_000,
      missingPenalty: 160,
      gapPenalty: 4,
      layoutPenalty: 8,
      annotationRiskPenalty: 0,
      permissiveThresholdPenalty: 0,
      candidateDisagreementPenalty: 0,
      layoutCalibrationPenalty: 1_406,
    },
  }
  const warningFreeCandidate = {
    score: 3_792,
    scoreBreakdown: {
      errorPenalty: 0,
      warningPenalty: 0,
      missingPenalty: 0,
      gapPenalty: 0,
      layoutPenalty: 8,
      annotationRiskPenalty: 0,
      permissiveThresholdPenalty: 0,
      candidateDisagreementPenalty: 354,
      layoutCalibrationPenalty: 3_430,
    },
  }

  assert.ok(
    digitizerTestUtils.candidatePublicationRank(boundaryWarningCandidate) <
      digitizerTestUtils.candidatePublicationRank(warningFreeCandidate)
  )
})
