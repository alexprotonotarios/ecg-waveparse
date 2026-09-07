ObjC.import('Foundation')
ObjC.import('Vision')

function unwrap(value) {
  return ObjC.unwrap(value)
}

// JXA invokes this global entry point; it is not called inside the module.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function run(args) {
  if (args.length < 1 || args.length > 2 || (args.length === 2 && args[1] !== '--accurate')) {
    throw new Error('usage: recognize_lead_labels.js CONTACT_SHEET [--accurate]')
  }

  const request = $.VNRecognizeTextRequest.alloc.init
  const accurate = args.length === 2
  request.recognitionLevel = accurate ? $.VNRequestTextRecognitionLevelAccurate : $.VNRequestTextRecognitionLevelFast
  request.usesLanguageCorrection = false
  request.minimumTextHeight = 0.004
  request.recognitionLanguages = $(['en-US'])
  request.customWords = $([
    'I', 'II', 'III',
    'aVR', 'aVL', 'aVF',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
    '1', '2', '3', '4', '5', '6',
    '1 2 3 4 5 6',
  ])

  const url = $.NSURL.fileURLWithPath(args[0])
  const handler = $.VNImageRequestHandler.alloc.initWithURLOptions(
    url,
    $.NSDictionary.alloc.init,
  )
  const error = Ref()
  const requests = $.NSArray.arrayWithObject(request)
  if (!handler.performRequestsError(requests, error)) {
    throw new Error(String(error[0]))
  }

  const observations = []
  const results = request.results
  for (let index = 0; index < results.count; index += 1) {
    const observation = results.objectAtIndex(index)
    const box = observation.boundingBox
    const candidates = observation.topCandidates(5)
    const candidateOutput = []
    for (let candidateIndex = 0; candidateIndex < candidates.count; candidateIndex += 1) {
      const candidate = candidates.objectAtIndex(candidateIndex)
      candidateOutput.push({
        text: unwrap(candidate.string),
        confidence: Number(candidate.confidence),
      })
    }
    observations.push({
      x: Number(box.origin.x),
      y: Number(box.origin.y),
      width: Number(box.size.width),
      height: Number(box.size.height),
      candidates: candidateOutput,
    })
  }

  return JSON.stringify({
    engine: accurate ? 'apple-vision-vnrecognizetextrequest-accurate' : 'apple-vision-vnrecognizetextrequest',
    revision: Number(request.revision),
    observations,
  })
}
