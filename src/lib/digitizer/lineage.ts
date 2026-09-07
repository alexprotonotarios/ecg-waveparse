import { EvidenceContractError } from "@/lib/digitizer/physical-contracts"

export type LineageContributor = { candidateId: string; sourceCanonicalStartSample: number; sourceSampleCount: number;
  alignment: { referenceLength: number; timeScale: number; shiftSamples: number; subtractOffsetUv: number } }
export type IntervalLineage = { startSample: number; endSample: number; operation: "selected_samples" | "aligned_peer_median" | "missing";
  contributors: LineageContributor[] }

export function directContributor(candidateId: string, sourceCanonicalStartSample: number, sampleCount: number): LineageContributor {
  return {candidateId,sourceCanonicalStartSample,sourceSampleCount:sampleCount,alignment:{referenceLength:sampleCount,timeScale:1,shiftSamples:0,subtractOffsetUv:0}}
}

export function intervalLineage(values: readonly number[], direct: LineageContributor, replacements: IntervalLineage[] = []): IntervalLineage[] {
  const result: IntervalLineage[] = []
  let previousIdentity = ""
  values.forEach((value,index)=>{
    const replacement = replacements.find(span=>index>=span.startSample && index<span.endSample)
    const operation = Number.isFinite(value) ? replacement?.operation ?? "selected_samples" : "missing"
    const contributors = operation==="missing" ? [] : replacement?.contributors ?? [direct]
    const identity=JSON.stringify([operation,contributors])
    if (result.length && identity===previousIdentity) result[result.length-1].endSample=index+1
    else result.push({startSample:index,endSample:index+1,operation,contributors})
    previousIdentity=identity
  })
  validateIntervalLineage(result,values.length)
  return result
}

export function validateIntervalLineage(lineage: IntervalLineage[], sampleCount: number) {
  const fail=():never=>{throw new EvidenceContractError("invalid_interval_lineage","Lineage must exhaustively map available samples to bounded source intervals.")}
  if (!Array.isArray(lineage) || !Number.isSafeInteger(sampleCount) || sampleCount<1) return fail()
  let next=0
  for (const span of lineage) {
    if (!span || !Number.isSafeInteger(span.startSample) || !Number.isSafeInteger(span.endSample) || span.startSample!==next || span.endSample<=next || span.endSample>sampleCount ||
        !["selected_samples","aligned_peer_median","missing"].includes(span.operation) || !Array.isArray(span.contributors) ||
        (span.operation==="missing" ? span.contributors.length!==0 : span.contributors.length===0)) return fail()
    const ids=new Set<string>()
    for (const contributor of span.contributors) {
      if (!contributor || typeof contributor !== "object") return fail()
      const alignment=contributor.alignment
      if (typeof contributor.candidateId !== "string" || !contributor.candidateId || ids.has(contributor.candidateId) || !Number.isSafeInteger(contributor.sourceCanonicalStartSample) || contributor.sourceCanonicalStartSample<0 ||
          !Number.isSafeInteger(contributor.sourceSampleCount) || contributor.sourceSampleCount<1 || !alignment ||
          !Number.isSafeInteger(alignment.referenceLength) || alignment.referenceLength!==sampleCount || !Number.isFinite(alignment.timeScale) || alignment.timeScale<=0 ||
          !Number.isFinite(alignment.shiftSamples) || !Number.isFinite(alignment.subtractOffsetUv)) return fail()
      ids.add(contributor.candidateId)
      const center=(alignment.referenceLength-1)/2
      const first=center+(span.startSample-center)*alignment.timeScale+alignment.shiftSamples
      const last=center+(span.endSample-1-center)*alignment.timeScale+alignment.shiftSamples
      if (Math.floor(first)<0 || Math.ceil(last)>=contributor.sourceSampleCount) return fail()
    }
    next=span.endSample
  }
  if (next!==sampleCount) return fail()
}
