import { isDeepStrictEqual } from "node:util"
import { EvidenceContractError } from "./physical-contracts"
import { JOINT_PROOF_ENCODING, jointGeometryProofHash, solveJointGeometry, validateFamilyVariantTiming } from "./source-joint-timing-contract"

type RecordValue = Record<string, unknown>
type Bounds = (value: unknown) => number[]
const fail = (): never => { throw new EvidenceContractError("invalid_source_grid_evidence", "Endpoint-family timing requires every source member, universal geometry and full parent exclusions.") }
const object = (value: unknown): RecordValue => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return fail()
  return value as RecordValue
}
const array = (value: unknown, minimum: number, maximum = minimum): unknown[] => {
  if (!Array.isArray(value) || value.length < minimum || value.length > maximum) return fail()
  return value
}
const number = (value: unknown): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) return fail()
  return value
}
const integer = (value: unknown, minimum: number, maximum: number) => {
  const n = number(value)
  if (!Number.isSafeInteger(n) || n < minimum || n > maximum) fail()
  return n
}
const numbers = (value: unknown, length: number) => array(value, length).map(number)
const same = (a: unknown, b: unknown) => { if (!isDeepStrictEqual(a, b)) fail() }
const median = (values: number[]) => { const a = [...values].sort((a,b)=>a-b); return (a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2 }
const roundEven = (x: number) => { const low = Math.floor(x); return x-low===.5 ? low+(low%2) : Math.round(x) }
const heightSupported = (values: number[]) => Math.max(...values)-Math.min(...values) <= Math.max(2,roundEven(median(values)*.05))
const contains = (outer: number[], inner: number[]) => inner[0]>=outer[0] && inner[1]>=outer[1] && inner[2]<=outer[2] && inner[3]<=outer[3]
const unique = (values: unknown[]) => new Set(values.map(jointGeometryProofHash)).size === values.length

function product<T>(groups: T[][]): T[][] {
  return groups.reduce<T[][]>((rows, group)=>rows.flatMap(row=>group.map(item=>[...row,item])), [[]])
}

/** Source pixels are freshly replayed in Python. Independently check complete
 * endpoint enumeration, component binding, universal geometry and timing here. */
export function validateFamilySourceTiming(timing: RecordValue, size: number[], bounds: Bounds) {
  const joint = object(timing.jointGeometry)
  if (timing.version!==5 || timing.method!=="source-endpoint-family-corresponding-grid-row-timing-v5" ||
      timing.sourceSeparators!==undefined || timing.panelRanges!==undefined || timing.separatorWindowConsensus!==undefined || timing.allVariantPhysicalTimingExact!==true ||
      timing.geometryProofEncoding!==JOINT_PROOF_ENCODING || timing.geometryProofSha256!==jointGeometryProofHash(joint) ||
      joint.method!=="source-horizontal-stroke-endpoint-families-v1" || joint.state!=="joint_geometry_supported" ||
      joint.geometryKind!=="endpoint_family" || joint.diagnosticOnly!==true || joint.nativeExtractionReady!==false ||
      joint.truthUsed!==false || joint.allHorizontalBoundariesExact!==true || joint.parentInkMustRemainExcluded!==true ||
      joint.admittedFamilySolutions!==1 || joint.decodedRasterSha256!==timing.decodedRasterSha256 ||
      joint.originalRejectedMethod!=="joint-source-rectangle-pulse-geometry-proposal-v1") fail()
  same(joint.imageSize,size)
  const {familyProofSha256, ...unsigned} = joint
  if (familyProofSha256!==jointGeometryProofHash(unsigned)) fail()
  const groups = array(joint.familyChoices,9).map((window,index)=>array(window,1,4).map(value=>{
    const family=object(value), parent=object(family.parentComponent), parentBox=bounds(parent.bounds)
    const core=numbers(family.horizontalCore,2), observed=numbers(family.horizontalObserved,2)
    const interval=numbers(family.centerIntervalPixels,2), members=array(family.members,1,4096).map(object)
    if(family.row!==Math.floor(index/3)||family.column!==index%3+1||!unique(members.map(m=>[m.bounds,m.observedBounds]))) fail()
    integer(parent.id,1,Number.MAX_SAFE_INTEGER)
    for(const member of members){
      const box=bounds(member.bounds), footprint=bounds(member.observedBounds)
      const decomposition=member.strokeDecomposition===undefined ? undefined : object(member.strokeDecomposition)
      same(decomposition ? decomposition.parentComponent : {id:member.componentId,bounds:member.observedBounds},parent)
      same([box[0],box[2]],core); same([footprint[0],footprint[2]],observed)
      same(member.centerIntervalPixels,interval)
      if(member.row!==family.row||member.column!==family.column||!contains(parentBox,footprint)||
          !contains(footprint,box)||box[1]!==footprint[1]||box[3]!==footprint[3]||
          member.centerX!==(box[0]+box[2]-1)/2||member.centerY!==(box[1]+box[3]-1)/2||
          member.height!==box[3]-box[1]||number(member.minimumBlackColumnSupport)<.8||number(member.minimumBlackColumnSupport)>1) fail()
      const centers=[(core[0]+core[1]-1)/2,(observed[0]+observed[1]-1)/2].sort((a,b)=>a-b)
      same(interval,centers)
      if(interval[1]-interval[0]>1) fail()
    }
    const rectangles=members.map(m=>bounds(m.observedBounds))
    same(family.memberUnionBounds,[Math.min(...rectangles.map(b=>b[0])),Math.min(...rectangles.map(b=>b[1])),Math.max(...rectangles.map(b=>b[2])),Math.max(...rectangles.map(b=>b[3]))])
    same(family.topRange,[Math.min(...rectangles.map(b=>b[1])),Math.max(...rectangles.map(b=>b[1]))])
    same(family.bottomRange,[Math.min(...rectangles.map(b=>b[3])),Math.max(...rectangles.map(b=>b[3]))])
    return family
  }))
  if(groups.some(window=>!unique(window.map(f=>[f.horizontalCore,f.horizontalObserved,f.centerIntervalPixels,f.parentComponent])))||
      groups.reduce((n,g)=>n*g.length,1)>4096||
      groups.reduce((n,g)=>n*g.reduce((sum,f)=>sum+(f.members as unknown[]).length,0),1)>4096) fail()
  const selected=array(joint.selectedFamilies,9).map(object)
  if(selected.some((family,i)=>!groups[i].some(option=>isDeepStrictEqual(option,family)))) fail()
  same(timing.sourceSeparatorFamilies,selected)
  const members=product(selected.map(f=>array(f.members,1,4096).map(object)))
  const variants=array(joint.variantGeometries,members.length).map(object)
  const digests=array(timing.variantTimingDigests,members.length)
  if(timing.variantCount!==members.length) fail()
  const first=variants[0], observations=["regions","rowGridMeasurements","pulseChoices"]
  const commonKeys=["state","truthUsed","imageSize","decodedRasterSha256","physicalCalibration","paperSpeedMmPerSecond",
    "panelDurationSeconds","rounding","coordinateSpace","rowPhysicalBoundariesX","rowPanelRanges","rowRecordedRanges",
    "rhythmPhysicalBoundariesX","rhythmPanelRanges","rhythmRange","rhythmOrigin","rhythmMapping","primaryQuarterMeasurements",
    "correspondingColumnConsistency","gridConsistencyMethod","finalPanelMeasurements","rhythmQuarterMeasurements",
    "sourcePulseEdges","sourceGridContext","rowGridMeasurements","gridPixelsPerMmX"]
  const common=Object.fromEntries(commonKeys.map(key=>[key,timing[key]]))
  variants.forEach((variant,index)=>{
    observations.forEach(key=>same(variant[key],first[key]))
    same(variant.separatorChoices,members[index].map(m=>[m]))
    const projected={...common,version:4,method:"source-joint-corresponding-grid-row-timing-v4",nativeExtractionReady:timing.nativeExtractionReady,
      jointGeometry:variant,geometryProofEncoding:JOINT_PROOF_ENCODING,geometryProofSha256:jointGeometryProofHash(variant),sourceSeparators:members[index]}
    if(digests[index]!==jointGeometryProofHash(projected)) fail()
    validateFamilyVariantTiming({...projected,sourceInkSupport:timing.sourceInkSupport},size,bounds)
  })
  validateEndpointEvidence(joint,groups,first,bounds)
  const context=object(timing.sourceGridContext), rows=numbers(context.rowCenters,4)
  const scales=array(timing.rowGridMeasurements,4).map(v=>{const m=object(v);return number(m.periodPixels)/number(m.periodMm)})
  const cal=object(timing.physicalCalibration), pulses=array(first.pulseChoices,4).map(v=>object(array(v,1)[0]))
  const selections=product(groups), checks=array(joint.familyChecks,selections.length).map(object)
  const admitted: RecordValue[][]=[]
  selections.forEach((selection,fi)=>{
    const combinations=product(selection.map(f=>(f.members as RecordValue[])))
    const failures: {variantIndex:number;failureReason:string}[]=[]
    combinations.forEach((marks,vi)=>{
      const {solutions}=solveJointGeometry(marks.map(m=>[m]),scales,median(scales),pulses,rows,number(cal.pixelsPerMmY),number(timing.paperSpeedMmPerSecond))
      if(solutions.length!==1) failures.push({variantIndex:vi,failureReason:"joint-geometry-missing-or-ambiguous"})
    })
    same(checks[fi],{familyCombinationIndex:fi,variantCombinations:combinations.length,allVariantsPassed:failures.length===0,failures})
    if(!failures.length) admitted.push(selection)
  })
  if(admitted.length!==1) fail()
  same(selected,admitted[0])
}

function validateEndpointEvidence(joint: RecordValue, groups: RecordValue[][], variant: RecordValue, bounds: Bounds) {
  const receipt=object(joint.sourceEndpointEvidence), regions=array(variant.regions,13).map(object)
  if(receipt.method!=="source-endpoint-witnessed-separator-decomposition-v1"||receipt.diagnosticOnly!==true||receipt.parentInkMustRemainExcluded!==true||
      !["separator-components-missing-or-excessive","joint-geometry-missing-or-ambiguous"].includes(String(receipt.originalFailureReason))) fail()
  same(receipt.limits,{minimumReferenceMarkers:6,minimumReferenceMarkersPerRow:2,maximumLongRunsPerComponent:64,maximumEndpointPairsPerComponent:256})
  const indices=array(receipt.referenceIndices,6,9).map(v=>integer(v,0,8)), heights=numbers(receipt.referenceHeights,indices.length)
  if(indices.some((v,i)=>i>0&&v<=indices[i-1])||[0,1,2].some(row=>indices.filter(i=>Math.floor(i/3)===row).length<2)||!heightSupported(heights)) fail()
  indices.forEach((index,i)=>{
    const family=object(array(groups[index],1)[0]), member=object(array(family.members,1)[0])
    if(member.strokeDecomposition!==undefined||member.height!==heights[i]) fail()
  })
  const details=array(receipt.regions,9-indices.length).map(object)
  const changed=groups.map((_,i)=>i).filter(i=>!indices.includes(i))
  details.forEach((detail,di)=>{
    const index=changed[di], region=regions[index+4], candidates=array(detail.candidates,1,4096).map(object)
    if(detail.row!==Math.floor(index/3)||detail.column!==index%3+1) fail()
    const flattened=groups[index].flatMap(f=>f.members as RecordValue[])
    same(candidates.map(jointGeometryProofHash).sort(),flattened.map(jointGeometryProofHash).sort())
    const evidence=object(detail.evidence), entries=array(evidence.components,1,64).map(object)
    for(const entry of entries){
      const parent=object(entry.parentComponent), box=bounds(parent.bounds)
      if(!array(region.components,1,64).some(c=>isDeepStrictEqual(c,parent))) fail()
      const runs=array(entry.sourceLongRuns,0,64).map(object)
      runs.forEach((run,i)=>{
        const x=integer(run.x,box[0],box[2]-1), top=integer(run.top,box[1],box[3]-1), bottom=integer(run.bottom,top+1,box[3])
        if(bottom-top<number(region.minimumLength)||
            (i>0&&(x<number(runs[i-1].x)||(x===runs[i-1].x&&top<=number(runs[i-1].bottom))))) fail()
      })
      const tops=[...new Set(runs.map(q=>number(q.top)))].sort((a,b)=>a-b), bottoms=[...new Set(runs.map(q=>number(q.bottom)))].sort((a,b)=>a-b)
      const pairs=product([tops,bottoms]).filter(([top,bottom])=>bottom>top&&heightSupported([...heights,bottom-top]))
      if(pairs.length>256) fail()
      same(entry.endpointPairsEnumerated,pairs)
      const items=array(entry.admittedRectangles,0,4096).map(object)
      const own=candidates.filter(c=>isDeepStrictEqual(object(c.strokeDecomposition).parentComponent,parent))
      for(const member of own){
        const proof=object(member.strokeDecomposition), core=bounds(member.bounds), requested=numbers(proof.requestedVerticalBounds,2)
        if(proof.sourcePixelsAltered!==false||!pairs.some(p=>isDeepStrictEqual(p,requested))||
            requested[0]>core[1]||requested[1]<core[3]||!heightSupported([...heights,number(member.height)])) fail()
        same(proof.parentInkExclusionBounds,box)
        const upper=runs.filter(q=>number(q.x)>=core[0]&&number(q.x)<core[2]&&q.top===core[1])
        const lower=runs.filter(q=>number(q.x)>=core[0]&&number(q.x)<core[2]&&q.bottom===core[3])
        if(!upper.length||!lower.length) fail()
        same(proof.upperEndpointWitnesses,upper); same(proof.lowerEndpointWitnesses,lower)
        if(!items.some(item=>isDeepStrictEqual(item.candidateBounds,core)&&isDeepStrictEqual(item.requestedVerticalBounds,requested))) fail()
      }
      for(const item of items){
        const core=bounds(item.candidateBounds), requested=numbers(item.requestedVerticalBounds,2)
        if(!pairs.some(p=>isDeepStrictEqual(p,requested))||!own.some(m=>isDeepStrictEqual(m.bounds,core))||requested[0]>core[1]||requested[1]<core[3]) fail()
      }
    }
    if(candidates.some(c=>!entries.some(entry=>isDeepStrictEqual(entry.parentComponent,object(c.strokeDecomposition).parentComponent)))) fail()
  })
}

export function validateFamilyExclusion(entry: RecordValue, family: RecordValue, size: number[], bounds: Bounds) {
  const parent=bounds(object(family.parentComponent).bounds)
  if(entry.kind!=="observed-separator-endpoint-family-parent-footprint"||entry.allEndpointMembersRetained!==true||
      entry.memberCount!==array(family.members,1,4096).length||entry.representativeBounds!==undefined) fail()
  same(bounds(entry.bounds),[Math.max(0,parent[0]-1),parent[1],Math.min(size[0],parent[2]+1),parent[3]])
  same(entry.parentBounds,parent); same(entry.memberUnionBounds,family.memberUnionBounds)
  same(entry.horizontalCore,family.horizontalCore); same(entry.centerIntervalPixels,family.centerIntervalPixels)
}
