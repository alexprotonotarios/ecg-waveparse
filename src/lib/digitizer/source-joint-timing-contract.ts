import { isDeepStrictEqual } from "node:util"
import { createHash } from "node:crypto"
import { EvidenceContractError } from "./physical-contracts"

type RecordValue = Record<string, unknown>
const fail = (): never => { throw new EvidenceContractError("invalid_source_grid_evidence", "Joint source timing must retain unique observed geometry and each row's physical mapping.") }
const object = (v: unknown): RecordValue => {
  if (!v || typeof v !== "object" || Array.isArray(v)) return fail()
  return v as RecordValue
}
const list = (v: unknown, n: number): unknown[] => {
  if (!Array.isArray(v) || v.length !== n) return fail()
  return v
}
const number = (v: unknown): number => {
  if (typeof v !== "number" || !Number.isFinite(v)) return fail()
  return v
}
const numbers = (v: unknown, n: number) => list(v,n).map(number)
const same = (a: unknown,b: unknown) => { if (!isDeepStrictEqual(a,b)) fail() }
const close = (a: unknown,b: number) => { if (Math.abs(number(a)-b)>1e-9) fail() }
const hash = (v: unknown) => typeof v === "string" && /^[a-f0-9]{64}$/.test(v)
const median = (a: number[]) => { const s=[...a].sort((a,b)=>a-b); return (s[Math.floor((s.length-1)/2)]+s[Math.floor(s.length/2)])/2 }
const spread = (a: number[]) => Math.max(...a)-Math.min(...a)
const mean = (a: number[]) => a.reduce((a,b)=>a+b,0)/a.length
const roundEven = (x: number) => { const low=Math.floor(x); return x-low===.5 ? low+(low%2) : Math.round(x) }
const int = (v: unknown,lo: number,hi: number) => { const n=number(v); if (!Number.isSafeInteger(n)||n<lo||n>hi) fail(); return n }
const supported = (v: unknown,minimum: number) => { const n=number(v); if(n<minimum||n>1) fail(); return n }

export const JOINT_PROOF_ENCODING = "typed-json-finite-f64be-v1"

/** Match Python's typed tree and binary numbers without JSON float formatting. */
export function jointGeometryProofHash(value: unknown): string {
  function tagged(v: unknown): unknown[] {
    if (v === null) return ["null"]
    if (typeof v === "boolean") return ["bool", v]
    if (typeof v === "number") {
      if (!Number.isFinite(v) || (Number.isInteger(v) && !Number.isSafeInteger(v))) return fail()
      const bytes = Buffer.alloc(8)
      bytes.writeDoubleBE(v === 0 ? 0 : v)
      return ["number", bytes.toString("hex")]
    }
    if (typeof v === "string") return ["string", v]
    if (Array.isArray(v)) return ["array", v.map(tagged)]
    if (v && typeof v === "object" && Object.getPrototypeOf(v) === Object.prototype) {
      const record = v as RecordValue
      const keys = Object.keys(record).sort((a, b) => Buffer.compare(Buffer.from(a), Buffer.from(b)))
      return ["object", keys.map(key => [key, tagged(record[key])])]
    }
    return fail()
  }
  return createHash("sha256").update(JOINT_PROOF_ENCODING + "\0" + JSON.stringify(tagged(value))).digest("hex")
}

type RowSolution = {separators: RecordValue[]; rowBoundaries: number[]; width: number; horizontalResidual: number; slope: number; intercept: number; verticalResidual: number}

/** The source-pixel proof is built/replayed in Python; independently check its bounds,
 * component provenance, bounded uniqueness and physical arithmetic here. */
export function validateJointSourceTiming(timing: RecordValue, size: number[], bounds: (v: unknown)=>number[]) {
  validateObservedJointTiming(timing, size, bounds, false)
}

/** Internal family projection: source decomposition is bound by its complete family contract. */
export function validateFamilyVariantTiming(timing: RecordValue, size: number[], bounds: (v: unknown)=>number[]) {
  validateObservedJointTiming(timing, size, bounds, true)
}

function validateObservedJointTiming(timing: RecordValue, size: number[], bounds: (v: unknown)=>number[], family: boolean) {
  if (timing.version!==4 || timing.method!=="source-joint-corresponding-grid-row-timing-v4" ||
      timing.nativeExtractionReady!==true || timing.rounding!=="nearest-integer-ties-to-even-v1" ||
      timing.coordinateSpace!=="original-source" || timing.panelRanges!==undefined ||
      timing.separatorWindowConsensus!==undefined || !hash(timing.geometryProofSha256) ||
      timing.rhythmOrigin!=="observed-rhythm-pulse-falling-stroke-right-edge-v1" ||
      timing.gridConsistencyMethod!=="four-row-corresponding-quarter-grid-v1") fail()
  const joint=object(timing.jointGeometry), context=object(timing.sourceGridContext)
  // Reports produced before this encoding retain their historical structural
  // checks. New reports bind the exact numeric proof across language transport.
  if (timing.geometryProofEncoding !== undefined &&
      (timing.geometryProofEncoding !== JOINT_PROOF_ENCODING ||
       timing.geometryProofSha256 !== jointGeometryProofHash(joint))) fail()
  const rows=numbers(context.rowCenters,4), xf=numbers(context.xFit,3), font=number(context.fontHeight)
  if (font<=0||font>256||rows[0]<0||rows[3]>=size[1]||rows.some((y,i)=>i>0&&y<=rows[i-1])) fail()
  const cal=object(timing.physicalCalibration), speed=number(timing.paperSpeedMmPerSecond)
  const ppmX=number(cal.pixelsPerMmX), ppmY=number(cal.pixelsPerMmY), amplitude=ppmY*number(cal.gainMmPerMv)
  const scale=number(cal.gridScaleMmX ?? cal.gridScaleMm), pulseStart=number(cal.pulseStartX), pulseEnd=number(cal.pulseEndX)
  if (![1,5].includes(scale)||pulseStart<=0||pulseStart>=pulseEnd||pulseEnd>size[0]||cal.gridScaleAmbiguous===true) fail()
  if (joint.version!==1||joint.method!=="joint-source-rectangle-pulse-geometry-proposal-v1"||
      joint.state!=="joint_geometry_supported"||joint.solutionCount!==1||joint.nativeExtractionReady!==false||
      joint.truthUsed!==false||joint.decodedRasterSha256!==timing.decodedRasterSha256) fail()
  same(joint.imageSize,size); same(joint.rowGridMeasurements,timing.rowGridMeasurements)
  const rowGrid=list(timing.rowGridMeasurements,4).map(object)
  const scales=rowGrid.map((m,i)=>{
    if(m.rowCenter!==rows[i]||m.periodMm!==scale) fail()
    supported(m.confidence,.18)
    const s=number(m.periodPixels)/scale
    if(s<1||s>40) fail()
    return s
  })
  const ppm=median(scales), tolerance=Math.max(2,ppm*.25)
  close(timing.gridPixelsPerMmX,ppm)
  if(spread(scales)/ppm>.02||Math.abs(ppm-ppmX)/ppm>.15) fail()

  const regions=list(joint.regions,13).map(object)
  const components=regions.map((region,index)=>{
    const pulse=index<4, row=pulse?index:Math.floor((index-4)/3), col=pulse?0:(index-4)%3+1
    const cx=xf[0]+xf[1]*row+xf[2]*col, cy=rows[row]
    const expected=pulse
      ? [Math.max(0,roundEven(pulseStart-2*ppmX)),Math.max(0,roundEven(cy-1.5*amplitude)),Math.min(size[0],roundEven(pulseEnd+2*ppmX)+1),Math.min(size[1],roundEven(cy+.75*amplitude)+1)]
      : [Math.max(0,roundEven(cx-font)),Math.max(0,roundEven(cy-2*font)),Math.min(size[0],roundEven(cx+font)+1),Math.min(size[1],roundEven(cy+2*font)+1)]
    const regionBox=bounds(region.bounds)
    same(regionBox,expected)
    if(region.kind!==(pulse?"pulse":"separator")||region.row!==row||region.column!==col||
        region.minimumLength!==Math.max(3,roundEven(pulse?amplitude*.45:font*.8))||
        !Array.isArray(region.components)||region.components.length>64) fail()
    const entries=(region.components as unknown[]).map(object), ids=new Set<number>()
    for(const comp of entries){
      const id=int(comp.id,1,Number.MAX_SAFE_INTEGER), box=bounds(comp.bounds)
      if(ids.has(id)||box[0]<regionBox[0]||box[1]<regionBox[1]||box[2]>regionBox[2]||box[3]>regionBox[3]) fail()
      ids.add(id)
      if(comp.width!==box[2]-box[0]||comp.height!==box[3]-box[1]||Number(comp.height)<Number(region.minimumLength)) fail()
      int(comp.pixels,Number(region.minimumLength),Number(comp.width)*Number(comp.height))
      same(comp.midpoint,[(box[0]+box[2]-1)/2,(box[1]+box[3]-1)/2])
    }
    return entries
  })
  const pulses=list(joint.pulseChoices,4).map((v,row)=>{
    const pulse=object(list(v,1)[0]), left=object(pulse.rising), right=object(pulse.falling)
    if(!components[row].some(c=>isDeepStrictEqual(c,left))||!components[row].some(c=>isDeepStrictEqual(c,right))||left.id===right.id||pulse.row!==row) fail()
    const a=bounds(left.bounds), b=bounds(right.bounds), lx=numbers(left.midpoint,2)[0], rx=numbers(right.midpoint,2)[0]
    const width=rx-lx, height=Math.min(a[3],b[3])-Math.max(a[1],b[1])-Math.max(number(left.width),number(right.width))
    const tol=Math.max(3,2*Math.max(number(left.width),number(right.width)),.15*height)
    if(width<=0||height<=0||Math.max(number(left.width),number(right.width))>Math.max(5,roundEven(scales[row]*.7))||
        Math.abs(width-scales[row]*speed*.2)/(scales[row]*speed*.2)>.15||Math.abs(height-amplitude)/amplitude>.15||
        Math.abs(a[1]-b[1])>tol||Math.abs(a[3]-b[3])>tol||pulse.endX!==b[2]||pulse.baselineY!==Math.min(a[3],b[3])-1) fail()
    close(pulse.widthPixels,width); close(pulse.heightPixels,height); supported(pulse.plateauSupport,.45)
    if(components[row].some(c=>c.id!==left.id&&c.id!==right.id&&number(numbers(c.midpoint,2)[0])>lx&&number(numbers(c.midpoint,2)[0])<rx&&bounds(c.bounds)[1]<=Math.max(a[1],b[1])+tol&&bounds(c.bounds)[3]>=Math.min(a[3],b[3])-tol)) fail()
    return pulse
  })
  if(Math.min(...pulses.map(p=>Math.abs(number(p.endX)-pulseEnd)))>tolerance) fail()
  const choices=list(joint.separatorChoices,9).map((v,index)=>{
    if(!Array.isArray(v)||v.length<1||v.length>4) fail()
    const seen=new Set<number>()
    return (v as unknown[]).map(value=>{
      const mark=object(value), id=int(mark.componentId,1,Number.MAX_SAFE_INTEGER)
      const decomposition=family && mark.strokeDecomposition!==undefined ? object(mark.strokeDecomposition) : undefined
      const parent=decomposition ? object(decomposition.parentComponent) : undefined
      const component=components[index+4].find(c=>c.id===(parent ? parent.id : id)) ?? fail()
      if(seen.has(id)) fail()
      seen.add(id)
      const core=bounds(mark.bounds), footprint=bounds(mark.observedBounds), minimum=Math.max(3,roundEven(font*.08)), maximum=Math.max(4,roundEven(font*.30))
      if(parent) {
        same(parent,component)
        const parentBox=bounds(parent.bounds)
        if(footprint[0]<parentBox[0]||footprint[1]<parentBox[1]||footprint[2]>parentBox[2]||footprint[3]>parentBox[3]) fail()
      } else same(footprint,component.bounds)
      if(mark.row!==Math.floor(index/3)||mark.column!==index%3+1||core[0]<footprint[0]||core[2]>footprint[2]||core[1]!==footprint[1]||core[3]!==footprint[3]||
          number(component.width)<minimum||number(component.width)>maximum||core[2]-core[0]<minimum||core[2]-core[0]>maximum) fail()
      const center=(core[0]+core[2]-1)/2, observed=(footprint[0]+footprint[2]-1)/2
      if(Math.abs(center-observed)>1||mark.centerX!==center||mark.centerY!==(core[1]+core[3]-1)/2||mark.height!==core[3]-core[1]) fail()
      same(mark.centerIntervalPixels,[Math.min(center,observed),Math.max(center,observed)])
      supported(mark.minimumBlackColumnSupport,.8)
      return mark
    })
  })
  const count=choices.reduce((n,c)=>n*c.length,1)
  if(count>4096||joint.candidateCombinations!==count) fail()
  const {rowChoices, solutions}=solveJointGeometry(choices,scales,ppm,pulses,rows,ppmY,speed)
  same(joint.rowProposalCounts,rowChoices.map(c=>c.length))
  if(solutions.length!==1) fail()
  const selected=object(joint.selected), solution=solutions[0], selectedRows=list(selected.rows,3).map(object)
  for(const key of ["rhythmOriginPrediction","rowOriginShear","originShearResidual"] as const) close(selected[key],solution[key])
  selectedRows.forEach((row,i)=>{
    same(row.separators,solution.rows[i].separators)
    numbers(row.rowBoundaries,5).forEach((x,j)=>close(x,solution.rows[i].rowBoundaries[j]))
    for(const key of ["width","horizontalResidual","slope","intercept","verticalResidual"] as const) close(row[key],solution.rows[i][key])
  })
  same(timing.sourceSeparators,solution.rows.flatMap(r=>r.separators))
  validateLocalTiming(timing,size,bounds,rows,scale,speed,solution.rows,pulses,scales)
}

/** Identical geometric gates for each endpoint combination; no convenient member is selected. */
export function solveJointGeometry(choices: RecordValue[][], scales: number[], ppm: number, pulses: RecordValue[], rows: number[], ppmY: number, speed: number) {
  const tolerance=Math.max(2,ppm*.25)
  const rowChoices: RowSolution[][]=[]
  for(let row=0;row<3;row++){
    const valid: RowSolution[]=[]
    for(const first of choices[row*3]) for(const second of choices[row*3+1]) for(const third of choices[row*3+2]){
      const marks=[first,second,third], heights=marks.map(m=>number(m.height)), xs=marks.map(m=>number(m.centerX)), ys=marks.map(m=>number(m.centerY))
      if(spread(heights)>Math.max(2,roundEven(median(heights)*.05))) continue
      const [a,b,c]=xs.map(x=>2*x), edges=Array.from({length:5},(_,j)=>(2*(4*a+b-2*c)+3*j*(c-a))/12)
      const width=(c-a)/4, residual=Math.abs(a-2*b+c)/6, expected=scales[row]*speed*2.5
      if(residual>Math.max(1,ppm*.25)||Math.abs(width-expected)/expected>.03||Math.abs(edges[0]-number(pulses[row].endX))>tolerance) continue
      const slope=(ys[2]-ys[0])/(xs[2]-xs[0]), intercept=mean(ys)-slope*mean(xs), vertical=Math.max(...ys.map((y,i)=>Math.abs(y-intercept-slope*xs[i])))
      if(!Number.isFinite(slope)||Math.abs(slope)>.05||vertical>Math.max(1,ppm*.25)||Math.abs(intercept+slope*numbers(object(pulses[row].falling).midpoint,2)[0]-number(pulses[row].baselineY))>ppmY) continue
      valid.push({separators:marks,rowBoundaries:edges,width,horizontalResidual:residual,slope,intercept,verticalResidual:vertical})
    }
    rowChoices.push(valid)
  }
  const solutions: {rows: RowSolution[]; rhythmOriginPrediction: number; rowOriginShear: number; originShearResidual: number}[]=[]
  for(const a of rowChoices[0]) for(const b of rowChoices[1]) for(const c of rowChoices[2]){
    const group=[a,b,c], heights=group.flatMap(r=>r.separators.map(m=>number(m.height)))
    if(spread(heights)>Math.max(2,roundEven(median(heights)*.05))||Array.from({length:5},(_,i)=>spread(group.map(r=>r.rowBoundaries[i]))).some(d=>d>ppm)||spread(group.map(r=>r.slope))*median(group.map(r=>r.width))>2) continue
    const origins=group.map(r=>r.rowBoundaries[0]), shear=(origins[2]-origins[0])/(rows[2]-rows[0]), intercept=mean(origins)-shear*mean(rows.slice(0,3)), predicted=intercept+shear*rows[3]
    const residual=Math.max(...origins.map((x,i)=>Math.abs(x-intercept-shear*rows[i])))
    if(residual>Math.max(1,ppm*.25)||Math.abs(predicted-number(pulses[3].endX))>tolerance) continue
    solutions.push({rows:group,rhythmOriginPrediction:predicted,rowOriginShear:shear,originShearResidual:residual})
  }
  return {rowChoices, solutions}
}

function validateLocalTiming(timing: RecordValue,size: number[],bounds: (v: unknown)=>number[],rows: number[],scale: number,speed: number,solution: RowSolution[],pulses: RecordValue[],globalScales: number[]) {
  const primary=list(timing.primaryQuarterMeasurements,3).map(v=>list(v,4).map(object)), rhythm=list(timing.rhythmQuarterMeasurements,4).map(object)
  const physical=list(timing.rowPhysicalBoundariesX,3).map(v=>numbers(v,5)), rhythmPhysical=numbers(timing.rhythmPhysicalBoundariesX,5)
  const primaryRanges=list(timing.rowPanelRanges,3).map(v=>list(v,4).map(r=>numbers(r,2))), rhythmRanges=list(timing.rhythmPanelRanges,4).map(v=>numbers(v,2))
  const radius=Math.max(8,roundEven(median(rows.slice(1).map((v,i)=>v-rows[i]))*.07)), nominal=globalScales[3]*speed*2.5, origin=number(pulses[3].endX)
  const rowBounds=[0,...rows.slice(0,3).map((v,i)=>Math.ceil((v+rows[i+1])/2)),size[1]]
  let rhythmEnd=origin
  close(rhythmPhysical[0],origin)
  for(let row=0;row<4;row++){
    const sourceEdges=row<3?solution[row].rowBoundaries:Array.from({length:5},(_,c)=>origin+c*nominal), observations=row<3?primary[row]:rhythm
    for(let column=0;column<4;column++){
      const m=observations[column], box=bounds(m.bounds), ppm=number(m.periodPixels)/scale
      same(box,[roundEven(sourceEdges[column]),Math.max(0,roundEven(rows[row])-radius),roundEven(sourceEdges[column+1]),Math.min(size[1],roundEven(rows[row])+radius+1)])
      if(m.row!==row||m.column!==column||m.periodMm!==scale||ppm<1||ppm>40||!hash(m.profileSha256)) fail()
      close(m.pixelsPerMm,ppm); supported(m.confidence,.18)
      const width=ppm*speed*2.5, expected=sourceEdges[column+1]-sourceEdges[column]
      if(Math.abs(width-expected)/expected>.03) fail()
      if(row<3){
        close(physical[row][column],sourceEdges[column])
        if(column===3){ close(m.nominalEndX,sourceEdges[4]); close(m.physicalEndX,sourceEdges[3]+width); close(physical[row][4],sourceEdges[3]+width) }
      } else { close(m.physicalStartX,rhythmEnd); rhythmEnd+=width; close(m.physicalEndX,rhythmEnd); close(rhythmPhysical[column+1],rhythmEnd) }
    }
    const edges=row<3?physical[row]:rhythmPhysical, ranges=row<3?primaryRanges[row]:rhythmRanges
    if(edges[0]<0||edges[4]>size[0]||edges.some((v,i)=>i>0&&v<=edges[i-1])) fail()
    ranges.forEach((range,col)=>{same(range,[roundEven(edges[col]),roundEven(edges[col+1])]); if(range[1]<=range[0]) fail()})
    if(row<3) same(list(timing.rowRecordedRanges,3)[row],[ranges[0][0],ranges[3][1]])
  }
  same(timing.rhythmRange,[rhythmRanges[0][0],rhythmRanges[3][1]])
  same(timing.finalPanelMeasurements,primary.map(r=>r[3]))
  list(timing.correspondingColumnConsistency,4).map(object).forEach((entry,col)=>{
    const scales=[...primary.map(row=>number(row[col].pixelsPerMm)),number(rhythm[col].pixelsPerMm)], relative=spread(scales)/median(scales)
    if(entry.column!==col||entry.maximumRelativeSpread!==.02||relative>.02) fail()
    same(entry.rowPixelsPerMm,scales); close(entry.relativeSpread,relative)
  })
  list(timing.sourcePulseEdges,4).map(object).forEach((edge,row)=>{
    const full=bounds(edge.observedBounds), core=bounds(edge.bounds), falling=bounds(object(pulses[row].falling).bounds), margin=roundEven((full[3]-full[1])*.2)
    same(full,falling)
    if(edge.row!==row||edge.sourceEndX!==full[2]||core[0]<full[0]||core[2]>full[2]||core[1]!==full[1]+margin||core[3]!==full[3]-margin||
        core[2]-core[0]>Math.max(3,roundEven(number(object(timing.physicalCalibration).pixelsPerMmX)*.7))||full[1]<rowBounds[row]||full[3]>rowBounds[row+1]||edge.centerX!==(core[0]+core[2]-1)/2) fail()
    supported(edge.minimumBlackColumnSupport,.65)
  })
  const inkRadius=Math.max(3,roundEven(median(rows.slice(1).map((v,i)=>v-rows[i]))*.12))
  list(timing.sourceInkSupport,16).map(object).forEach((m,index)=>{
    const row=Math.floor(index/4), col=index%4, range=row<3?primaryRanges[row][col]:rhythmRanges[col]
    if(m.row!==row||m.column!==col) fail()
    supported(m.fraction,.2)
    same(bounds(m.bounds),[range[0],Math.max(0,roundEven(rows[row])-inkRadius),range[1],Math.min(size[1],roundEven(rows[row])+inkRadius+1)])
  })
}
