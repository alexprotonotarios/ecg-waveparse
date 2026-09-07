"""Freeze a current-source development evaluation from cached paired captures.

Membership is hash ordered within predeclared capture/layout strata; no result
files or truth values select membership. Existing source and truth bytes remain
unchanged. These previously exposed recordings are never called a final set.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ecg_benchmark.coordinates import contract_for_case, strict_semantics
from ecg_benchmark.evaluation import validate_independence
from ecg_benchmark.io import read_leads
from scripts.build_pmcardio_truth import recording_identity, source_acquisition

SCORER_FILES = ['scripts/waveparse-regression.mjs','scripts/evaluation-protocol.mjs','scripts/score_digitization.py','ecg_benchmark/scoring.py',
                'ecg_benchmark/coordinates.py','ecg_benchmark/measurements.py','ecg_benchmark/io.py',
                'ecg_pipeline/domain.py','config/ecg-domain.v1.json']
STRATA = [('photos_iphone','standard_3x4'),('photos_iphone','standard_6x2'),
          ('photos_scans','standard_3x4'),('photos_scans','standard_6x2'),
          ('photos_crumbles','standard_3x4'),('photos_screens','standard_3x4'),
          ('digital_data_opacity_1','standard_3x4'),('digital_data_opacity_1','standard_12x1')]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def build(manifest_path: Path, payload_path: Path, output: Path) -> dict:
    original = json.loads(manifest_path.read_text()); cases = original['cases']
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    chosen, used = [], set()
    for category, layout in STRATA:
        candidates = sorted((case for case in cases if case['strata']['artifactFamily'] == category and case['layout'] == layout
                             and case['strata']['pageCount'] == 1 and case['strata']['truthPanelSamples'] == 5000//int(layout[-1])
                             and recording_identity(case['strata']['sourceWaveformKey']) not in used),
                            key=lambda c: hashlib.sha256(('waveparse-capture-dev-20260906-v1:'+c['caseId']).encode()).hexdigest())
        if not candidates: raise ValueError('A predeclared capture stratum has no eligible unused recording.')
        chosen.append(candidates[0]); used.add(recording_identity(candidates[0]['strata']['sourceWaveformKey']))
    prepared, members = [], []
    for index, original_case in enumerate(chosen):
        case = json.loads(json.dumps(original_case)); case_id = f'capture_dev_{index+1:02d}'
        directory = output/case_id; directory.mkdir()
        artifacts = {}
        for key, name in [('image','image'),('truth','truth.csv'),('annotations','annotations.json')]:
            path = (manifest_path.parent/original_case[key]['path']).resolve()
            if sha(path) != original_case[key]['sha256']: raise ValueError('Existing capture artifact changed.')
            if key == 'image':
                with Image.open(path) as image:
                    suffix = {'JPEG': '.jpg', 'PNG': '.png', 'TIFF': '.tiff', 'WEBP': '.webp'}.get(image.format)
                if suffix is None: raise ValueError('Unsupported cached image format.')
                name += suffix
            shutil.copyfile(path,directory/name)
            artifacts[key] = {'path': f'{case_id}/{name}', 'sha256': sha(directory/name)}
        contract = contract_for_case(case)
        truth = read_leads(directory/'truth.csv')
        # This is a truth-frame consistency check, not independent identity
        # evidence for any prediction. Do not fabricate candidate identities.
        check = strict_semantics(truth,truth,contract,truth_rate=case['sampleRateHz'],candidate_rate=case['sampleRateHz'])
        if not all(segment['placementPassed'] for segment in check['segments']):
            raise ValueError('Paired truth disagrees with independently declared panel support.')
        case.update(caseId=case_id, groupId='pmcardio_'+recording_identity(case['strata']['sourceWaveformKey']),
                    split='development', truthCoordinateFrame='canonical_display',
                    segmentDurationSeconds=case['strata']['truthPanelSamples']/case['sampleRateHz'], **artifacts)
        case['strata']['lockedSplit'] = False
        case['provenance'].update(acquisition=source_acquisition(case['strata']['artifactFamily']), pretrainedTrainingOverlap='unknown',
                                  licenceEvidence='https://zenodo.org/records/13617673')
        case['evaluationProvenance'] = {'collection':'development','previouslyExposed':True,
            'truthMethod':'PMcardio source cut-out lead segments, converted from mV to uV and mapped to printed panels; rhythm rows excluded.',
            'licenceEvidence':case['provenance']['licenceEvidence'],'pretrainedTrainingOverlap':'unknown'}
        prepared.append(case)
        members.append({'caseId':case_id,'sourceId':case['sourceId'],'groupId':case['groupId'],'layout':case['layout'],
                        'artifact':case['degradation']['kind'],'acquisition':case['provenance']['acquisition'],
                        **{f'{key}Sha256':value['sha256'] for key,value in artifacts.items()}})
    protocol = {'version':1,'collection':'engineering_development','clinicalValidationUse':False,
        'dataset':'PMcardio ECG Image Database','membership':members,'pretrainedTrainingOverlap':'unknown',
        'licenceEvidence':'https://zenodo.org/records/13617673','previouslyExposed':True,
        'selection':'Hash order within eight capture/layout strata; unique underlying recordings, single pages, declared full-layout panel support; no outcome filtering.',
        'independentUnit':'recording; patient linkage not available in the cached image metadata',
        'stratificationKeys':['layout','artifact','acquisition'],
        'gates':{'globalRmseUv':{'maximum':100},'macroMeanCoverage':{'minimum':.9},'macroMeanCorrelation':{'minimum':.9}},
        'sourceLock':{'payloadManifestSha256':sha(payload_path),'scorerFiles':{p:sha(ROOT/p) for p in SCORER_FILES}},
        'originalManifestSha256':sha(manifest_path),
        'limitations':['Previously exposed development recordings; not a final cohort.','PMcardio cut-out lead truth excludes rhythm rows; no adjudicated clinical endpoint annotations.',
                       'Eight recording groups give coarse development evidence, not patient-population performance or model-training independence.']}
    # The original annotation bytes remain intact even though the evaluation
    # case receives an opaque local alias. That alias is not a label correction.
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    manifest = {'version':1,'suiteId':'capture_development_v1','protocolSha256':sha(output/'protocol.json'),
                'expectedCaseIds':[c['caseId'] for c in prepared], 'preparationFailures':[], 'cases':prepared,
                'grouping':validate_independence(prepared,require_provenance=True)}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'.keep-artifacts.json').write_text('{"version":1,"reason":"Frozen paired capture development evaluation; preserve original bytes and outcomes."}\n')
    return {'cases':len(prepared),'groups':len(used),'protocolSha256':sha(output/'protocol.json'),'manifestSha256':sha(output/'manifest.json')}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','payload','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.manifest.resolve(),args.payload.resolve(),args.output.resolve()),indent=2))
