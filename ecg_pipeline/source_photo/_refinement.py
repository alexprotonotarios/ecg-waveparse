"""Apply the existing refinement to frozen initial paths; retain every old gap."""
import numpy as np
from ecg_pipeline.native_grid_digitizer import refine_source_rhythm_path

def refine(evidence, support, publication, old, *, row_center, row_spacing, synthetic=False):
    lo, hi = [270, 1990] if synthetic else old['range']
    top, bottom = [int(row_center) - 120, int(row_center) + 120] if synthetic else old['yBounds']
    columns = np.arange(lo, hi, dtype=np.int32)
    path = np.asarray(old['pathY'], dtype=np.int32)
    validity = np.asarray(old['valid'], dtype=bool)
    changed, retained, receipt = refine_source_rhythm_path(evidence, support, columns, path, validity, source_interval=[lo, hi], y_start=top, y_end=bottom, row_center=row_center, row_spacing=row_spacing)
    retained &= ~publication[changed, columns]
    assert not np.any(retained & ~validity)
    result = {k: old[k] for k in ['lead', 'row', 'column'] if k in old}
    result.update(range=[lo, hi], yBounds=[top, bottom], pathY=changed.tolist(), valid=retained.astype(int).tolist(), sourceColumnCount=len(columns), retainedColumns=int(retained.sum()), coverage=float(retained.mean()), refinementReceipt=receipt, changedColumns=int(np.count_nonzero(changed != path)), newlyLostColumns=int(np.count_nonzero(validity & ~retained)), oldGapsRecovered=0)
    return result
