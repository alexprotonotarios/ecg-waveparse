"""Fallback for freshly observed unsupported grids; no case identity or cached input."""
import copy
from . import _local_anchor_observer, _supported_rebuild, _grid_mesh, _grid_holes
from ._grid_observer import clean

def propose(image,source_paths,grid):
    if grid.get('field',{}).get('cells') or 'guardedObservedRows' not in grid:
        return grid
    context={'sourcePaths':source_paths,'grid':grid}
    observed=_local_anchor_observer.observe(image,context)
    replay=_supported_rebuild.rebuild(context,observed['candidateGuardedObservedRows'],observed['candidateNodeSupport'])
    field=replay['proposal']['finalField']
    result=copy.deepcopy(grid)
    result['supportedGridFallback']={'localAnchorObservation':observed,'fieldReplay':replay,'diagnosticOnly':True,'sourceSeedRowsAndNodesUnchanged':True}
    if field['cells']:
        mesh=_grid_mesh.build_mesh(field['cells'])
        result.update(state='partial',field=field,mesh=mesh,extension=_grid_holes.extend_holes(mesh,grid['majorIndices'],replay['inputs']['gapPassed']))
    return clean(result)
