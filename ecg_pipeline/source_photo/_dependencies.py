"""Explicit source-photo algorithm dependencies."""
from . import _crops as crops
from . import _constraints as constraints
from . import _composite as composite
from . import _label_observer as label_observer
from . import _edge_calibration as edge_calibration
from . import _pulse_fallback as pulse_fallback
from . import _period as period
from . import _windows as windows
from . import _local_timing as local_timing
from . import _calibration_observer as calibration_observer
from . import _initial_trace as initial_trace
from . import _refinement as refinement
from . import _ink as ink
from . import _masks as masks
from . import _events as events
from . import _excursions as excursions
from . import _review_policy as review_policy
from . import _path_observer as path_observer
from . import _grid_baseline as grid_baseline
from . import _grid_ridges as grid_ridges
from . import _grid_calibration as grid_calibration
from . import _grid_components as grid_components
from . import _grid_crossings as grid_crossings
from . import _grid_census as grid_census
from . import _grid_consensus as grid_consensus
from . import _grid_field as grid_field
from . import _grid_mesh as grid_mesh
from . import _grid_holes as grid_holes
from . import _grid_observer as grid_observer
from . import _vertical_ridges as vertical_ridges
from . import _ray_model as ray_model
from . import _ray_sampling as ray_sampling
from . import _converter as converter
from . import _grid_mesh as coordinate_mesh
from . import _coordinate_map as coordinate_map
from . import _curve_domain as curve_domain

PARTS = {
    "crops": crops,
    "constraints": constraints,
    "composite": composite,
    "label_observer": label_observer,
    "edge_calibration": edge_calibration,
    "pulse_fallback": pulse_fallback,
    "period": period,
    "windows": windows,
    "local_timing": local_timing,
    "calibration_observer": calibration_observer,
    "initial_trace": initial_trace,
    "refinement": refinement,
    "ink": ink,
    "masks": masks,
    "events": events,
    "excursions": excursions,
    "review_policy": review_policy,
    "path_observer": path_observer,
    "grid_baseline": grid_baseline,
    "grid_ridges": grid_ridges,
    "grid_calibration": grid_calibration,
    "grid_components": grid_components,
    "grid_crossings": grid_crossings,
    "grid_census": grid_census,
    "grid_consensus": grid_consensus,
    "grid_field": grid_field,
    "grid_mesh": grid_mesh,
    "grid_holes": grid_holes,
    "grid_observer": grid_observer,
    "vertical_ridges": vertical_ridges,
    "ray_model": ray_model,
    "ray_sampling": ray_sampling,
    "converter": converter,
    "coordinate_mesh": coordinate_mesh,
    "coordinate_map": coordinate_map,
    "curve_domain": curve_domain,
}
