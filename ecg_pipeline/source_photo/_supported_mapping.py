"""Compose unchanged maintained cell/mesh/hole mapping on fixed source paths."""
from ecg_pipeline.source_photo import _converter as converter,_grid_mesh as mesh,_coordinate_map as coordinate,_waveforms as waveforms

def mapper(field, geometry):
 if not field['cells']:return converter.CellMap
 base=mesh.mapper_class(converter.CellMap,geometry['mesh'])
 return coordinate.mapper_class(base,geometry['mesh'],geometry['extension'],mesh.triangle_value)

def convert(paths,field,geometry,offset,ranges,gain):
 factory=mapper(field,geometry)
 return [waveforms.clean(converter.convert(path,field['cells'],offset,ranges,gain=gain,mapper_factory=factory)) for path in paths]
