import functools, types
from . import _supported_cells as direct, _supported_domains as domains

def propose(c, brackets):
    domains.original=types.SimpleNamespace(build_field=functools.partial(direct.build_field,brackets=brackets,offset=c['sourceInterval'][0]))
    return domains.propose(c['nodes'],c['observations'],c['majorIndices'],c['gapPassed'],node_support=c['nodeSupport'])
