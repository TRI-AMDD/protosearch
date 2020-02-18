"""Pymatgen coordination environment"""

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.chemenv.coordination_environments.\
    coordination_geometry_finder import LocalGeometryFinder


def get_neighbors_set(atoms):
    LocalGeometryFinder = LocalGeometryFinder()
    strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters()
    
    structure = AseAtomsAdaptor.get_structure(atoms)
    
    LocalGeometryFinder.setup_structure(structure=structure)

    StructureEnv = LocalGeometryFinder.compute_structure_environments(
        maximum_distance_factor=1.41,
        only_cations=False)
    
    LightStructureEnv = LightStructureEnvironments.from_structure_environments(
        strategy=strategy, structure_environments=StructureEnv)

    neighbors_set = LightStructureEnv.as_dict()['neighbors_sets']

    return neighbors_set
