from protosearch.build_bulk.build_bulk import BuildBulk
from ase.visualize import view

BB = BuildBulk(47, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], ['Cu', 'Pd', 'Pt','Au', 'Cu', 'Pd', 'Pt','Au'])

atoms = BB.get_atoms(proximity=1)

view(atoms)
