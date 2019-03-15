from ase.visualize import view
from protosearch.build_bulk.cell_parameters import CellParameters


CP = CellParameters(spacegroup=1,
                    wyckoffs=['a', 'a', 'a', 'a'],
                    species=['Fe', 'O', 'O', 'O'])


parameters = CP.get_parameter_estimate()

print(parameters)

atoms = CP.get_atoms(parameters)
view(atoms)
