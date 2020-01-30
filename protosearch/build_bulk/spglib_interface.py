import sys
import numpy as np
from spglib import get_symmetry_dataset, standardize_cell
from ase import Atoms


class SpglibInterface:
    """Prototype classification of atomic structure in ASE Atoms format"""

    def __init__(self, atoms, tolerance=1e-3):

        self.atoms = atoms

        self.spglibdata = get_symmetry_dataset((atoms.get_cell(),
                                                atoms.get_scaled_positions(),
                                                atoms.get_atomic_numbers()),
                                               symprec=tolerance)
        self.tolerance = tolerance
        self.spacegroup = self.spglibdata['number']

    def get_spacegroup(self):
        return self.spglibdata['number']

    def get_conventional_atoms(self):
        """Transform from primitive to conventional cell"""

        std_cell = self.spglibdata['std_lattice']
        positions = self.spglibdata['std_positions']
        numbers = self.spglibdata['std_types']

        atoms = Atoms(numbers=numbers,
                      cell=std_cell,
                      pbc=True)

        atoms.set_scaled_positions(positions)

        atoms.wrap()

        return atoms

    def get_primitive_atoms(self, atoms):
        """Transform to primitive cell"""

        lattice, scaled_positions, numbers = standardize_cell(self.atoms,
                                                              to_primitive=True,
                                                              no_idealize=True,
                                                              symprec=1e-5)

        atoms = Atoms(numbers=numbers,
                      cell=lattice,
                      pbc=True)

        atoms.set_scaled_positions(scaled_positions)

        atoms.wrap()

        return atoms
