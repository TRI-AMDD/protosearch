import sys
import numpy as np
from spglib import get_symmetry_dataset
from ase import Atoms
from ase.spacegroup import Spacegroup
from ase.build import cut


from .wyckoff_symmetries import WyckoffSymmetries, wrap_coordinate


class PrototypeClassification(WyckoffSymmetries):
    """Prototype classification of atomic structure in ASE Atoms format"""

    def __init__(self, atoms, tolerance=1e-3):

        self.spglibdata = get_symmetry_dataset((atoms.get_cell(),
                                                atoms.get_scaled_positions(),
                                                atoms.get_atomic_numbers()),
                                               symprec=tolerance)
        self.tolerance = tolerance

        self.spacegroup = self.spglibdata['number']
        self.Spacegroup = Spacegroup(self.spacegroup)
        self.atoms = self.get_conventional_atoms(atoms)

        super().__init__(spacegroup=self.spacegroup)

        self.set_wyckoff_species()
        WyckoffSymmetries.wyckoffs = self.wyckoffs
        WyckoffSymmetries.species = self.species

    def get_conventional_atoms(self, atoms):
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

    def set_wyckoff_species(self):
        self.wyckoffs = []
        self.species = []

        relative_positions = np.dot(np.linalg.inv(
            self.atoms.cell.T), self.atoms.positions.T).T

        relative_positions = np.round(relative_positions, 10)

        normalized_sites = \
            self.Spacegroup.symmetry_normalised_sites(
                np.array(relative_positions,
                         ndmin=2))

        equivalent_sites = []
        for i, site in enumerate(normalized_sites):
            indices = np.all(np.isclose(site, normalized_sites[:i+1],
                                        rtol=self.tolerance), axis=1)
            index = np.min(np.where(indices)[0])
            equivalent_sites += [index]

        unique_site_indices = list(set(equivalent_sites))

        unique_sites = normalized_sites[unique_site_indices]
        count_sites = [list(equivalent_sites).count(i)
                       for i in unique_site_indices]

        symbols = self.atoms[unique_site_indices].get_chemical_symbols()

        for i, position in enumerate(unique_sites):
            found = False
            for w in sorted(self.wyckoff_symmetries.keys()):
                m = self.wyckoff_multiplicities[w]
                if not count_sites[i] == m:
                    continue
                for w_sym in self.wyckoff_symmetries[w]:
                    if found:
                        break
                    if self.is_position_wyckoff(position, w_sym):
                        found = True
                        self.wyckoffs += [w]
                        self.species += [symbols[i]]
                        break
            if not found:
                print('Error: position: ', position, ' not identified')

        indices = np.argsort(self.species)

        #print(self.species, self.wyckoffs)

        free_wyckoffs = self.get_free_wyckoffs()
        self.atoms_wyckoffs = []
        self.free_atoms = []
        for site in equivalent_sites:
            index = unique_site_indices.index(site)
            w_position = self.wyckoffs[index]
            self.atoms_wyckoffs += [w_position + str(index)]
            self.free_atoms += [w_position in free_wyckoffs]

    def get_spacegroup(self):
        return self.spacegroup

    def get_wyckoff_species(self):
        return self.wyckoffs, self.species

    def get_classification(self, include_parameters=True):

        p_name = self.get_prototype_name(self.species)

        structure_name = str(self.spacegroup)
        for spec, wy_spec in zip(self.species, self.wyckoffs):
            structure_name += '_{}_{}'.format(spec, wy_spec)

        prototype = {'p_name': p_name,
                     'structure_name': structure_name,
                     'spacegroup': self.spacegroup,
                     'wyckoffs': self.wyckoffs,
                     'species': self.species}

        if include_parameters:
            cell_parameters = self.get_cell_parameters(self.atoms)

            return prototype, cell_parameters

        else:
            return prototype
