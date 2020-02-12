import sys
import numpy as np
from spglib import get_symmetry_dataset, standardize_cell
from ase import Atoms
from ase.spacegroup import Spacegroup
from ase.build import cut


from .wyckoff_symmetries import WyckoffSymmetries, wrap_coordinate

from protosearch import build_bulk
path = build_bulk.__path__[0]

wyckoff_data = path + '/Wyckoff.dat'
wyckoff_pairs = path + '/Symmetric_Wyckoff_Pairs.dat'


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

    def get_primitive_atoms(self, atoms):
        """Transform from primitive to conventional cell"""

        lattice, scaled_positions, numbers = standardize_cell(atoms,
                                                              to_primitive=True,
                                                              no_idealize=False,
                                                              symprec=1e-5)

        atoms = Atoms(numbers=numbers,
                      cell=lattice,
                      pbc=True)

        atoms.set_scaled_positions(scaled_positions)

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

        w_sorted = []
        # for s in set(self.species):
        #    indices = [i for i, s0 in enumerate(self.species) if s0==s]
        #    w_sorted += sorted([self.wyckoffs[i] for i in indices])
        #self.wyckoffs = w_sorted

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


def get_wyckoff_pair_symmetry_matrix(spacegroup):
    letters, multiplicity = get_wyckoff_letters_and_multiplicity(spacegroup)

    n_points = len(letters)
    pair_names = []
    for i in range(n_points):
        for j in range(n_points):
            pair_names.append('{}_{}'.format(letters[i], letters[j]))

    M = np.zeros([n_points**2, n_points**2])

    with open(wyckoff_pairs, 'r') as f:
        sg = 1
        for i, line in enumerate(f):
            if len(line) == 1:
                if sg < spacegroup:
                    sg += 1
                    continue
                else:
                    break
            if sg < spacegroup:
                continue
            w_1 = line[:3]
            if not w_1 in pair_names:
                continue
            k = pair_names.index(w_1)
            pairs0 = line.split('\t')[2: -1]
            for w_2 in pairs0:
                j = pair_names.index(w_2)
                M[k, j] = 1

    free_letters = []
    for l in letters:
        i = pair_names.index(l + '_' + l)
        if M[i, i] == 1:
            free_letters += [l]

    np.fill_diagonal(M, 1)

    return pair_names, M, free_letters


def get_wyckoff_letters_and_multiplicity(spacegroup):
    letters = np.array([], dtype=str)
    multiplicity = np.array([])
    with open(wyckoff_data, 'r') as f:
        i_sg = np.inf
        i_w = np.inf
        for i, line in enumerate(f):
            if '1 {} '.format(spacegroup) in line \
               and not i_sg < np.inf:
                i_sg = i
            if i > i_sg:
                if len(line) > 15:
                    continue
                if len(line) == 1:
                    break
                multi, w, sym, sym_multi = line.split(' ')
                letters = np.insert(letters, 0, w)
                multiplicity = np.insert(multiplicity, 0, multi)

    return letters, multiplicity
