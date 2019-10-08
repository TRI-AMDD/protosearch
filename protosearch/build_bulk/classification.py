import numpy as np
from ase.spacegroup import get_spacegroup

from .wyckoff_symmetries import WyckoffSymmetries


class PrototypeClassification(WyckoffSymmetries):
    """Prototype classification of atomic structure in ASE Atoms formar"""

    def __init__(self, atoms):
        self.atoms = atoms
        self.Spacegroup = get_spacegroup(self.atoms)
        self.spacegroup = self.Spacegroup.no

        super().__init__(spacegroup=self.spacegroup)

        self.set_wyckoff_species()
        self.set_wyckoff_mapping()

    def set_wyckoff_species(self):
        self.wyckoffs = []
        self.species = []

        relative_positions = np.dot(np.linalg.inv(
            self.atoms.cell.T), self.atoms.positions.T).T

        relative_positions = np.round(relative_positions, 10)

        unique_sites = self.Spacegroup.unique_sites(relative_positions)

        sites, kinds = self.Spacegroup.equivalent_sites(unique_sites,
                                                        onduplicates='replace')
        count_sites = []
        for i in range(len(unique_sites)):
            count_sites += [kinds.count(i)]

        symbols = [a.symbol for i, a in enumerate(self.atoms)
                   if np.any(np.all(relative_positions[i] == unique_sites, axis=1))]

        taken_sites = []

        for w in sorted(self.wyckoff_symmetries.keys()):
            m = self.wyckoff_multiplicities[w]
            for w_sym in self.wyckoff_symmetries[w]:
                M = w_sym[:, :3]
                c = w_sym[:, 3]

                M_inv, dim_x, dim_y = self.get_inverse_wyckoff_matrix(M)
                for i, rp in enumerate(unique_sites):
                    if i in taken_sites:
                        continue
                    if not count_sites[i] == m:
                        continue

                    r_vec = (rp - c)[dim_x]

                    w00 = np.dot(r_vec, M_inv.T)

                    w0 = np.zeros([3])
                    w0[dim_y] = w00

                    rp0 = np.dot(w0, M.T) + c
                    rp0 = [r-1 if r >= 1 else r+1 if r < 0 else r for r in rp0]
                    if np.all(np.isclose(rp, rp0)):
                        self.wyckoffs += [w]
                        self.species += [symbols[i]]
                        taken_sites += [i]
                        break

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
