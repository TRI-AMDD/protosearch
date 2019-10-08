import string
import numpy as np

from ase.geometry import cell_to_cellpar

from protosearch import build_bulk

path = build_bulk.__path__[0]
wyckoff_data = path + '/Wyckoff.dat'

crystal_class_coordinates = {'P': [[0, 0, 0]],
                             'A': [[0, 0, 0],
                                   [0, 0.5, 0.5]],
                             'C': [[0, 0, 0],
                                   [0.5, 0.5, 0]],
                             'I': [[0, 0, 0],
                                   [0.5, 0.5, 0.5]],
                             'R': [[0, 0, 0]],
                             'R-3': [[0, 0, 0],
                                     [2/3, 1/3, 1/3],
                                     [1/3, 2/3, 2/3]],
                             'F': [[0, 0, 0],
                                   [0, 0.5, 0.5],
                                   [0.5, 0, 0.5],
                                   [0.5, 0.5, 0]],
                             'H': [[0, 0, 0],
                                   [2/3, 1/3, 1/3],
                                   [1/3, 2/3, 2/3]]}


class WyckoffSymmetries:
    """Class to handle Wyckoff positions and symmetries"""

    def __init__(self,
                 spacegroup,
                 wyckoffs=None,
                 ):
        self.spacegroup = spacegroup

        self.set_wyckoff_symmetries()
        if wyckoffs is not None:
            self.wyckoffs = wyckoffs
            self.set_wyckoff_mapping()

    def set_wyckoff_symmetries(self):
        """Construct dictionary with  """
        alphabet = list(string.ascii_lowercase)

        self.wyckoff_symmetries = {}
        self.wyckoff_site_symmetries = {}
        self.wyckoff_multiplicities = {}
        position = None
        with open(wyckoff_data, 'r') as f:
            i_sg = np.inf
            i_w = np.inf
            for i, line in enumerate(f):
                if '1 {} '.format(self.spacegroup) in line \
                   and not i_sg < np.inf:
                    i_sg = i
                    SG_letter = line.split(' ')[2]
                    if 'R-3' in SG_letter:
                        SG_letter = 'R-3'
                    else:
                        SG_letter = SG_letter[0]
                    self.class_coordinates = crystal_class_coordinates[SG_letter]

                if i > i_sg:
                    if len(line) == 1:
                        break
                    if line.split(' ')[1] in alphabet:
                        i_w = i
                        position, site_symmetry = line.split(' ')[1:3]
                        self.wyckoff_multiplicities.update(
                            {position: int(line.split(' ')[0])})
                        self.wyckoff_site_symmetries.update(
                            {position: site_symmetry})
                        wyckoff_array = np.zeros([0, 3, 4])
                    if i > i_w:
                        if len(line) < 40:
                            break
                        arr = np.array(line.split(' ')[:-1], dtype=float)
                        for cc in self.class_coordinates:
                            x = arr[:4].copy()
                            x[3] += cc[0]
                            y = arr[4:8].copy()
                            y[3] += cc[1]
                            z = arr[8:].copy()
                            z[3] += cc[2]
                            new_arr = np.expand_dims(
                                np.array([x, y, z]), axis=0)

                            wyckoff_array = np.append(
                                wyckoff_array, new_arr, axis=0)
                if position:
                    self.wyckoff_symmetries.update({position: wyckoff_array})

    def set_wyckoff_mapping(self):
        # Name of coordinate variables (Enumerator consistent)
        self.coor_variables = []
        # Map atomic indices to those of same wyckoff position
        self.symmetry_map = []
        # Symmetry matrix and offset for each atom
        self.wyckoff_map = []
        # Atom index free to move or now
        self.free_atoms = []
        # Wyckoff position for atom indices
        self.atoms_wyckoffs = []
        # free Wyckoff directions
        self.w_free_dir = {}

        xyz = list('xyz')
        relative_positions = []

        symbols = []
        atoms_index = 0

        for i_w, w in enumerate(self.wyckoffs):
            w_id = w + str(i_w)
            m = self.wyckoff_multiplicities[w]
            w_arrays = self.wyckoff_symmetries[w]
            self.w_free_dir.update({w_id: []})

            for w_array in w_arrays:
                self.wyckoff_map += [w_array]
                M = w_array[:, :3]

                free_wyckoff = False
                for d in range(3):
                    if not np.all(M[:, d] == 0):
                        v = xyz[d] + w_id
                        if not v in self.coor_variables:
                            self.coor_variables += [v]
                        self.w_free_dir[w_id] += [d]
                        free_wyckoff = True
                self.free_atoms += [free_wyckoff]
                self.atoms_wyckoffs += [w_id]

            sym_indices = [atoms_index + i for i in range(m)]
            for i in sym_indices:
                self.symmetry_map += [sym_indices]
            atoms_index += m

    def get_prototype_name(self, species):
        alphabet = list(string.ascii_uppercase)
        unique_symbols = []
        symbol_count = []
        for i, s in enumerate(species):
            if s in unique_symbols:
                index = unique_symbols.index(s)
                symbol_count[index] += self.wyckoff_multiplicities[self.wyckoffs[i]]
            else:
                symbol_count += [self.wyckoff_multiplicities[self.wyckoffs[i]]]
                unique_symbols += [s]

        min_rep = min(symbol_count)

        for n in list(range(1, min_rep + 1))[::-1]:
            if np.all(np.array(symbol_count) % n == 0):
                repetition = n
                break

        p_name = ''
        for i in np.argsort(symbol_count):
            p_name += alphabet[i]
            factor = symbol_count[i] // n
            if factor > 1:
                p_name += str(factor)

        p_name += '_' + str(repetition)

        added_species = ['']
        for i, w in enumerate(self.wyckoffs):
            s = species[i]
            if not s == added_species[-1]:
                p_name += '_'
                p_name += w
            else:
                if w == p_name[-1]:
                    p_name += '2'
                elif p_name[-1].isdigit() and w == p_name[-2]:
                    p_name = p_name[:-1] + str(int(p_name[-1]) + 1)
                else:
                    p_name += w

            added_species += [s]
        p_name += '_' + str(self.spacegroup)

        return p_name

    def get_wyckoff_transform_vector(self, vector, index_a, index_b, cell):
        vector = np.dot(np.linalg.inv(cell.T),
                        vector)

        vector0 = vector.copy()
        normv0 = np.linalg.norm(vector0)
        M_a = self.wyckoff_map[index_a][:3, :][:, :3]
        M_b = self.wyckoff_map[index_b][:3, :][:, :3]
        red_dim_x = 0
        red_dim_y = 0
        for i in range(3):
            i_x = i - red_dim_x
            i_y = i - red_dim_y
            if np.all(M_a[i_x, :] == 0):
                M_a = np.delete(M_a, i_x, 0)
                vector = np.delete(vector, i_x, 0)
                red_dim_x += 1
            if np.all(M_a[:, i_y] == 0):
                M_a = np.delete(M_a, i_y, 1)
                M_b = np.delete(M_b, i_y, 1)
                red_dim_y += 1

        M_a_inv = np.dot(np.linalg.inv(np.dot(M_a.T, M_a)), M_a.T)
        transform_vector = np.dot(M_b, np.dot(M_a_inv, vector))

        if not list(transform_vector):
            transform_vector = np.zeros(3)

        transform_vector = np.dot(cell.T, transform_vector)

        return transform_vector

    def get_inverse_wyckoff_matrix(self, M):
        x = []
        y = []
        for i in range(3):
            if not np.all(M[i, :] == 0):
                x += [i]
            if not np.all(M[:, i] == 0):
                y += [i]
        M = M[x, :][:, y]

        M_inv = np.dot(np.linalg.inv(np.dot(M.T, M)), M.T)

        return M_inv, x, y

    def get_cell_parameters(self, atoms):
        cell = atoms.cell
        cell_parameters = {}

        # self.get_free_wyckoff_parameters()
        parameters = cell_to_cellpar(cell)

        for i, param in enumerate(['a', 'b', 'c',
                                   'alpha', 'beta', 'gamma']):

            cell_parameters.update({param: parameters[i]})

        if np.all(self.free_atoms == 0):
            return cell_parameters

        dir_map = ['x', 'y', 'z']
        relative_positions = np.dot(np.linalg.inv(
            cell.T), atoms.positions.T).T

        for i, aw in enumerate(self.atoms_wyckoffs):
            if aw == self.atoms_wyckoffs[i-1]:
                continue

            for d in self.w_free_dir[aw]:
                w_p_name = dir_map[d] + aw
                if w_p_name in cell_parameters:
                    continue
                position = relative_positions[i]
                position = self.get_parameter_from_position(position, aw[0])
                cell_parameters.update({w_p_name: position[d]})
        return cell_parameters

    def get_parameter_from_position(self, position, w_position):

        for w_sym in self.wyckoff_symmetries[w_position]:
            M = w_sym[:, :3]
            c = w_sym[:, 3]

            M_inv, dim_x, dim_y = self.get_inverse_wyckoff_matrix(M)
            r_vec = (position - c)[dim_x]

            w00 = np.dot(r_vec, M_inv.T)
            w0 = np.zeros([3])
            w0[dim_y] = w00

            rp0 = np.dot(w0, M.T) + c

            if np.all(np.isclose(position, rp0)):
                return w0
