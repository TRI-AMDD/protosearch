import sys
import io
import time
import numpy as np
from numpy.random import rand
import ase
from ase import Atoms, Atom
from ase import visualize
from ase.geometry import get_distances, cell_to_cellpar, cellpar_to_cell
from ase.data import atomic_numbers as a_n
from ase.data import covalent_radii as cradii
from ase.io.vasp import read_vasp, write_vasp
from ase.geometry.geometry import wrap_positions
import bulk_enumerator as be

from protosearch.build_bulk.classification import get_classification
from protosearch import build_bulk

path = build_bulk.__path__
wyckoff_data = path[0] + '/Wyckoff.dat'

crystal_class_coordinates = {'P': [[0, 0, 0]],
                             'A': [[0, 0, 0],
                                   [0, 0.5, 0.5]],
                             'C': [[0, 0, 0],
                                   [0.5, 0.5, 0]],
                             'I': [[0, 0, 0],
                                   [0.5, 0.5, 0.5]],
                             'R': [[0, 0, 0]],
                             'F': [[0, 0, 0],
                                   [0, 0.5, 0.5],
                                   [0.5, 0, 0.5],
                                   [0.5, 0.5, 0]],
                             'H': [[0, 0, 0],
                                   [2/3, 1/3, 1/3],
                                   [1/3, 2/3, 2/3]]}


class CellParameters:
    """
    Provides a fair estimate of cell parameters including lattice constants,
    angles, and free wyckoff coordinates.

    Parameters:

    spacegroup: int
       int between 1 and 230
    wyckoffs: list
       wyckoff positions, for example ['a', 'a', 'b', 'c']
    species: list
       atomic species, for example ['Fe', 'O', 'O', 'O']
    """

    def __init__(self,
                 spacegroup,
                 wyckoffs,
                 species,
                 verbose=True,
                 wyckoffs_from_oqmd=False,
                 ):
        self.spacegroup = spacegroup
        self.wyckoffs = wyckoffs
        self.species = species
        self.verbose = verbose
        # WIP
        self.wyckoffs_from_oqmd = wyckoffs_from_oqmd

        b = be.bulk.BULK()
        b.set_spacegroup(self.spacegroup)
        b.set_wyckoff(self.wyckoffs)
        b.set_species(self.species)
        self.parameters = b.get_parameters()
        b.delete()

        self.coor_parameters = []
        self.angle_parameters = []
        self.lattice_parameters = []
        for p in self.parameters:
            if np.any([coor in p for coor in ['x', 'y', 'z']]):
                self.coor_parameters += [p]
            elif p in ['alpha', 'beta', 'gamma']:
                self.angle_parameters += [p]
            elif p in ['a', 'b/a', 'c/a']:
                self.lattice_parameters += [p]
        self.set_lattice_dof()
        self.parameter_guess = self.initial_guess()

    def get_parameter_estimate(self, master_parameters=None):
        """
        Optimize lattice parameters for Atoms object generated with the bulk
        Enumerator.
        First wyckoff coordinates are optimized, then the angles, and at last
        the lattice constant.

        Parameters:
        master_parameters: dict
           fixed cell parameters and values that will not be optimized.
        """

        cell_parameters = self.parameter_guess
        master_parameters = master_parameters or {}

        if master_parameters:
            master_parameters = clean_parameter_input(master_parameters)
            cell_parameters.update(master_parameters)
        atoms = self.get_atoms(fix_parameters=cell_parameters)
        optimize_wyckoffs = False

        if not np.all([c in master_parameters
                       for c in self.coor_parameters]):
            optimize_wyckoffs = True
            coor_guess = self.get_wyckoff_coordinates()
            cell_parameters.update(coor_guess)
        # Angle optimization is too slow. Just use initial guess for now
        """
        if self.angle_parameters:
            if not np.all([c in master_parameters for c in self.angle_parameters]):
                angle_guess = self.get_angles(cell_parameters)
                cell_parameters.update(angle_guess)
        """
        if not np.all([c in master_parameters for c in self.lattice_parameters]):
            if self.verbose:
                print('Optimizing lattice constants for {} - {} - {}'
                      .format(self.spacegroup, self.wyckoffs, self.species))

            cell_parameters, covalent_density = \
                self.get_lattice_constants(cell_parameters,
                                           optimize_wyckoffs)
            if not cell_parameters:
                return None

            if covalent_density < 0.05:
                print('Warning: very low density. Omitting this structure')
                return None

        cell_parameters.update(master_parameters)
        atoms = self.get_atoms(fix_parameters=cell_parameters)
        if not atoms:
            return None
        elif self.check_prototype(atoms):
            return cell_parameters
        else:
            if self.verbose:
                print("Structure reduced to another spacegroup")
            return None

    def set_lattice_dof(self):
        """Set degrees of freedom for lattice constants and angles
        based on spacegroup
        """
        sg = self.spacegroup
        if sg in [1, 2]:  # Triclinic
            relax = [0, 1, 2]
            fixed_angles = {}

        elif 3 <= sg <= 15:  # Monoclinic
            relax = [0, 1, 2]
            fixed_angles = {'alpha': 90, 'gamma': 90}

        elif 16 <= sg <= 74:  # Orthorombic
            relax = [0, 1, 2]
            fixed_angles = {'alpha': 90, 'beta': 90,
                            'gamma': 90}

        elif 75 <= sg <= 142:  # Tetragonal
            relax = [[0, 1], 2]  # a = b
            fixed_angles = {'alpha': 90, 'beta': 90,
                            'gamma': 90}

        elif 143 <= sg <= 167:  # Trigonal
            # Hexagonal ?
            relax = [[0, 1], 2]  # a = b
            fixed_angles = {'alpha': 90, 'beta': 90,
                            'gamma': 120}

        elif 168 <= sg <= 194:  # Hexagonal
            relax = [[0, 1], 2]  # a = b
            fixed_angles = {'alpha': 90, 'beta': 90,
                            'gamma': 120}

        elif 195 <= sg <= 230:  # cubic
            relax = [[0, 1, 2]]  # a = b = c
            fixed_angles = {'alpha': 90, 'beta': 90,
                            'gamma': 90}

        self.d_o_f = relax
        self.fixed_angles = fixed_angles

    def get_atoms(self, fix_parameters=None, primitive=False, species=None):
        """
        Get ASE atoms object generated with the Enumerator
        with parameters specified in `fix_parameters`. If all parameters
        are not provided, a very rough estimate will be applied.
        """

        species = species or self.species
        if fix_parameters:
            self.parameter_guess.update(fix_parameters)

        parameter_guess_values = []
        for p in self.parameters:
            parameter_guess_values += [self.parameter_guess[p]]

        b = be.bulk.BULK()
        b.set_spacegroup(self.spacegroup)
        b.set_wyckoff(self.wyckoffs)
        b.set_species(species)
        b.set_parameter_values(self.parameters, parameter_guess_values)

        if primitive:
            poscar = b.get_primitive_poscar()
        else:
            poscar = b.get_std_poscar()
        b.delete()

        if poscar == "":
            mess = ("Enumerator failed to create poscar!!!!" + "\n"
                    "This indicates that the symmetry of the crystal is reduced \n"
                    + "Please check the Wyckoff coordinates"
                    )
            print(mess)
            self.atoms = None
        else:
            self.atoms = read_vasp(io.StringIO(poscar))

        return self.atoms

    def check_prototype(self, atoms):
        """Check that spacegroup and wyckoff positions did not change"""

        b2 = be.bulk.BULK()

        poscar = io.StringIO()
        write_vasp(poscar, atoms=atoms, vasp5=True,
                   long_format=False, direct=True)

        b2.set_structure_from_file(poscar.getvalue())
        sg2 = b2.get_spacegroup()
        s2 = b2.get_species()
        w2 = b2.get_wyckoff()
        b2.delete()

        wyckoff_species = sorted([self.wyckoffs[i] + self.species[i]
                                  for i in range(len(self.species))])

        wyckoff_species2 = sorted([w2[i] + s2[i] for i in range(len(s2))])

        if not sg2 == self.spacegroup:
            if self.verbose:
                print('Symmetry reduced from spacegroup {} to {}'.format(
                    self.spacegroup, sg2))
            return False

        # This check doesn't work since the wyckoffs can change
        # due to a flip of the axis (if c < a for example without  changing the symmetry).
        # if not wyckoff_species2 == wyckoff_species:
        #    if self.verbose:
        #        print('Wyckoffs changed from {} to {}'.format(
        #            self.wyckoffs, w2))
        #    return False

        return True

    def initial_guess(self):
        """ Rough estimate of parameters """

        parameter_guess = {}

        covalent_radii = [cradii[a_n[s]] for s in self.species]
        mean_radii = np.mean(covalent_radii)

        lattice_parameters = \
            [lat_par for lat_par in ['a', 'b', 'c'] if lat_par in self.parameters]

        for lat_par in lattice_parameters:
            parameter_guess.update({lat_par: mean_radii * 4})

        if 'b/a' in self.parameters:
            parameter_guess.update({'b/a': 1.1})

        if 'c/a' in self.parameters:
            parameter_guess.update({'c/a': 1.4})

        if 'alpha' in self.parameters:
            parameter_guess.update({'alpha': 84})

        if 'beta' in self.parameters:
            parameter_guess.update({'beta': 96})

        if 'gamma' in self.parameters:
            parameter_guess.update({'gamma': 84})

        for i, p in enumerate(self.coor_parameters):
            parameter_guess.update({p: rand(1)[0] * 0.99})

        self.parameter_guess = parameter_guess

        atoms = self.get_atoms()

        natoms = atoms.get_number_of_atoms()

        parameter_guess.update({'a': mean_radii * 4 * natoms ** (1 / 3)})

        return parameter_guess

    def get_wyckoff_coordinates(self, view_images=False):
        # Determine high-symmetry positions taken in the unit cell.
        atoms = self.get_atoms()
        relative_positions = np.dot(np.linalg.inv(
            atoms.cell.T), atoms.positions.T).T
        high_sym_idx = []
        taken_positions = []
        for i, p in enumerate(relative_positions):
            high_sym = []
            for px in p:
                high_sym += np.any([np.isclose(px, value) for
                                    value in [0, 1/4, 1/3, 1/2, 2/3, 3/4, 1]])
            if np.all(high_sym):
                taken_positions += [p]

        dir_map = {'x': 0,
                   'y': 1,
                   'z': 2}
        natoms = atoms.get_number_of_atoms()

        w_free_param = {}
        parameters_axis = {'x': [], 'y': [], 'z': []}

        for iw, w in enumerate(self.wyckoffs):
            w_enum = w + str(iw)
            w_free_param.update({w_enum: []})
            for c in [c for c in self.coor_parameters
                      if w_enum == c[1:]]:
                direction = c.replace(w_enum, '')
                xyz = dir_map[direction]
                if not xyz in w_free_param[w_enum]:
                    w_free_param[w_enum] += [dir_map[direction]]
                    parameters_axis[direction] += [c]
        self.w_free_param = w_free_param
        for direction, parameters in parameters_axis.items():
            n_points = len(parameters)
            if n_points == 0:
                continue

            # Always stay away from these positions that often acts as
            # mirror planes for wyckoff coordinates
            high_sym_pos = [0, 0.5, 1]
            if taken_positions:
                high_sym_pos0 = sorted([t[dir_map[direction]]
                                        for t in taken_positions[:]])
            else:
                high_sym_pos0 = []

            # Add other existing high symmetry points
            high_sym_pos += [h for h in high_sym_pos0 if np.any(
                [np.isclose(h, value) for value in [1/4, 1/3, 2/3, 3/4]])]
            high_sym_pos = sorted(high_sym_pos)

            # map distance between points
            dist = np.array(high_sym_pos[1:]) - \
                np.array(high_sym_pos[:-1])

            n_points_arr = np.round(n_points * dist)

            diff = sum(n_points_arr) - n_points
            if diff > 0:
                n_points_arr[np.argmax(n_points_arr)] -= diff
            if diff < 0:
                n_points_arr[np.argmin(n_points_arr)] -= diff

            variables = []
            for i in range(len(high_sym_pos) - 1):
                variables += \
                    list(np.linspace(high_sym_pos[i],
                                     high_sym_pos[i + 1],
                                     int(n_points_arr[i]) + 2)[1: -1])
            # Small random shift
            variables += (rand(len(variables)) - 0.5) / \
                100 * atoms.get_number_of_atoms()

            # Shift slightly away from other high symmetry positions
            for i, v in enumerate(variables):
                if np.any([np.isclose(v, v0) for v0 in [1/4, 1/3, 2/3, 3/4]]):
                    variables[i] += 0.03

            # Shuffle variables to avoid positions only along the diagonal
            variables_shuffle = []
            n_splits = dir_map[direction] + 1 + len(variables) // 10
            for cut in range(n_splits):
                idx = list(range(cut, len(variables),
                                 n_splits))
                variables_shuffle += list(np.array(variables)[idx])
            for i, v in enumerate(variables_shuffle):
                self.parameter_guess.update({parameters[i]: v})
        return self.parameter_guess

    def get_sorted_atoms(self, fix_parameters=None):
        """Sort atoms accordingly to wyckoff positions as needed for wyckoff 
        optimization, and map out wyckoff symmetry constraints
        """

        self.free_wyckoff_idx = []
        self.fixed_wyckoff_idx = []

        for i, w in enumerate(self.wyckoffs):
            if np.any([c[1:] == w + str(i) for c in self.coor_parameters]):
                self.free_wyckoff_idx += [i]
            else:
                self.fixed_wyckoff_idx += [i]

        # Replace species by dummy atoms
        set_species = list(set(self.species))
        species = []
        noble = ['He', 'Ar', 'Ne', 'Kr', 'Xe']
        for s in self.species:
            i = set_species.index(s)
            species += [noble[i]]

        atoms = self.get_atoms(fix_parameters=fix_parameters, species=species)
        relative_positions = np.dot(np.linalg.inv(
            atoms.cell.T), atoms.positions.T).T

        sorted_atoms = ase.Atoms(cell=atoms.cell, pbc=True)
        multiplicity = []
        atoms_wyckoffs = []
        self.symmetry_map = {}
        self.wyckoff_map = {}
        symmetry_map = {}
        position_symmetries = []
        self.free_atoms = []
        for i_w, w in enumerate(self.wyckoffs):
            species[i_w] = self.species[i_w]
            atoms_test = self.get_atoms(fix_parameters=fix_parameters,
                                        species=species)
            wyckoff_coordinate = []
            for direction in ['x', 'y', 'z']:
                wyckoff_coordinate += [
                    fix_parameters.get(direction + w + str(i_w), 0)]

            count_added = 0
            for atom_test in atoms_test:
                for i, atom in enumerate(atoms):
                    if np.all(atom_test.position == atom.position) \
                       and not atom_test.symbol == atom.symbol:
                        count_added += 1
                        sorted_atoms += atom_test
                        atoms[i].symbol = atom_test.symbol
                        atoms_wyckoffs += [w + str(i_w)]
                        if i_w in self.fixed_wyckoff_idx:
                            self.free_atoms += [0]
                        else:
                            self.free_atoms += [1]
                        if count_added == 1:
                            continue
            multiplicity += [count_added] * count_added
            try:
                self.assign_wyckoff_symmetries(w, sorted_atoms,
                                               count_added)
            except:
                return None

        self.atoms_wyckoffs = atoms_wyckoffs
        self.multiplicity = multiplicity
        return sorted_atoms

    def get_wyckoff_symmetries(self, wyckoff_position):

        wyckoff_array = np.zeros([0, 3, 4])
        with open(wyckoff_data, 'r') as f:
            i_sg = np.inf
            i_w = np.inf
            for i, line in enumerate(f):
                if '1 {} '.format(self.spacegroup) in line \
                   and not i_sg < np.inf:
                    i_sg = i
                    SG_letter = line.split(' ')[2][0]
                    class_coordinates = crystal_class_coordinates[SG_letter]

                if i > i_sg:
                    if len(line) == 1:
                        break
                    if line.split(' ')[1] == wyckoff_position:
                        i_w = i

                    if i > i_w:
                        if len(line) < 40:
                            break

                        arr = np.array(line.split(' ')[:-1], dtype=float)
                        for cc in class_coordinates:
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

        return wyckoff_array

    def assign_wyckoff_symmetries(self, wyckoff_position, atoms, index):

        count_atoms0 = len(atoms) - index

        relative_positions = np.dot(np.linalg.inv(atoms.cell.T),
                                    atoms[-index:].positions.T).T

        wyckoff_array = self.get_wyckoff_symmetries(wyckoff_position)

        # Determine principal wyckoff position
        w_sym0 = wyckoff_array[0]
        M = w_sym0[:, :3]
        c = w_sym0[:, 3]
        coordinate = None
        for rp in relative_positions:
            rp0 = np.dot(rp - c, M.T) + c
            if np.all(np.isclose(rp, rp0)):
                coordinate = rp
                break

        atoms_test = Atoms(cell=atoms.cell, pbc=True)
        for w_sym in wyckoff_array:
            # r_i = r_0 * M^T + c_i
            # M: 3x3 matrix, coordinate mapping
            # c + constant vector term in relative coordinate
            M = w_sym[:, :3]
            c = w_sym[:, 3]
            w_rel_pos = np.dot(coordinate, M.T) + c
            w_abs_pos = np.dot(atoms.cell.T, w_rel_pos)
            atoms_test += Atom('O', position=w_abs_pos)

        atoms.wrap()
        atoms_test.wrap()

        Dm, distances = get_distances(atoms[-index:].positions,
                                      atoms_test.positions,
                                      cell=atoms.cell, pbc=True)

        d = np.nonzero(np.isclose(distances, 0, atol=0.001))

        d_map = d[1][d[0]]
        sym_indices = [count_atoms0 + i for i in range(index)]

        for i in sym_indices:
            self.symmetry_map.update({str(i): sym_indices})
            self.wyckoff_map.update(
                {str(i): wyckoff_array[d_map[i - count_atoms0]]})

    def get_wyckoff_transform_vector(self, vector, index_a, index_b, cell):

        vector = np.dot(np.linalg.inv(cell.T),
                        vector)

        vector0 = vector.copy()
        normv0 = np.linalg.norm(vector0)
        M_a = self.wyckoff_map[str(index_a)][:3, :][:, :3]
        M_b = self.wyckoff_map[str(index_b)][:3, :][:, :3]
        dim_y = list(range(3))
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

    def get_angles(self, fix_parameters={}):
        """
        Get an estimate for unit cell angles. The angles are optimized by
        minimizing the volume of the unit cell.
        ** Work in progess

        """
        fix_parameters.update(self.fixed_angles)

        temp_parameters = fix_parameters.copy()
        temp_parameters = self.get_lattice_constants(
            temp_parameters)

        atoms = self.get_atoms(temp_parameters)
        step_size = 1
        Volume0 = atoms.get_volume()
        volume0 = atoms.get_volume()
        direction = -1
        diff = 1
        j = 1
        Diff = 1
        while abs(Diff) > 0.05:  # Outer convergence criteria
            if self.verbose:
                print('Angle iteration {}'.format(j))

            for angle in self.angle_parameters:
                direction = -1
                direction_turns = 0
                step_size = 1
                angle0 = self.parameter_guess[angle]
                diff = 1
                gradient = 1
                i = 0
                while abs(diff) >= 0.05 and direction_turns < 2:
                    i += 1
                    delta_angle = step_size * gradient
                    angletest = angle0 + step_size * gradient
                    temp_parameters = fix_parameters.copy()
                    temp_parameters.update({angle: angletest})
                    temp_parameters = self.get_lattice_constants(
                        temp_parameters)

                    cell_params = []
                    for param in ['a', 'b/a', 'c/a', 'alpha', 'beta', 'gamma']:
                        cell_params += [temp_parameters[param]]

                    cell = cellpar_to_cell(cell_params)
                    atoms.set_cell(cell, scale_atoms=True)

                    volume = atoms.get_volume()
                    gradient = (volume - volume0) / delta_angle
                    diff = (volume - volume0) / volume0

                    if diff < 0:
                        angle0 = angletest
                        self.parameter_guess.update({angle: angle0})
                        fix_parameters.update({angle: angle0})
                        volume0 = volume
                    else:
                        direction_turns += 1
                        direction *= -1

                    step_size *= 0.5
            Diff = abs(volume0 - Volume0) / Volume0
            Volume0 = volume0
            j += 1

        return self.parameter_guess

    def get_lattice_constants(self, fix_parameters={}, optimize_wyckoffs=False,
                              proximity=1, view_images=False):
        """
        Get lattice constants by reducing the cell size (one direction at
        the time) until atomic distances on the closest pair reaches the
        sum of the covalent radii.

        if the optimize_wyckoffs parameter is set, the wyckoff coordinates 
        are optimized together with the lattice constants. 
        """

        if not fix_parameters:
            fix_parameters = self.parameter_guess
        if optimize_wyckoffs:
            atoms = self.get_sorted_atoms(fix_parameters)
        else:
            atoms = self.get_atoms(fix_parameters)
        if not atoms:
            return None, None

        cell = atoms.cell

        Dm, distances = get_interatomic_distances(atoms)
        covalent_radii = np.array([cradii[n] for n in atoms.numbers])
        M = covalent_radii * np.ones([len(atoms), len(atoms)])
        self.min_distances = (M + M.T) * proximity

        # scale up or down
        images = [atoms.copy()]
        soft_limit = 1.36
        scale = np.min(distances / self.min_distances / soft_limit)
        atoms.set_cell(atoms.cell * 1 / scale, scale_atoms=True)
        Dm, distances = get_interatomic_distances(atoms)
        images += [atoms.copy()]
        # Volume taken up by atoms
        covalent_volume = np.sum(4/3 * np.pi * covalent_radii ** 3)

        self.hard_limit = soft_limit
        increment = 0.99
        t0 = time.time()
        t = 0
        while self.hard_limit > 1:
            cell_norm = np.linalg.norm(cell, axis=1)
            axis_length_ix = np.argsort(
                np.mean(cell_norm[d]) for d in self.d_o_f)
            if optimize_wyckoffs:
                images += self.run_wyckoff_optimization_loop(atoms)
                atoms = images[-1].copy()
                Dm, distances = get_interatomic_distances(atoms)
                cell_volume = atoms.get_volume()

            for direction in self.d_o_f:
                self.hard_limit *= increment
                while np.all(distances > self.min_distances * self.hard_limit):
                    cell = atoms.cell.copy()
                    cell[direction, :] *= increment
                    atoms.set_cell(cell, scale_atoms=True)
                    Dm, distances = get_interatomic_distances(atoms)
                    cell_volume = atoms.get_volume()
                    t = time.time() - t0
                images += [atoms.copy()]
            t = time.time() - t0

        if np.any(distances < self.min_distances):
            scale = np.min(distances / self.min_distances)
            atoms.set_cell(atoms.cell * 1 / scale, scale_atoms=True)
            images += [atoms.copy()]

        # Volume taken up by atoms
        covalent_volume = np.sum(4/3 * np.pi * covalent_radii ** 3)
        cell_volume = atoms.get_volume()
        density = covalent_volume / cell_volume

        if view_images:
            ase.visualize.view(images)

        new_parameters = self.read_cell_parameters(atoms)
        fix_parameters.update(new_parameters)
        return fix_parameters, density

    def read_cell_parameters(self, atoms):

        new_parameters = {}
        parameters = cell_to_cellpar(atoms.cell)
        parameters[1:3] /= parameters[0]

        for i, param in enumerate(['a', 'b/a', 'c/a',
                                   'alpha', 'beta', 'gamma']):

            if param == 'c/a' and not 'c/a' in self.parameters:
                param = 'b/a'
            new_parameters.update({param: parameters[i]})

        if not self.coor_parameters:
            return new_parameters

        dir_map = ['x', 'y', 'z']
        relative_positions = np.dot(np.linalg.inv(
            atoms.cell.T), atoms.positions.T).T

        for i, aw in enumerate(self.atoms_wyckoffs):
            if aw == self.atoms_wyckoffs[i-1]:
                continue
            for d in self.w_free_param[aw]:
                w_p_name = dir_map[d] + aw
                new_parameters.update({w_p_name: relative_positions[i][d]})
        return new_parameters

    def get_move_pairs(self, distances, cutoff=1.1):
        move_indices = np.nonzero(
            distances / self.min_distances < self.hard_limit * cutoff)
        move_pairs = []
        if not len(move_indices[0]) > 0:
            return []

        for m in range(len(move_indices[0])):
            pair = sorted([move_indices[0][m], move_indices[1][m]])
            if np.any([self.free_atoms[a] for a in pair]) and pair[0] != pair[1]:
                move_pairs += [pair]
        if not move_pairs:
            return []

        move_pairs = np.unique(move_pairs, axis=0)
        clean_move_pairs = []
        del_pairs = []
        for i, pair in enumerate(move_pairs):
            sym_pairs = []
            for a in self.symmetry_map[str(pair[0])]:
                for b in self.symmetry_map[str(pair[1])]:
                    sym_pairs += [sorted([a, b])]
            if not np.any([sym_pair in move_pairs[:i] for sym_pair in sym_pairs]):
                clean_move_pairs += [pair]

        return clean_move_pairs

    def run_wyckoff_optimization_loop(self, atoms):
        Dm, distances = get_interatomic_distances(atoms)
        relative_distances = distances / self.min_distances
        move_pairs = self.get_move_pairs(distances, 1.1)
        atoms_distance = 1
        niter = 0
        images = [atoms.copy()]
        while len(move_pairs) > 0 and niter < atoms.get_number_of_atoms() * 3:
            for a1, a2 in move_pairs:
                transform_vector = Dm[a1][a2]
                transform_vector /= np.linalg.norm(transform_vector)

                move = -np.abs(1.02 - relative_distances[a1][a2]) * \
                    self.min_distances[a1][a2] * \
                    atoms_distance * self.hard_limit
                transform_vector *= move

                # Project onto wyckoff coordinate vector
                transform_vector1 = self.get_wyckoff_transform_vector(
                    transform_vector, a1, a1, atoms.cell)

                atoms[a1].position += transform_vector1

                if not a2 in self.symmetry_map.get(str(a1), []):
                    # No symmetry restrictions bewteen atom1 and atom2
                    transform_vector2 = self.get_wyckoff_transform_vector(
                        -transform_vector, a2, a2, atoms.cell)
                    atoms[a2].position += transform_vector2
                    sym_iter = [a1, a2]
                    trans_vectors = [transform_vector1, transform_vector2]
                else:
                    sym_iter = [a1]
                    trans_vectors = [transform_vector1]
                for i, ia in enumerate(sym_iter):
                    # Move all atoms in same wyckoff position as atom1
                    # and atom2
                    co_trans = self.symmetry_map.get(str(ia), []).copy()
                    if ia in co_trans:
                        co_trans.remove(ia)

                    for ic in co_trans:
                        transform_vector_w = \
                            self.get_wyckoff_transform_vector(
                                trans_vectors[i], ia, ic, atoms.cell)
                        atoms[ic].position += transform_vector_w

            atoms.wrap()
            images += [atoms.copy()]
            cell_volume = atoms.get_volume()
            Dm, distances = get_interatomic_distances(atoms)
            relative_distances = distances / self.min_distances
            atoms_distance *= 0.9
            move_pairs = self.get_move_pairs(distances, 1.1 * atoms_distance)
            niter += 1
        return images


def clean_parameter_input(cell_parameters):
    if 'a' in cell_parameters:
        a = cell_parameters['a']
        for l_c in ['b', 'c']:
            if l_c in cell_parameters:
                l = cell_parameters[l_c]
                del cell_parameters[l_c]
                cell_parameters.update({l_c + '/a': l/a})

    return cell_parameters


def get_interatomic_distances(atoms):
    Dm, distances = get_distances(atoms.positions,
                                  cell=atoms.cell, pbc=True)
    min_cell_width = np.min(np.linalg.norm(atoms.cell, axis=1))
    min_cell_width *= np.ones(len(atoms))
    np.fill_diagonal(distances, min_cell_width)

    return Dm, distances
