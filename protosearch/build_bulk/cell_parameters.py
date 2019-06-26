import io
import numpy as np
from numpy.random import rand
import ase
from ase import visualize
from ase.geometry import get_distances, cell_to_cellpar, cellpar_to_cell
from ase.data import atomic_numbers as a_n
from ase.data import covalent_radii as cradii
from ase.io.vasp import read_vasp, write_vasp
from ase.geometry.geometry import wrap_positions
import bulk_enumerator as be

from protosearch.build_bulk.classification import get_classification


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
                print('Optimizing lattice constants')
            cell_parameters = self.get_lattice_constants(cell_parameters,
                                                         optimize_wyckoffs)
        cell_parameters.update(master_parameters)
        atoms = self.get_atoms(fix_parameters=cell_parameters)

        if self.check_prototype(atoms):
            out = cell_parameters
        else:
            if self.verbose:
                print("Structure reduced to another spacegroup")
            out = None
        return(out)

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
            relax = [[0, 1], 2]  # a = c
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
        write_vasp(filename=poscar, atoms=atoms, vasp5=True,
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
                print('Symmetry reduced to {} from {}'.format(
                    sg2, self.spacegroup))
            return False
        if not wyckoff_species2 == wyckoff_species2:
            if self.verbose:
                print('Wyckoffs reduced to {} from {}'.format(
                    w2, self.wyckoffs))
            return False

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
            parameter_guess.update({p: rand(1)[0] * 0.9})

        self.parameter_guess = parameter_guess

        atoms = self.get_atoms()

        natoms = atoms.get_number_of_atoms()

        parameter_guess.update({'a': mean_radii * 4 * natoms ** (1 / 3)})

        return parameter_guess

    def get_wyckoff_coordinates(self, view_images=False):
        # Determine high-symmetry positions taken in the unit cell.
        atoms = self.get_atoms()
        relative_positions = np.dot(atoms.cell, atoms.positions.T).T / \
            np.linalg.norm(atoms.cell, axis=1) ** 2
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
        for w in set(self.wyckoffs):
            w_free_param.update({w: []})
            for i in range(0, 100):
                for c in [c for c in self.coor_parameters
                          if w + str(i) == c[1:]]:
                    direction = c.replace(w + str(i), '')
                    xyz = dir_map[direction]
                    if not xyz in w_free_param[w]:
                        w_free_param[w] += [dir_map[direction]]
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
            variables += (np.random.random(len(variables)) - 0.5) / 10

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
        relative_positions = np.dot(atoms.cell, atoms.positions.T).T / \
            np.linalg.norm(atoms.cell, axis=1) ** 2
        sorted_atoms = ase.Atoms(cell=atoms.cell, pbc=True)
        multiplicity = []
        atoms_wyckoffs = []
        symmetry_map = {}
        position_symmetries = []
        self.free_atoms = []
        for i_w, w in enumerate(self.wyckoffs):
            species[i_w] = self.species[i_w]
            atoms_test = self.get_atoms(fix_parameters=fix_parameters,
                                        species=species)

            count_added = 0
            for atom_test in atoms_test:
                for i, atom in enumerate(atoms):
                    if np.all(atom_test.position == atom.position) \
                       and not atom_test.symbol == atom.symbol:
                        count_added += 1
                        sorted_atoms += atom_test
                        atoms[i].symbol = atom_test.symbol
                        atoms_wyckoffs += [w]
                        if i_w in self.fixed_wyckoff_idx:
                            self.free_atoms += [0]
                        else:
                            self.free_atoms += [1]
                        if count_added == 1:
                            continue
                        # if multiplicity > 1, map out symmetries
                        pos1 = np.array(sorted_atoms[-1].position)
                        pos2 = np.array(sorted_atoms[-2].position)

                        pos2min = \
                            np.array(wrap_positions([-sorted_atoms[-2].position],
                                                    atoms.cell)[0])
                        even = np.where(np.isclose(pos1, pos2))[0]
                        odd = np.where(np.isclose(pos1, pos2min))[0]

                        sym = []
                        for i in range(3):
                            if i in even:
                                sym += [1]
                            elif i in odd:
                                sym += [-1]
                            else:
                                sym += [0]
                        n1 = str(len(sorted_atoms) - 2)
                        n2 = str(len(sorted_atoms) - 1)

                        symmetry_map[n1] = {n2: sym}
                        symmetry_map[n2] = {n1: sym}

            multiplicity += [count_added] * count_added

            if count_added == 1:
                continue

        self.symmetry_map = symmetry_map
        self.atoms_wyckoffs = atoms_wyckoffs
        self.multiplicity = multiplicity
        return sorted_atoms

    def get_wyckoff_coordinates_old(self, view_images=False):
        """
        Get an estimate for free wyckoff coordinates. The positions are
        optimized from the interatomic distances, d, by mininizing
        the repulsion R = \Sum_{i>j} 1/d^12_{ij}.

        Since the initial guess is random, the structure is going to be
        different for each run. *** Other solutions for this?
        """

        atoms = self.get_atoms()
        natoms = atoms.get_number_of_atoms()

        # get triangle of matrix without diagonal
        Dm, distances = get_interatomic_distances(atoms)
        R0 = np.sum(1 / (distances ** 12))  # initial repulsion
        r0 = R0
        fix_parameters = {}
        for coor_param in self.coor_parameters:
            fix_parameters.update(
                {coor_param: self.parameter_guess[coor_param]})

        images = []
        direction = 1
        Diff = 1
        j = 1
        while Diff > 0.05 and j < 11:  # Outer convergence criteria
            if self.verbose:
                print('Wyckoff coordinate iteration {}, conv: {}'.format(
                    j, Diff))

            # Change one parameter at the time
            for coor_param in self.coor_parameters:
                cp0 = self.parameter_guess[coor_param]
                diff = 1
                step_size = 1
                k = 1
                while diff > 0.05 and k < 10:  # Inner loop convergence criteria
                    cptest = cp0 + direction * 0.2 / j * step_size
                    temp_parameters = fix_parameters.copy()
                    temp_parameters.update({coor_param: cptest})
                    try:
                        atoms = self.get_atoms(temp_parameters)
                    except:
                        continue
                    Dm, distances = get_interatomic_distances(atoms)

                    r = np.sum(1 / (distances ** 12))
                    diff = abs(r - r0) / r0

                    if r < r0:  # Lower repulsion - apply change
                        k += 1
                        cp0 = cptest
                        self.parameter_guess.update({coor_param: cp0})
                        fix_parameters.update({coor_param: cp0})
                        r0 = r
                        images += [atoms]
                    else:  # Try other direction, and decrease step size
                        direction *= -1
                        step_size += 0.9

            Diff = abs(r0 - R0) / R0
            R0 = r0
            j += 1

        if view_images:
            ase.visualize.view(images)
        return self.parameter_guess

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
        """

        if not fix_parameters:
            fix_parameters = self.parameter_guess

        if optimize_wyckoffs:
            atoms = self.get_sorted_atoms(fix_parameters)
        else:
            atoms = self.get_atoms(fix_parameters)

        cell = atoms.cell

        Dm, distances = get_interatomic_distances(atoms)
        covalent_radii = np.array([cradii[n] for n in atoms.numbers])
        M = covalent_radii * np.ones([len(atoms), len(atoms)])
        self.min_distances = (M + M.T) * proximity

        # scale up or down
        images = [atoms.copy()]
        soft_limit = 1.3
        scale = np.min(distances / self.min_distances / soft_limit)
        atoms.set_cell(atoms.cell * 1 / scale, scale_atoms=True)

        Dm, distances = get_interatomic_distances(atoms)
        self.hard_limit = soft_limit
        images += [atoms.copy()]

        self.covalent_volume = np.sum(4/3 * np.pi * covalent_radii ** 3)
        cell_volume = atoms.get_volume()
        density = self.covalent_volume / cell_volume
        niter = 0
        while density < 0.3:
            for direction in self.d_o_f:
                self.hard_limit *= 0.95
                while np.all(distances > self.min_distances * self.hard_limit):
                    cell = atoms.cell.copy()
                    cell[direction, :] *= 0.95
                    atoms.set_cell(cell, scale_atoms=True)
                    Dm, distances = get_interatomic_distances(atoms)
                    cell_volume = atoms.get_volume()
                    density = self.covalent_volume / cell_volume

                images += [atoms.copy()]
            if optimize_wyckoffs:
                images += self.run_wyckoff_optimization_loop(atoms)
                atoms = images[-1].copy()
                Dm, distances = get_interatomic_distances(atoms)
            niter += 1

        if np.any(distances < self.min_distances):
            scale = np.min(distances / self.min_distances)
            atoms.set_cell(atoms.cell * 1 / scale, scale_atoms=True)
            images += [atoms.copy()]

        if view_images:
            ase.visualize.view(images)

        prototype, parameter_dict = get_classification(atoms)

        for param in parameter_dict:
            if param in ['b', 'c']:
                fix_parameters.update({param + '/a': parameter_dict[param]})
            else:
                fix_parameters.update({param: parameter_dict[param]})

        return fix_parameters

    def get_move_pairs(self, distances):
        move_indices = np.nonzero(distances / self.min_distances < 1)
        move_pairs = []
        if len(move_indices[0]) > 0:
            for m in range(len(move_indices[0])):
                pair = sorted([move_indices[0][m], move_indices[1][m]])
                if np.any([self.free_atoms[a] for a in pair]) and pair[0] != pair[1]:
                    move_pairs += [pair]
            if move_pairs:
                move_pairs = np.unique(move_pairs, axis=0)
        return move_pairs

    def run_wyckoff_optimization_loop(self, atoms):
        Dm, distances = get_interatomic_distances(atoms)
        relative_distances = distances / self.min_distances
        move_pairs = self.get_move_pairs(distances)
        atoms_proxy = 0.9
        niter = 0
        images = []
        while len(move_pairs) > 0 and niter < 10:
            for a1, a2 in move_pairs:
                if not np.any([self.free_atoms[a] for a in [a1, a2]]):
                    continue
                if a1 == a2:
                    continue
                transform_vector = Dm[a1][a2]
                transform_vector /= np.linalg.norm(transform_vector)
                free_a1 = self.w_free_param[self.atoms_wyckoffs[a1]]
                free_a2 = self.w_free_param[self.atoms_wyckoffs[a2]]

                move = (1.02 - relative_distances[a1][a2]) * \
                    self.min_distances[a1][a2] * atoms_proxy
                if a1 in self.symmetry_map and a2 in self.symmetry_map:
                    if a2 in self.symmetry_map[a1]:
                        trans_dir1 = np.array([-1, -1, -1])[free_a1]
                        trans_dir2 = trans_dir1 * \
                            self.symmetry_map[a1][a2][free_a2]
                        symmetry_pair = False
                elif a1 in self.symmetry_map or a2 in self.symmetry_map:
                    symmetry_pair = True
                else:
                    trans_dir1 = np.array([-1, -1, -1])[free_a1]
                    trans_dir2 = np.array([1, 1, 1])[free_a2]
                    symmetry_pair = False
                atoms[a1].position[free_a1] += transform_vector[free_a1] * \
                    trans_dir1 * move
                atoms[a2].position[free_a2] += transform_vector[free_a2] * \
                    trans_dir2 * move
                for a in self.symmetry_map:
                    ia = int(a)
                    if ia in [a1, a2]:
                        co_trans = int(list(self.symmetry_map[a].keys())[0])
                        if co_trans in [a1, a2]:
                            continue
                        sym = np.array(list(self.symmetry_map[a].values())[0])
                        if ia == a1:
                            trans_dir = trans_dir1 * sym[free_a1]
                            free = free_a1
                        elif ia == a2:
                            trans_dir = trans_dir2 * sym[free_a2]
                            free = free_a2
                        atoms[co_trans].position[free] += \
                            transform_vector[free] * trans_dir * move
                atoms.wrap()
            images += [atoms.copy()]
            cell_volume = atoms.get_volume()
            density = self.covalent_volume / cell_volume
            Dm, distances = get_interatomic_distances(atoms)
            relative_distances = distances / self.min_distances
            move_pairs = self.get_move_pairs(distances)
            atoms_proxy *= 0.9
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
