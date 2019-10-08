import sys
import io
import time
import string
import numpy as np
from numpy.random import rand, randint
import ase
from ase import Atoms, Atom
from ase.visualize import view
from ase.geometry import get_distances, find_mic, cell_to_cellpar, cellpar_to_cell
from ase.data import atomic_numbers as a_n
from ase.data import covalent_radii as cradii
from ase.geometry.geometry import wrap_positions
from ase.spacegroup import crystal, get_spacegroup
from catkit.gen.utils.connectivity import get_voronoi_neighbors, get_cutoff_neighbors

from protosearch import build_bulk
from .wyckoff_symmetries import WyckoffSymmetries
from .classification import PrototypeClassification

path = build_bulk.__path__[0]


class CellParameters(WyckoffSymmetries):
    """
    Provides a fair estimate of cell parameters including lattice constants,
    angles, and free wyckoff coordinates. Users must suply either atoms object
    or spacegroup, wyckoff positons and species.

    Parameters:

    spacegroup: int
       int between 1 and 230
    wyckoffs: list
       wyckoff positions, for example ['a', 'a', 'b', 'c']
    species: list
       atomic species, for example ['Fe', 'O', 'O', 'O']
    wyckoffs_from_oqmd: bool
       Fetch positions from experimentally observed structure if present

    """

    def __init__(self,
                 spacegroup=None,
                 wyckoffs=None,
                 species=None,
                 verbose=True,
                 wyckoffs_from_oqmd=False
                 ):
        super().__init__(spacegroup=spacegroup,
                         wyckoffs=wyckoffs)

        self.spacegroup = spacegroup
        self.wyckoffs = wyckoffs
        self.species = species

        self.set_lattice_dof()

        self.p_name = self.get_prototype_name(species)

        self.parameter_guess = self.initial_guess()

        self.verbose = verbose
        self.wyckoffs_from_oqmd = wyckoffs_from_oqmd

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
            relax = [[0, 1], 2]  # a = b
            fixed_angles = {'alpha': 90, 'beta': 90,
                            'gamma': 120}

        elif 168 <= sg <= 194:  # Hexagonal
            relax = [[0, 1], 2]  # a = b
            fixed_angles = {'alpha': 90, 'beta': 90,
                            'gamma': 120}

        elif 195 <= sg <= 230:  # Cubic
            relax = [[0, 1, 2]]  # a = b = c
            fixed_angles = {'alpha': 90, 'beta': 90,
                            'gamma': 90}

        self.lattice_variables = \
            np.array(['a', 'b', 'c'])[[r[0] if isinstance(r, list)
                                       else r for r in relax]]

        self.angle_variables = \
            [a for a in ['alpha', 'beta', 'gamma']
             if not a in fixed_angles]

        self.d_o_f = relax
        self.fixed_angles = fixed_angles

    def initial_guess(self):
        """ Rough estimate of parameters """

        parameter_guess = {}

        covalent_radii = [cradii[a_n[s]] for s in self.species]
        mean_radii = np.mean(covalent_radii)

        for i, lat_par in enumerate(self.lattice_variables):
            parameter_guess.update({lat_par: mean_radii * 4 * (1 + 0.1 * i)})

        if 'alpha' in self.angle_variables:
            parameter_guess.update({'alpha': 88})

        if 'beta' in self.angle_variables:
            parameter_guess.update({'beta': 94})

        if 'gamma' in self.angle_variables:
            parameter_guess.update({'gamma': 86})

        for i, p in enumerate(self.coor_variables):
            parameter_guess.update({p: rand(1)[0] * 0.99})

        self.parameter_guess = parameter_guess

        atoms = self.construct_atoms()

        natoms = atoms.get_number_of_atoms()

        parameter_guess.update({'a': mean_radii * 4 * natoms ** (1 / 3)})

        return parameter_guess

    def get_parameter_estimate(self,
                               master_parameters=None,
                               proximity=0.90,
                               max_candidates=1):
        """
        Optimize lattice parameters for prototype. 

        Parameters:
        master_parameters: dict
           fixed cell parameters and values that will not be optimized.
        proximity: float close to 1
           Proximity of atoms, in the relative sum of atomic radii.
        """

        cell_parameters = self.parameter_guess
        master_parameters = master_parameters or {}

        if master_parameters:
            cell_parameters.update(master_parameters)

        optimize_wyckoffs = (not np.all([c in master_parameters for c
                                         in self.coor_variables])
                             and len(self.coor_variables) > 0)

        optimize_angles = (not np.all([c in master_parameters
                                       for c in self.angle_variables])
                           and len(self.angle_variables) > 0)

        optimize_lattice = not np.all([c in master_parameters for
                                       c in self.lattice_variables])

        if not np.any([optimize_wyckoffs, optimize_angles, optimize_lattice]):
            print('No parameters to optimize!')
            return [cell_parameters]

        if optimize_wyckoffs or optimize_angles:
            # Use simple GA to optimize parameters - returns several candidates
            atoms_list = self.run_ga_optimization(master_parameters=master_parameters,
                                                  optimize_lattice=optimize_lattice,
                                                  optimize_angles=optimize_angles,
                                                  optimize_wyckoffs=optimize_wyckoffs)
            if max_candidates:
                atoms_list = atoms_list[:max_candidates]
            opt_cell_parameters = []

            for atoms in atoms_list:
                opt_parameters = \
                    self.get_cell_parameters(atoms)

                opt_cell_parameters += [opt_parameters]

        elif optimize_lattice:
            # Scale fixed lattice points
            opt_atoms_list = []
            opt_cell_parameters = []
            if self.verbose:
                print('Optimizing lattice constants for {} - {} - {}'
                      .format(self.spacegroup, self.wyckoffs, self.species))

            atoms = self.construct_atoms(cell_parameters)
            opt_atoms = \
                self.optimize_lattice_constants(atoms,
                                                optimize_wyckoffs=optimize_wyckoffs,
                                                view_images=False,
                                                proximity=proximity)

            opt_cell_parameters = [self.get_cell_parameters(opt_atoms)]
        else:
            opt_cell_parameters = cell_parameters_list
            opt_atoms_list = atoms_list

        return opt_cell_parameters

    def get_reflections(self, wyckoff_coordinate):
        """ Use site symmetries to reduce the values of points that are sampled"""

        w_pos = wyckoff_coordinate[1]

        sym = self.wyckoff_site_symmetries[w_pos]

        if '4' in sym:
            return 1/8
        elif '3' in sym:
            return 1/4
        elif '2' in sym:
            return 1/2
        else:
            return 1

    def run_ga_optimization(self,
                            master_parameters=None,
                            optimize_lattice=True,
                            optimize_angles=True,
                            optimize_wyckoffs=True,
                            use_fitness_sharing=False):
        """
        Genetic algorithm optimization for free wyckoff coordinates
        and lattice angles, based on the hard-sphere volume of the structures.
        """

        cell_parameters = self.initial_guess()
        if master_parameters:
            cell_parameters.update(master_parameters)
        population = []

        if optimize_wyckoffs:
            self.reflections = {}
            for v in self.coor_variables:
                r = self.get_reflections(v)
                self.reflections.update({v: r})

        n_parameters = 0
        if optimize_angles:
            n_parameters += len(self.angle_variables)
        if optimize_wyckoffs:
            n_parameters += len(self.coor_variables)

        population_size = max([10, n_parameters * 5])

        for n in range(population_size):
            parameters = {}
            if optimize_wyckoffs:
                for i, p in enumerate(self.coor_variables):
                    parameters.update(
                        {p: rand(1)[0] * 0.99 * self.reflections[p]})
            if optimize_angles:
                for i, p in enumerate(self.angle_variables):
                    parameters.update({p: (rand(1)[0] - 0.5) * 30 + 90})
            population += [parameters]

        all_structures = []
        best_fitness = []
        best_generation_fitness = []
        j = 0
        while len(best_fitness) < 3 or len(set(best_fitness[-3:])) > 1:
            """Stop interation if search has not found better
            candidate in three rounds """
            volumes = []
            fitness = []
            graphs = []
            images = []
            for pop in population:
                cell_parameters.update(pop)
                atoms = self.construct_atoms(cell_parameters)
                if optimize_lattice:
                    atoms = self.optimize_lattice_constants(atoms,
                                                            proximity=0.95)
                if atoms is None:
                    continue

                volume = atoms.get_volume()
                connections = self.get_connections(atoms)
                fit = self.get_fitness(atoms)

                fitness += [fit]
                all_structures += [{'parameters': pop,
                                    'atoms': atoms.copy(),
                                    'volume': volume,
                                    'fitness': fit,
                                    'graph': connections}]

            j += 1
            if use_fitness_sharing:
                fitness_sharing_count = \
                    np.array([graph_connections.count(graph_connections[i])
                              for i in range(len(fitness))])
                fitness /= fitness_sharing_count

            indices = np.argsort(fitness)[::-1]
            survived = np.array(population)[indices][:population_size // 2]

            best_generation_fitness += [fitness[0]]
            best_fitness += [max(best_generation_fitness + best_fitness)]

            population = self.crossover(survived, population_size - 1) + \
                self.mutation(survived, 1)

            population = population[:population_size]
            j += 1

        all_fitness = np.array([s['fitness'] for s in all_structures])
        indices = np.argsort(all_fitness)[::-1]
        all_fitness = np.array(all_fitness)[indices]
        all_structures = np.array(all_structures)[indices]

        all_atoms = [s['atoms'] for s in all_structures]
        all_graphs = np.array([s['graph'] for s in all_structures])

        volume_0 = min([all_structures[i]['volume']
                        for i in range(len(all_structures))])

        indices = [i for i, f in enumerate(all_fitness) if not
                   f < -2 and not
                   np.any(all_graphs[i] in all_graphs[:i]) and not
                   all_structures[i]['volume'] > 5 * volume_0]
        all_atoms = [all_atoms[i] for i in indices]

        return all_atoms

    def get_fitness(self, atoms):
        """ Fitness of structure for the GA. Currently only works
        for Oxides and metals. Fitness is based on the prefered connectivity of atoms,
        as discussed in D. Waroquiers et al. Chem. Mater. 2017, 29, 8346−8360"""

        CM = get_cutoff_neighbors(atoms, cutoff=2.5)

        # Assume prefered O-con = 6 for now
        # favored_O_connections = {'Ti': {'3': 6, '4': 5.5},
        #                         'Cr': {'3': 6},
        #                         'Fe': {'3': 5.7, '4': 5.5},
        #                         'Sc': {'3': 6.1},
        #                         'Ti': {'3': 6, '4': 5.8},
        #                         'V': {'3': 6, '4': 5.5, '5': 4.5},
        #                         'Cr': {'2': 5, '3': 6, '4': 5.5, '5': 4.3, '6': 4},
        #                         'Mn': {'2': 5, '3': 6, '4': 5.5, '5': 4.3, '6': 4}}

        #favored_Oxy_states = {'Ti': 4, 'Cr': 3, 'Fe': 3, 'Sc': 3}

        metal_n = [3, 4, 11, 12, 13] + list(range(19, 32)) + list(range(37, 51)) + \
            list(range(55, 85))
        symbols = list(atoms.get_chemical_symbols())
        metals = [[i, a.symbol]
                  for i, a in enumerate(atoms) if a.number in metal_n]
        fitness = 0
        if 'O' in symbols:
            iOs = [i for i, a in enumerate(atoms) if a.symbol == 'O']
            restids = [i for i in range(len(atoms)) if not i in iOs]
            avg_oxy_state = symbols.count('O') * 2 / len(metals)
            for i, m in metals:
                #f_O_s = favored_Oxy_states[m]
                f_O_c = 6  # favored_O_connections[m][str(f_O_s)]
                key = '{}-{}'.format(*list(sorted(['O', m])))
                n_metal = symbols.count(m)
                fitness -= abs(sum(CM[i, iOs]) - f_O_c)

            fitness -= np.sum(CM[iOs, ][:, iOs]) / len(iOs)/2

            fitness -= np.sum(CM[restids, ][:, restids]) / len(restids)/2
        elif np.all([z in metal_n for z in atoms.get_atomic_numbers()]):
            fitness = np.sum(CM) / len(atoms)

        return fitness/len(atoms)

    def get_connections(self, atoms, cutoff=1):
        connectivity = get_voronoi_neighbors(atoms)
        atoms_connections = {}

        for i in range(len(atoms)):
            for j in range(len(atoms)):
                symbols = '-'.join(sorted([atoms[i].symbol, atoms[j].symbol]))
                if not symbols in atoms_connections:
                    atoms_connections[symbols] = 0
                atoms_connections[symbols] += connectivity[i][j] / 2

        return atoms_connections

    def get_n_cross_connections(self, atoms):
        connections = self.get_connections(atoms)
        n_cross_connections = 0
        n_same_connections = 0
        for pair in connections.keys():
            a, b = pair.split('-')
            if a == b and not a == 'O':
                n_same_connections += connections[pair]
            else:
                n_cross_connections += connections[pair]

        n_metal = len([1 for atom in atoms if not atom.symbol == 'O'])
        n_O = len([1 for atom in atoms if atom.symbol == 'O'])
        if n_O == 0:
            n_O = 1

        return n_cross_connections / n_metal / n_O

    def crossover(self, population, n_children=None):
        """Generate new candidate by mixing values of two"""
        children = []
        variables = list(population[0].keys())
        for i, pop in enumerate(population):
            for j, pop2 in enumerate(population):
                if i == j or np.all(np.isclose(sorted(pop.values()),
                                               sorted(pop2.values()),
                                               rtol=0.05)):  # inbred child
                    continue
                child = {}
                for p in variables:
                    c_point = 0.5 + (rand(1)[0] - 0.5) * 0.5
                    value = pop[p] * c_point + pop2[p] * (1 - c_point)
                    child.update({p: value})

                children += [child]
        n = n_children or len(population)
        return children[:n]

    def mutation(self, population, n_children=None):

        mutated = []
        variables = list(population[0].keys())
        for i, pop in enumerate(population):
            mutant = pop.copy()
            indices = randint(len(variables), size=len(variables) // 2)
            for j in indices:
                p = variables[j]
                value = pop[p]
                if p in self.coor_variables:
                    value = rand(1)[0] * self.reflections[p]
                elif p in self.angle_variables:
                    value = (rand(1)[0] - 0.5) * 30 + 90
                mutant.update({p: value})
            mutated += [mutant]
        n = n_children or len(population)
        return mutated[:n]

    def construct_atoms(self, cell_parameters=None, primitive_cell=False):
        if cell_parameters:
            self.parameter_guess.update(cell_parameters)

        relative_positions = []
        symbols = []

        for i_w, w in enumerate(self.wyckoffs):
            m = self.wyckoff_multiplicities[w]
            w_arrays = self.wyckoff_symmetries[w]

            wyckoff_coordinate = []
            for direction in ['x', 'y', 'z']:
                wyckoff_coordinate += [
                    self.parameter_guess.get(direction + w + str(i_w), 0)]

            for w_array in w_arrays:
                M = w_array[:, :3]
                c = w_array[:, 3]
                relative_positions += [
                    tuple(np.dot(wyckoff_coordinate, M.T) + c)]

            symbols += [self.species[i_w]] * m

        cellpar = []
        for p in ['a', 'b', 'c']:
            value = self.parameter_guess.get(p, None)
            if value is None:
                value = self.parameter_guess.get('a', 4)
            cellpar += [value]

        for angle in ['alpha', 'beta', 'gamma']:
            value = self.parameter_guess.get(angle, None)
            if not value:
                value = self.fixed_angles.get(angle)
            cellpar += [value]

        atoms = crystal(tuple(symbols),
                        relative_positions,
                        spacegroup=self.spacegroup,
                        cellpar=cellpar,
                        primitive_cell=primitive_cell,
                        onduplicates='keep')

        return atoms

    def check_prototype(self, atoms):
        """Check that spacegroup and wyckoff positions did not change"""

        PC = PrototypeClassification(atoms)

        prototype = PC.get_classification(include_parameters=False)

        if not self.spacegroup == prototype['spacegroup']:
            print('Spacegroup changed')
            return False

        elif not (self.wyckoffs == prototype['wyckoffs']
                  and self.species == prototype['species']):
            print('Wyckoff positions changed')
            return False
        else:
            return True

    def get_wyckoff_coordinates_oqmd(self):

        self.get_free_wyckoff_parameters()

        atoms = self.construct_atoms()
        oqmd_db = ase.db.connect(path + '/oqmd_ver3.db')

        structure = list(oqmd_db.select(proto_name=self.p_name,
                                        formula=atoms.get_chemical_formula(),
                                        limit=1))
        if len(structure) == 0:
            structure = list(oqmd_db.select(proto_name=self.p_name,
                                            limit=1))
        if len(structure) == 0:
            return None

        oqmd_parameters = self.get_cell_parameters(
            structure[0].toatoms(), True)
        try:
            for parameter in np.append(self.coor_variables, self.angle_variables):
                self.parameter_guess.update(
                    {parameter: oqmd_parameters[parameter]})
        except:
            return None
        return self.parameter_guess

    def optimize_lattice_constants(self,
                                   atoms,
                                   optimize_wyckoffs=False,
                                   view_images=False,
                                   proximity=0.95):
        """
        Optimize lattice parameters by reducing the cell size (one direction at
        the time) until atomic distances on the closest pair reaches the
        sum of the covalent radii.

        if the optimize_wyckoffs parameter is set, the wyckoff coordinates 
        are optimized together with the lattice constants. 
        """

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

            min_distances = np.unravel_index(
                np.argmin(distances/self.min_distances),
                distances.shape)
            min_dir_vector = Dm.copy()[min_distances]
            min_dir_vector /= np.linalg.norm(min_dir_vector)
            dot_directions = []
            for direction in self.d_o_f:
                v = np.zeros([3])
                v[direction] = 1
                v[direction] /= np.linalg.norm(v[direction])
                dot_directions += [np.abs(np.dot(v, min_dir_vector))]

            direction_order = np.argsort(dot_directions)[::-1]

            for direction in np.array(self.d_o_f)[direction_order]:
                self.hard_limit *= increment
                while np.all(distances > self.min_distances * self.hard_limit):
                    cell = atoms.cell.copy()
                    cell[direction, :] *= increment
                    atoms.set_cell(cell, scale_atoms=True)
                    Dm, distances = get_interatomic_distances(atoms.copy(),
                                                              Dm.copy(),
                                                              increment,
                                                              direction)

                    cell_volume = atoms.get_volume()
                    t = time.time() - t0
                images += [atoms.copy()]
            t = time.time() - t0

        if view_images:
            ase.visualize.view(images)

        return images[-1]

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
            for a in self.symmetry_map[pair[0]]:
                for b in self.symmetry_map[pair[1]]:
                    sym_pairs += [sorted([a, b])]
            if not np.any([sym_pair in move_pairs[:i] for sym_pair in sym_pairs]):
                clean_move_pairs += [pair]

        return clean_move_pairs

    def run_wyckoff_optimization_loop(self, atoms):
        Dm, distances = get_interatomic_distances(atoms)
        relative_distances = distances / self.min_distances
        move_pairs = self.get_move_pairs(distances, 1.1)
        atoms_distance = 0.9
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

                if not a2 in self.symmetry_map[a1]:
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
                    # Move all atoms in same wyckoff position as atom1 and atom2
                    co_trans = self.symmetry_map[ia].copy()
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
            atoms_distance *= 0.90
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


def get_interatomic_distances(atoms, D=None, scale=None, direction=None):
    if D is not None:
        D[:, :, direction] *= scale
        D.shape = (-1, 3)
        distances = np.sqrt((D**2).sum(1))
        D.shape = (-1, len(atoms), 3)
        distances.shape = (-1, len(atoms))

    else:
        D, distances = get_distances(atoms.positions,
                                     cell=atoms.cell, pbc=True)

    min_cell_width = np.min(np.linalg.norm(atoms.cell, axis=1))
    min_cell_width *= np.ones(len(atoms))
    np.fill_diagonal(distances, min_cell_width)

    return D, distances
