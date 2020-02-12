import numpy as np
from numpy.random import rand, randint
from ase.visualize import view
from ase.geometry import get_distances, cell_to_cellpar
from ase.data import atomic_numbers as a_n
from ase.data import covalent_radii as cradii
from ase.spacegroup import crystal

from protosearch.ml_modelling.regression_model import get_regression_model
from protosearch.ml_modelling.fingerprint import clean_features
from .wyckoff_symmetries import WyckoffSymmetries
from .fitness_function import get_fitness, get_connections


class CellParameters(WyckoffSymmetries):
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
                 spacegroup=None,
                 wyckoffs=None,
                 species=None,
                 verbose=True
                 ):

        super().__init__(spacegroup=spacegroup,
                         wyckoffs=wyckoffs)

        self.spacegroup = spacegroup
        self.wyckoffs = wyckoffs
        self.species = species

        self.set_lattice_dof()
        if self.wyckoffs is not None:
            #self.p_name = self.get_prototype_name(species)
            self.parameter_guess = self.initial_guess()

        self.verbose = verbose

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

        self.natoms = atoms.get_number_of_atoms()

        parameter_guess.update({'a': mean_radii * 4 * self.natoms ** (1 / 3)})

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
            opt_cell_parameters = []

            if self.verbose:
                print('Running ML + GA for {} - {} - {}. {} atoms'
                      .format(self.spacegroup, self.wyckoffs, self.species,
                              self.natoms))

            atoms_list = \
                self.run_ml_ga_optimization(
                    master_parameters=master_parameters,
                    optimize_lattice=optimize_lattice,
                    optimize_angles=optimize_angles,
                    optimize_wyckoffs=optimize_wyckoffs,
                    max_candidates=max_candidates)

            for atoms in atoms_list:
                opt_parameters = \
                    self.get_cell_parameters(
                        atoms)  # .update(master_parameters)
                opt_cell_parameters += [opt_parameters]

        elif optimize_lattice:
            # Scale fixed lattice points
            opt_atoms_list = []
            opt_cell_parameters = []
            if self.verbose:
                print('Optimizing lattice constants for {} - {} - {}. {} atoms'
                      .format(self.spacegroup, self.wyckoffs, self.species,
                              self.natoms))

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

    def crossover(self, population, n_per_couple=10, n_children=None):
        """Generate new candidate by mixing values of two"""
        children = []
        variables = list(population[0].keys())
        for i, pop in enumerate(population):
            for j, pop2 in enumerate(population):
                if i == j or np.all(np.isclose(sorted(pop.values()),
                                               sorted(pop2.values()),
                                               rtol=0.05)):  # inbred child
                    continue
                for m in range(n_per_couple):
                    child = {}
                    for p in variables:
                        c_point = rand(1)[0]  # 0.5 + (rand(1)[0] - 0.5) * 0.5
                        value = pop[p] * c_point + pop2[p] * (1 - c_point)
                        child.update({p: value})

                    children += [child]
        n = n_children or len(population)
        return children[:n]

    def mutation(self, population, n_per_pop=10, n_children=None):
        mutated = []
        variables = list(population[0].keys())
        for i, pop in enumerate(population):
            for m in range(n_per_pop):
                mutant = pop.copy()
                indices = randint(len(variables), size=len(variables) // 2)

                for j in indices:
                    p = variables[j]
                    value = pop[p]
                    if p in self.coor_variables:
                        value += (rand(1)[0] - 0.5) * 0.1
                    elif p in self.angle_variables:
                        value = (rand(1)[0] - 0.5) * 30 + 90
                    mutant.update({p: value})
                mutated += [mutant]
            n = n_children or len(population)
        return mutated[:n]

    def run_ml_ga_optimization(self,
                               master_parameters=None,
                               optimize_lattice=True,
                               optimize_angles=True,
                               optimize_wyckoffs=True,
                               use_fitness_sharing=False,
                               batch_size=3,
                               max_candidates=1,
                               debug=False):
        """
        ML-accelerated Genetic algorithm optimization 
        for free wyckoff coordinates and lattice angles.
        """

        cell_parameters = self.initial_guess()
        if master_parameters:
            cell_parameters.update(master_parameters)
        population = []

        feature_variables = []
        if optimize_wyckoffs:
            feature_variables += self.coor_variables

        if optimize_angles:
            feature_variables += self.angle_variables

        n_parameters = len(feature_variables)
        population_size = min([5000, n_parameters * 100])
        population = []

        test_features = []
        for n in range(population_size):
            t_f = []
            parameters = {}
            if optimize_wyckoffs:
                for i, p in enumerate(self.coor_variables):
                    val = rand(1)[0] * 0.99
                    parameters.update({p: val})
                    t_f += [val]

            if optimize_angles:
                for i, p in enumerate(self.angle_variables):
                    val = (rand(1)[0] - 0.5) * 30 + 90
                    parameters.update({p: val})
                    t_f += [val]
            population += [parameters]
            test_features += [t_f]

        atoms = self.construct_atoms()
        covalent_radii = np.array([cradii[n] for n in atoms.numbers])
        M = covalent_radii * np.ones([len(atoms), len(atoms)])
        self.min_distances = (M + M.T)

        primitive_voronoi = len(atoms) > 64

        covalent_radii = np.array([cradii[n] for n in atoms.numbers])
        covalent_volume = np.sum(4/3 * np.pi * covalent_radii ** 3)
        cell_length = (covalent_volume * 2) ** (1/3)

        test_features = np.array(test_features)
        batch_indices = np.random.randint(len(test_features),
                                          size=batch_size)

        train_features = None

        fitness = np.array([])
        all_structures = []

        converged = False
        iter_id = 1
        train_population = []

        while not converged:
            bad_indices = []
            for i in batch_indices:
                pop = population[i]
                train_population += [pop]
                cell_parameters.update(pop)
                atoms = self.construct_atoms(cell_parameters)
                atoms = self.optimize_lattice_constants(atoms,
                                                        proximity=0.9,
                                                        optimize_wyckoffs=False)
                if atoms is None:
                    bad_indices +=[i]
                    continue
                parameters = cell_to_cellpar(atoms.get_cell())

                if primitive_voronoi:
                    # Use primitive cell for voronoi analysis for
                    # large systems
                    for i, param_name in enumerate(['a', 'b', 'c',
                                                    'alpha', 'beta', 'gamma']):
                        if param_name in cell_parameters:
                            cell_parameters.update({param_name: parameters[i]})

                    atoms = self.construct_atoms(cell_parameters,
                                                 primitive_cell=True)

                fit = get_fitness(atoms)
                connections = None
                if fit > -2:
                    # Don't do voronoi for very dilute structures
                    connections = get_connections(atoms, decimals=1)

                fitness = np.append(fitness, fit)
                all_structures += [{'parameters': cell_parameters,
                                    'atoms': atoms.copy(),
                                    'fitness': fit,
                                    'graph': connections}]

            best_fitness = np.max(fitness)
            print('iter {} best_fitnes:'.format(iter_id),  np.max(fitness))
            batch_indices = [idx for idx in batch_indices if not idx 
                             in bad_indices]
            if train_features is None:
                train_features = test_features[batch_indices]
            else:
                train_features = np.append(train_features,
                                           test_features[batch_indices],
                                           axis=0)

            test_features = np.delete(test_features, batch_indices, axis=0)
            test_features = np.delete(test_features, bad_indices, axis=0)

            population = np.delete(population, batch_indices, axis=0)

            indices = np.argsort(fitness)[::-1]
            ga_survived = np.array(train_population)[indices][:10]
            new_population = self.crossover(ga_survived) + \
                self.mutation(ga_survived)

            for pop in new_population:
                val = []
                for v in feature_variables:
                    val += [pop[v]]
                val = np.expand_dims(val, axis=0)

                test_features = np.append(test_features, val, axis=0)
                population = np.append(population, pop)

            features, bad_indices = \
                clean_features({'train': train_features,
                                'test': test_features})

            fitness = np.delete(fitness, bad_indices['train'])
            test_features = np.delete(test_features, bad_indices['test'],
                                      axis=0)
            train_features = np.delete(train_features, bad_indices['train'],
                                       axis=0)
            population = np.delete(population, bad_indices['train'])

            try:
                Model = get_regression_model('catlearn')(
                    features['train'],
                    np.array(fitness),
                    optimize_hyperparameters=True,
                    kernel_width=3,
                    #bounds=((0.5, 5),)
                )
            except:
                Model = get_regression_model('catlearn')(
                    features['train'],
                    np.array(fitness),
                    optimize_hyperparameters=False,
                    kernel_width=3)

            result = Model.predict(features['test'])
            predictions = result['prediction']
            unc = result['uncertainty']
            AQU = predictions + 0.5 * unc
            if debug:
                import pylab as p
                idx = np.argsort(predictions)
                p.plot(range(len(predictions)), predictions[idx])
                p.plot(range(len(predictions)),
                       predictions[idx] + unc[idx], '--')
                p.plot(range(len(predictions)),
                       predictions[idx] * 0 + best_fitness, '--')
                p.show()

            if iter_id > 30 or len(predictions) < 7:
                converged = True
            elif not np.max(AQU) > best_fitness and iter_id > 5:
                converged = True
            # elif best_fitness > 0.95:
            #    converged = True

            batch_indices = np.argsort(AQU)[::-1][:batch_size]
            iter_id += 1

        indices = np.argsort(fitness)[::-1][:max_candidates]
        fitness = fitness[indices]

        all_structures = np.array(all_structures)[indices]
        all_graphs = np.array([s['graph'] for s in all_structures])

        if fitness[0] < 0.8:
            indices = [0]
        else:
            indices = [i for i, f in enumerate(fitness) if
                       f > 0.8 and not
                       np.any(all_graphs[i] in all_graphs[:i])
                       ]

        if primitive_voronoi:
            # generate conventional structure
            all_atoms = [self.construct_atoms(all_structures[i]['parameters'])
                         for i in indices]
        else:
            all_atoms = [s['atoms'] for s in all_structures]
            all_atoms = [all_atoms[i] for i in indices]
        print('final fitness: ', fitness[indices])
        return all_atoms

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
                        onduplicates='keep',
                        symprec=1e-5)

        # if not primitive_cell:
        #    #print(len(relative_positions), len(atoms))
        #    if not len(relative_positions) == len(atoms):
        #        return None
        return atoms

    """
    def check_prototype(self, atoms):
        #Check that spacegroup and wyckoff positions did not change

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
    """

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
        if not atoms:
            return atoms

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
            min_dir_vector /= distances[min_distances]
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
                images += [atoms.copy()]

        if view_images:
            view(images)

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
        atoms_distance = 0.90
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
