import io
import numpy as np
from numpy.random import rand
import ase
from ase import visualize
from ase.geometry import get_distances, cell_to_cellpar, cellpar_to_cell
from ase.data import atomic_numbers as a_n
from ase.data import covalent_radii as cradii
from ase.io.vasp import read_vasp, write_vasp
import bulk_enumerator as be


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
                 species):

        self.spacegroup = spacegroup
        self.wyckoffs = wyckoffs
        self.species = species

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

        if self.coor_parameters:
            if not np.all([c in master_parameters for c in self.coor_parameters]):
                coor_guess = self.get_wyckoff_coordinates()
                cell_parameters.update(coor_guess)
        if self.angle_parameters:
            if not np.all([c in master_parameters for c in self.angle_parameters]):
                angle_guess = self.get_angles(cell_parameters)
                cell_parameters.update(angle_guess)

        if not np.all([c in master_parameters for c in self.lattice_parameters]):
            print('Optimizing lattice constants')
            cell_parameters = self.get_lattice_constants(cell_parameters)

        atoms = self.get_atoms(fix_parameters=cell_parameters)

        if self.check_prototype(atoms):
            return cell_parameters
        else:
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

    def get_atoms(self, fix_parameters=None, primitive=False):
        """
        Get ASE atoms object generated with the Enumerator
        with parameters specified in `fix_parameters`. If all parameters
        are not provided, a very rough estimate will be applied.
        """

        if fix_parameters:
            self.parameter_guess.update(fix_parameters)

        parameter_guess_values = []
        for p in self.parameters:
            parameter_guess_values += [self.parameter_guess[p]]

        b = be.bulk.BULK()
        b.set_spacegroup(self.spacegroup)
        b.set_wyckoff(self.wyckoffs)
        b.set_species(self.species)

        parameters = b.get_parameters()

        b.set_parameter_values(parameters, parameter_guess_values)
        if primitive:
            poscar = b.get_primitive_poscar()
        else:
            poscar = b.get_std_poscar()
        b.delete()
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
        w2 = b2.get_wyckoff()

        b2.delete()
        if not sg2 == self.spacegroup:
            print('Symmetry reduced to {} from {}'.format(sg2, self.spacegroup))
            return False
        if not w2 == self.wyckoffs:
            print('Wyckoffs reduced to {} from {}'.format(w2, self.wyckoffs))
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
            parameter_guess.update({'alpha': 85})

        if 'beta' in self.parameters:
            parameter_guess.update({'beta': 85})

        if 'gamma' in self.parameters:
            parameter_guess.update({'gamma': 85})

        for i, p in enumerate(self.coor_parameters):
            parameter_guess.update({p: rand(1)[0] * 0.9})

        self.parameter_guess = parameter_guess
        atoms = self.get_atoms(parameter_guess)
        natoms = atoms.get_number_of_atoms()

        parameter_guess.update({'a': mean_radii * 4 * natoms ** (1 / 3)})

        return parameter_guess

    def get_wyckoff_coordinates(self, view_images=False):
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
        distances = get_interatomic_distances(atoms)
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
            print('Wyckoff coordinate iteration {}, conv: {}'.format(j, Diff))
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
                    distances = get_interatomic_distances(atoms)

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
        atoms = self.get_atoms(fix_parameters)
        step_size = 1
        Volume0 = 10000
        volume0 = Volume0
        volume = 1
        direction = -1
        diff = 1
        j = 1
        Diff = 1
        while Diff > 0.01:  # Outer convergence criteria
            print('Angle iteration {}'.format(j))
            for angle in self.angle_parameters:
                direction = -1
                step_size = 1
                angle0 = self.parameter_guess[angle]
                diff = 1
                i = 0
                while diff > 0.05:  # and i < 5: #for i in range(5):
                    i += 1
                    angletest = angle0 + direction * 20 / j * step_size
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
                    diff = abs(volume0 - volume) / volume0
                    if volume < volume0:
                        angle0 = angletest
                        self.parameter_guess.update({angle: angle0})
                        fix_parameters.update({angle: angle0})
                        volume0 = volume
                    else:
                        direction *= -1

                    step_size *= 0.5

            Diff = abs(volume0 - Volume0) / Volume0
            Volume0 = volume0
            j += 1

        return self.parameter_guess

    def get_lattice_constants(self, fix_parameters={}, proximity=1.0):
        """
        Get lattice constants by reducing the cell size (one direction at
        the time) until atomic distances on the closest pair reaches the
        sum of the covalent radii.
        """

        if not fix_parameters:
            fix_parameters = self.parameter_guess

        atoms = self.get_atoms(fix_parameters)
        cell = atoms.cell

        distances = get_interatomic_distances(atoms)

        covalent_radii = np.array([cradii[n] for n in atoms.numbers])
        M = covalent_radii * np.ones([len(atoms), len(atoms)])
        min_distances = (M + M.T) * proximity

        # scale up or down
        soft_limit = 1.2
        scale = np.min(distances / min_distances / soft_limit)
        atoms.set_cell(atoms.cell * 1 / scale, scale_atoms=True)

        distances = get_interatomic_distances(atoms)
        hard_limit = soft_limit
        while np.all(distances > min_distances):
            for direction in self.d_o_f:
                hard_limit *= 0.90
                while np.all(distances > min_distances * hard_limit):
                    cell = atoms.cell.copy()
                    cell[direction, :] *= 0.90
                    atoms.set_cell(cell, scale_atoms=True)
                    distances = get_interatomic_distances(atoms)

        cell = atoms.cell
        new_parameters = cell_to_cellpar(cell)
        new_parameters[1:3] /= new_parameters[0]

        for i, param in enumerate(['a', 'b/a', 'c/a',
                                   'alpha', 'beta', 'gamma']):
            if param in self.parameters:
                fix_parameters.update({param: new_parameters[i]})
        return fix_parameters


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

    return distances
