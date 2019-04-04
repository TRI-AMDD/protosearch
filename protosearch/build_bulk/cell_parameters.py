import io
import numpy as np
from numpy.random import rand
import ase
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

        self.b = be.bulk.BULK()
        self.b.set_spacegroup(self.spacegroup)
        self.b.set_wyckoff(self.wyckoffs)
        self.b.set_species(self.species)
        self.parameters = self.b.get_parameters()
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

    def get_parameter_estimate(self):
        """ 
        Optimize lattice parameters for Atoms object generated with the bulk
        Enumerator.
        First wyckoff coordinates are optimized, then the angles, and at last
        the lattice constant.
        """

        fix_parameters = self.parameter_guess
        if self.coor_parameters:
            coor_guess = self.get_wyckoff_coordinates()
            fix_parameters.update(coor_guess)
        if np.any([angle in self.parameters for angle in
                   ['alpha', 'beta', 'gamma']]):
            angle_guess = self.get_angles(fix_parameters)
            fix_parameters.update(angle_guess)

        fix_parameters = self.get_lattice_constants(fix_parameters)

        if self.check_prototype(self.atoms):
            return fix_parameters
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

    def get_atoms(self, fix_parameters=None):
        """
        Get ASE atoms object generated with the Enumerator 
        with parameters specified in `fix_parameters`. If all parameters
        are not provided, a very rough estimate will be applied.
        """
        self.parameter_guess.update(fix_parameters or {})

        parameter_guess_values = []
        for p in self.parameters:
            parameter_guess_values += [self.parameter_guess[p]]

        self.b.set_parameter_values(self.parameters, parameter_guess_values)
        poscar = self.b.get_std_poscar()
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

        if not sg2 == self.spacegroup \
           or not w2 == self.wyckoffs:
            print('Symmetry reduced to {}, {}'.format(b2.get_spacegroup(), b2))
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

        parameter_guess_values = []
        for p in self.parameters:
            parameter_guess_values += [parameter_guess[p]]

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

        atoms *= (2, 2, 2)

        # get triangle of matrix without diagonal
        idx = np.triu_indices(len(atoms), 1)
        Dm, distances = get_distances(
            atoms.positions, cell=atoms.cell, pbc=True)
        R0 = np.sum(1 / (distances[idx] ** 12))  # initial repulsion
        r0 = R0
        fix_parameters = {}
        for coor_param in self.coor_parameters:
            fix_parameters.update(
                {coor_param: self.parameter_guess[coor_param]})

        images = []
        direction = 1
        Diff = 1
        j = 1
        while Diff > 0.01:  # Outer convergence criteria
            print('Wyckoff coordinate iteration {}, conv: {}'.format(j, Diff))
            # Change one parameter at the time
            for coor_param in self.coor_parameters:
                cp0 = self.parameter_guess[coor_param]
                diff = 1
                step_size = 1
                while diff > 0.01:  # Inner loop convergence criteria
                    cptest = cp0 + direction * 0.2 / j * step_size
                    temp_parameters = fix_parameters.copy()
                    temp_parameters.update({coor_param: cptest})
                    try:
                        atoms = self.get_atoms(temp_parameters) * (2, 2, 2)
                    except:
                        continue

                    Dm, distances = get_distances(atoms.positions,
                                                  cell=atoms.cell,
                                                  pbc=True)

                    r = np.sum(1 / (distances[idx] ** 12))
                    diff = abs(r - r0) / r0

                    if r < r0:  # Lower repulsion - apply change
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

    def get_lattice_constants(self, fix_parameters={}):
        """
        Get lattice constants by reducing the cell size (one direction at 
        the time) until atomic distances on the closest pair reaches the 
        sum of the covalent radii. 
        """

        if not fix_parameters:
            fix_parameters = self.parameter_guess

        atoms = self.get_atoms(fix_parameters)
        cell0 = atoms.cell  # initial cell
        atoms *= (2, 2, 2)

        Dm, distances = get_distances(
            atoms.positions, cell=atoms.cell, pbc=True)

        covalent_radii = np.array([cradii[n] for n in atoms.numbers])

        M = covalent_radii * np.ones([len(atoms), len(atoms)])

        min_distances = (M + M.T) * 1.2
        np.fill_diagonal(min_distances, 0)

        while np.any(distances < min_distances * 1.2):
            atoms.set_cell(atoms.cell * 1.1, scale_atoms=True)
            Dm, distances = get_distances(atoms.positions,
                                          cell=atoms.cell, pbc=True)

        soft_limit = 1.5
        while np.all(distances >= min_distances * soft_limit):
            atoms.set_cell(atoms.cell * 0.9, scale_atoms=True)
            Dm, distances = get_distances(atoms.positions,
                                          cell=atoms.cell, pbc=True)

        hard_limit = soft_limit
        while np.all(distances >= min_distances):
            for direction in self.d_o_f:
                hard_limit *= 0.95
                while np.all(distances >= min_distances * hard_limit):
                    cell = atoms.cell.copy()
                    cell[direction, :] *= 0.9
                    atoms.set_cell(cell, scale_atoms=True)
                    Dm, distances = get_distances(atoms.positions,
                                                  cell=atoms.cell, pbc=True)

        cell = atoms.cell / 2

        new_parameters = cell_to_cellpar(cell)
        new_parameters[1:3] /= new_parameters[0]

        for i, param in enumerate(['a', 'b/a', 'c/a',
                                   'alpha', 'beta', 'gamma']):
            if param in self.parameters:
                fix_parameters.update({param: new_parameters[i]})

        return fix_parameters
