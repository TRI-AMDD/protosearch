import sys
import os
import io
import subprocess

import ase
from ase.calculators.vasp import Vasp
from ase.io.vasp import read_vasp
import bulk_enumerator as be

from protosearch.utils.standards import Standards
from protosearch.utils.valence import Valence
from protosearch.workflow.prototype_db import PrototypeSQL
from .cell_parameters import CellParameters

from protosearch import __version__ as version


class BuildBulk(CellParameters):
    """
    Set up Vasp calculations on TRI-AWS for bulk structure created with the
    Bulk prototype enumerator developed by A. Jain described in:
    A. Jain and T. Bligaard, Phys. Rev. B 98, 214112 (2018)

    The module takes the following parameters:

    spacegroup: int
       int between 1 and 230
    wyckoffs: list
       wyckoff positions, for example ['a', 'a', 'b', 'c']
    species: list
       atomic species, for example ['Fe', 'O', 'O', 'O']
    ncpus: int
       number of cpus on AWS to use, default: 1
    queue: 'small', 'medium',
       Queue specificatin for AWS
    calc_parameters: dict
       Optional specification of parameters, such as {ecut: 300}. 
       If not specified, the parameter standards given in 
       ./utils/standards.py will be applied
    cell_parameters: dict
       Optional specification of cell parameters, 
       such as {'a': 3.7, 'alpha': 75}.
       Otherwise a fair guees for parameters will be provided by the
       CellParameters module.

    The following environment valiables must be set in order to submit to AWS:

    TRI_PATH: Your TRI sync directory, which is usually at ~/matr.io
    TRI_USERNAME: Your TRI username
    """

    def __init__(self,
                 spacegroup,
                 wyckoffs,
                 species,
                 ncpus=1,
                 queue='small',
                 calc_parameters={},
                 cell_parameters={}
                 ):

        super().__init__(spacegroup=spacegroup,
                         wyckoffs=wyckoffs,
                         species=species)

        assert (0 < spacegroup < 231 and isinstance(spacegroup, int)), \
            'Spacegroup must be an integer between 1 and 230'

        self.spacegroup = spacegroup
        self.wyckoffs = wyckoffs
        self.species = species

        TRI_PATH = os.environ['TRI_PATH']
        username = os.environ['TRI_USERNAME']
        self.basepath = TRI_PATH + '/model/vasp/1/u/{}'.format(username)

        self.ncpus = ncpus
        self.queue = queue

        self.calc_parameters = calc_parameters
        self.calc_value_list = []
        for param in Standards.sorted_calc_parameters:
            if param in self.calc_parameters:
                self.calc_value_list += [self.calc_parameters[param]]
            else:
                self.calc_value_list += [Standards.calc_parameters[param]]

        self.cell_parameters = self.get_parameter_estimate()
        self.cell_parameters.update(cell_parameters)

        self.cell_param_list = []
        self.cell_value_list = []
        for param in Standards.sorted_cell_parameters:
            if param in self.cell_parameters:
                self.cell_value_list += [self.cell_parameters[param]]
                self.cell_param_list += [param]

        self.poscar = self.get_poscar()
        self.atoms = read_vasp(io.StringIO(self.poscar))

        nbands_index = Standards.sorted_calc_parameters.index('nbands')
        nbands = self.calc_value_list[nbands_index]
        if nbands < 0:
            self.calc_value_list[nbands_index] = self.get_nbands(
                n_empty=abs(nbands))
        else:
            self.calc_value_list[nbands_index] = self.get_nbands()

    def submit_calculation(self):
        """Submit calculation for unique structure. 
        First the execution path is set, then the initial POSCAR and models.py
        is written to the directory.

        The calculation is submitted as a parametrized model with trisub.
        """

        self.set_execution_path()

        self.write_poscar(self.excpath + '/initial.POSCAR')
        self.write_model(self.excpath + '/model.py')

        parameterstr_list = ['{}'.format(param)
                             for param in self.calc_value_list]
        parameterstr = '/' + '/'.join(parameterstr_list)
        print(parameterstr)
        subprocess.call(
            ('trisub -p {} -q {} -c {}'.
             format(parameterstr, self.queue, self.ncpus)
             ).split(), cwd=self.excpath)

    def get_poscar(self):
        """Get POSCAR structure file from the Enumerator """
        b = be.bulk.BULK()
        b.set_spacegroup(self.spacegroup)
        b.set_wyckoff(self.wyckoffs)
        b.set_species(self.species)

        b.set_parameter_values(self.cell_param_list, self.cell_value_list)
        self.prototype_name = b.get_name()

        return b.get_primitive_poscar()

    def write_poscar(self, filepath):
        """Write POSCAR to specified file"""
        with open(filepath, 'w') as f:
            f.write(self.poscar)

    def set_execution_path(self):
        """Create a unique path for each structure for submission """

        variables = ['BB']  # start BuildBulk identification
        variables = [version.replace('.', '')]
        variables += ['PT']  # specify prototype for species
        variables += [str(self.spacegroup)]

        for spec, wy_spec in zip(self.species, self.wyckoffs):
            variables += [spec + wy_spec]

        self.p_name = ''.join(variables[3:])

        variables += ['CP']  # Cell Parameters
        for cell_key, cell_value in zip(self.cell_param_list,
                                        self.cell_value_list):

            variables += ['{}{}'.format(cell_key, cell_value)]

        calc_name = ''.join(variables).replace('c/a', 'c').replace('b/a', 'b').\
            replace('.', 'D').replace('-', 'M')

        self.excroot = '{}/{}'.format(self.basepath, calc_name)
        if not os.path.isdir(self.excroot):
            os.mkdir(self.excroot)

        calc_revision = 1
        path_exists = True
        while os.path.isdir('{}/_{}'.format(self.excroot, calc_revision)):
            calc_revision += 1

        self.excpath = '{}/_{}'.format(self.excroot, calc_revision)
        os.mkdir(self.excpath)

    def write_model(self, filepath):
        """ Write a model.py to the calc directory, which uses the ASE interface
        to Vasp.
        Since a parameterized model will be submitted with trisub, 
        the parameters are given the values #1, #2, #3... etc"""

        modelstr = '#!/usr/bin/env python\n\n'
        modelstr += 'from ase.io import read\n' + \
            'from ase.units import Rydberg as Ry\n' + \
            'from ase.calculators.vasp import Vasp\n\n'

        modelstr += "atoms = read('initial.POSCAR')\n\n"

        for i, calc_key in enumerate(Standards.sorted_calc_parameters):
            factor = None
            if calc_key in Standards.calc_decimal_parameters:
                factor = Standards.calc_decimal_parameters[calc_key]
            if factor:
                modelstr += '{} = #{} * {}\n'.format(calc_key, i+1, factor)

            elif isinstance(self.calc_value_list[i], str):
                modelstr += "{} = '#{}'\n".format(calc_key, i+1)
            else:
                modelstr += '{} = #{}\n'.format(calc_key, i+1)

        modelstr += '\ncalc = Vasp(\n'
        for i, calc_key in enumerate(Standards.sorted_calc_parameters):
            modelstr += '    {}={},\n'.format(calc_key, calc_key)

        modelstr += ')\n\ncalc.calculate(atoms)\n'

        with open(filepath, 'w') as f:
            f.write(modelstr)

    def get_nbands(self, n_empty=5):
        """ get the number of bands from structure, based on the number of
        valence electrons listed in utils/valence.py"""
        elements = [atom.symbol for atom in self.atoms]

        N_ions = len(self.atoms)
        N_val = 0

        for ele in elements:
            N_val += Valence.__dict__.get(ele)

        nbands = int(N_val / 2 + N_ions / 2 + n_empty)
        nbands += nbands % self.ncpus

        return nbands
