import os
import shlex
import subprocess

from protosearch.utils import get_basepath
from protosearch.build_bulk.classification import get_classification
from .calculator import get_calculator
from .vasp import get_poscar_from_atoms


class TriSubmit():
    """
    Set up (VASP) calculations on TRI-AWS for bulk structure created with the
    Bulk prototype enumerator developed by A. Jain described in:
    A. Jain and T. Bligaard, Phys. Rev. B 98, 214112 (2018)

    Parameters:

    atoms: ASE Atoms object
    ncpus: int
        number of cpus on AWS to use, default: 1
    queue: 'small', 'medium', etc
        Queue specificatin for AWS
    calculator: str
        'vaps' or 'espresso'
    calc_parameters: dict
        Optional specification of parameters, such as {ecut: 300}. 
        If not specified, the parameter standards given in 
        ./utils/standards.py will be applied
    basepath: str or None
        Path for job submission of in TRI filesync (s3) directory
        F.ex: '~/matr.io//model/<calculator>/1/u/<username>
    basepath_ext: str or None
        Extention to job submission path. Also works when specifying
        the environment variables explained below.

    Set the following environment valiables in order to set the
    job submission path automatically:
        TRI_PATH: Your TRI sync directory, which is usually at ~/matr.io
        TRI_USERNAME: Your TRI username
    """

    def __init__(self,
                 atoms,
                 ncpus=1,
                 queue='small',
                 calculator='vasp',
                 calc_parameters=None,
                 basepath=None,
                 basepath_ext=None
                 ):

        self.atoms = atoms
        self.poscar = get_poscar_from_atoms(atoms)

        prototype, self.cell_parameters = get_classification(atoms)
        self.spacegroup = prototype['spacegroup']
        self.wyckoffs = prototype['wyckoffs']
        self.species = prototype['species']
        self.cell_param_list = []
        self.cell_value_list = []
        for param in self.cell_parameters:
            self.cell_value_list += [self.cell_parameters[param]]
            self.cell_param_list += [param]

        if basepath:
            self.basepath = basepath
            if basepath_ext:
                self.basepath += '/{}'.format(basepath_ext)
            assert calculator in self.basepath, \
                'Your job submission path must match the calculator'
        else:
            self.basepath = get_basepath(calculator=calculator,
                                         ext=basepath_ext)

        self.calculator = calculator
        self.ncpus = ncpus
        self.queue = queue

        self.master_parameters = calc_parameters
        self.Calculator = self.get_calculator()
        self.calc_parameter_list, self.calc_values = \
            self.Calculator.get_parameters()

        dict_indices = [i for i, c in enumerate(self.calc_values)
                        if isinstance(c, dict)]
        for i in dict_indices:
            del self.calc_parameter_list[i]
            del self.calc_values[i]

    def submit_calculation(self):
        """Submit calculation for unique structure. 
        First the execution path is set, then the initial POSCAR and models.py
        are written to the directory.

        The calculation is submitted as a parametrized model with trisub.
        """

        self.set_execution_path()
        self.write_poscar(self.excpath)
        self.write_model(self.excpath)

        parameterstr_list = ['{}'.format(param)
                             for param in self.calc_values]
        parameterstr = '/' + '/'.join(parameterstr_list)

        command = shlex.split('trisub -p {} -q {} -c {}'.format(
            parameterstr, self.queue, self.ncpus))
        subprocess.call(command, cwd=self.excpath)

    def write_poscar(self, filepath):
        """Write POSCAR to specified file"""
        with open(filepath + '/initial.POSCAR', 'w') as f:
            f.write(self.poscar)

    def set_execution_path(self):
        """Create a unique submission path for each structure """

        # specify prototype for species
        path_ext = [str(self.spacegroup)]
        # wyckoffs at position
        species_wyckoffs_id = ''
        for spec, wy_spec in zip(self.species, self.wyckoffs):
            species_wyckoffs_id += spec + wy_spec
        path_ext += [species_wyckoffs_id]
        # cell parameters
        cell_param_id = ''
        if len(self.cell_param_list) < 10:
            for cell_key, cell_value in zip(self.cell_param_list,
                                            self.cell_value_list):

                cell_param_id += '{}{}'.format(cell_key, round(cell_value, 4)).\
                    replace('c/a', 'c').replace('b/a', 'b').\
                    replace('.', 'D').replace('-', 'M')

        path_ext += [cell_param_id]

        self.excroot = self.basepath
        for ext in path_ext:
            self.excroot += '/{}'.format(ext)
            if not os.path.isdir(self.excroot):
                os.mkdir(self.excroot)

        calc_revision = 1
        path_exists = True
        while os.path.isdir('{}/_{}'.format(self.excroot, calc_revision)):
            calc_revision += 1

        self.excpath = '{}/_{}'.format(self.excroot, calc_revision)
        os.mkdir(self.excpath)

    def get_calculator(self):
        symbols = self.atoms.symbols
        Calculator = get_calculator(self.calculator)
        return Calculator(symbols,
                          self.master_parameters,
                          self.ncpus)

    def write_model(self, filepath):
        """ Write model.py"""
        modelstr = self.Calculator.get_parametrized_model()

        with open(filepath + '/model.py', 'w') as f:
            f.write(modelstr)

    def write_simple_model(self, filepath):
        """ Write model.py"""
        modelstr = self.Calculator.get_model()

        with open(filepath + '/model_clean.py', 'w') as f:
            f.write(modelstr)
