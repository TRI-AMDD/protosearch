import os
import shlex
import subprocess

from protosearch import __version__ as version
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
            self.basepathget_basepath(calculator=calculator,
                                      ext=basepath_ext)

        self.calculator = calculator
        self.ncpus = ncpus
        self.queue = queue

        self.calc_parameters = calc_parameters

    def submit_calculation(self):
        """Submit calculation for unique structure. 
        First the execution path is set, then the initial POSCAR and models.py
        is written to the directory.

        The calculation is submitted as a parametrized model with trisub.
        """

        self.set_execution_path()
        self.write_poscar(self.excpath)
        self.write_model(self.excpath)

        parameterstr_list = ['{}'.format(param)
                             for param in self.calc_value_list]
        parameterstr = '/' + '/'.join(parameterstr_list)

        command = shlex.split('trisub -p {} -q {} -c {}'.format(
            parameterstr, self.queue, self.ncpus))
        subprocess.call(command, cwd=self.excpath)

    def write_poscar(self, filepath):
        """Write POSCAR to specified file"""
        with open(filepath + '/initial.POSCAR', 'w') as f:
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
        """ Write model.py"""
        symbols = self.atoms.symbols
        Calculator = get_calculator(self.calculator)

        modelstr, self.calc_value_list \
            = Calculator(self.calc_parameters,
                         symbols,
                         self.ncpus).get_model()

        with open(filepath + '/model.py', 'w') as f:
            f.write(modelstr)
