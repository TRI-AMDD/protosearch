import os
import shlex
import subprocess
from ase.io import write

from protosearch.utils import get_tri_basepath
from protosearch.build_bulk.classification import PrototypeClassification
from .calculator import get_calculator
from .vasp import get_poscar_from_atoms

def get_submitter(cluster):
    if cluster == 'tri':
        return TriSubmit
    elif cluster == 'nersc':
        return NerscSubmit


class Submit:
    """
    Base class for writing job submission files for crystallographic
    prototype structures
    """

    def __init__(self,
                 atoms,
                 basepath,
                 calc_parameters=None,
                 ):

        self.atoms = atoms

        PC = PrototypeClassification(self.atoms)
        prototype, cell_parameters = PC.get_classification()

        self.spacegroup = prototype['spacegroup']
        self.wyckoffs = prototype['wyckoffs']
        self.species = prototype['species']

        self.cell_param_list = []
        self.cell_value_list = []
        for param in sorted(cell_parameters):
            value = cell_parameters[param]
            if value in self.cell_value_list or value in [90, 120]:
                continue
            self.cell_value_list += [cell_parameters[param]]
            self.cell_param_list += [param]

        self.basepath = basepath
        self.calculator = 'vasp'
        self.master_parameters = calc_parameters

    def write_submission_files(self):
        self.write_poscar(self.excpath)
        self.write_model(self.excpath)
        # self.write_parameters(self.excpath)

    def write_poscar(self, filepath):
        """Write POSCAR to specified file"""
        write(filepath + '/initial.POSCAR', images=self.atoms, format='vasp')

    def set_execution_path(self, strict_format=True):
        """Create a unique submission path for each structure"""

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
                param_str = '{}{}'.format(cell_key, round(cell_value, 2)).\
                    replace('c/a', 'c').replace('b/a', 'b')
                if strict_format:
                    param_str = param_str.replace('.', 'D').replace('-', 'M')
                cell_param_id += param_str

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
                          # self.ncpus
                          )

    def write_model(self, filepath):
        """ Write model.py"""
        import json
        calculator = self.get_calculator()
        modelstr = calculator.get_model()
        parameters = calculator.calc_parameters

        with open(filepath + '/model.py', 'w') as f:
            f.write(modelstr)
        with open(filepath + '/parameters.json', 'w') as f:
            json.dump(parameters, f)

    # def write_parameters(self, filepath):
    #    with open(filepath + '/parameters.json', 'w') as f:
    #        f.write(self.calc_parameters)


class TriSubmit(Submit):
    """
    Set up (VASP) calculations on TRI-AWS for bulk structure enumerated
    with the Bulk prototype enumerator developed by A. Jain, described in:
    A. Jain and T. Bligaard, Phys. Rev. B 98, 214112 (2018)

    Parameters:

    atoms: ASE Atoms object
    calc_parameters: dict
        Optional specification of parameters, such as {ecut: 300}.
        If not specified, the parameter standards given in
        ./utils/standards.py will be applied
    ncpus: int
        number of cpus on AWS to use, default: 1
    queue: 'small', 'medium', etc
        Queue specificatin for AWS
    calculator: str
        Currently only 'vaps' is implemented
    basepath: str or None
        Path for job submission of in TRI filesync (s3) directory
        F.ex: '~/matr.io//model/<calculator>/1/u/<username>

    Alternative to setting basepath, use the following environment valiables
    to set the job submission path automatically:
        TRI_PATH: Your TRI sync directory, which is usually at ~/matr.io
        TRI_USERNAME: Your TRI username
    """

    def __init__(self,
                 atoms,
                 calc_parameters=None,
                 ncpus=None,
                 queue='medium',
                 calculator='vasp',
                 basepath=None,
                 basepath_ext=None
                 ):

        if basepath is None:
            basepath = get_tri_basepath(calculator=calculator,
                                        ext=basepath_ext)

        super().__init__(atoms,
                         basepath,
                         calc_parameters)

        if ncpus is None:
            ncpus = get_ncpus_from_volume(atoms)

        self.ncpus = ncpus
        self.queue = queue

    def submit_calculation(self):
        """Submit calculation for unique structure.
        First the execution path is set, then the initial POSCAR and models.py
        are written to the directory.

        The calculation is submitted as a regular model with trisub.
        """

        self.set_execution_path()
        self.write_submission_files()

        command = shlex.split('trisub -q {} -c {}'.format(
            self.queue, self.ncpus))
        subprocess.call(command, cwd=self.excpath)


class NerscSubmit(Submit):
    """
    WIP!

    Set up VASP calculations on NERSC for bulk structure enumerated
    with the Bulk prototype enumerator developed by A. Jain, described in:
    A. Jain and T. Bligaard, Phys. Rev. B 98, 214112 (2018)

    atoms: ASE Atoms object
    calc_parameters: dict
        Optional specification of parameters, such as {ecut: 300}.
        If not specified, the parameter standards given in
        ./utils/standards.py will be applied
    qos: quality of service SLURM option
       'debug', 'scavenger' or 'premium'
    partition: SLURM partition
       'regular' or 'scavenger'
    time: SLURM job time allocation
    nodes: SLURM node allocation
    ntasks: number of processes per node. 66 on cori knl  
    basepath: str
        Path to your root project directory for running calculations
        defaults to $SCRATCH/protosearch
    """

    def __init__(self,
                 atoms,
                 account,
                 calc_parameters=None,
                 qos='premium',
                 partition='regular',
                 time='1.00.00',
                 nodes=10,
                 ntasks=66,
                 basepath='$SCRATCH/protosearch',
                 ):

        super().__init__(atoms,
                         basepath,
                         calc_parameters)

        self.qos = qos
        self.partition = partition
        self.time = time
        self.account = account
        self.nodes = nodes
        self.ntasks = ntasks

    def submit_calculation(self):
        """Submit calculation for structure
        First the execution path is set, then the initial POSCAR and models.py
        are written to the directory.
        """

        self.set_execution_path(strict_format=False)
        self.write_submission_files()
        self.write_submit_script()

        command = shlex.split('sh submit.sh')
        subprocess.call(command, cwd=self.excpath)

    def write_submit_script(self):
        script = get_nersc_submit_script(self,
                                         self.excpath,
                                         qos=self.qos,
                                         partition=self.partition,
                                         time=self.time,
                                         account=self.account,
                                         nodes=self.nodes,
                                         ntasks=self.ntasks)
        with open(self.excpath + '/submit.sh', 'w') as f:
            f.write(script)


def get_ncpus_from_volume(atoms):
    ncpus = int((atoms.get_volume() / 80 // 4) * 4)
    return max(ncpus, 1)


def get_nersc_submit_script(self,
                            excpath,
                            account,
                            qos='premium',
                            partition='regular',
                            time='1:00:00',
                            nodes=10,
                            ntasks=66):
    """Write SLURM submit script for NERSC. Takes regular SLURM parameters
    qos (debug, scavenger, premium)
    partition (regular, scavenger)
    """

    script = '#!/bin/bash\n'
    script += '#SBATCH -q {}\n'.format(qos)
    script += '#SBATCH -p {}\n'.format(partition)
    script += '#SBATCH -t {}\n'.format(time)
    script += '#SBATCH -A {}\n'.format(account)
    script += '#SBATCH -e error.txt\n'
    script += '#SBATCH -o output.txt\n'
    script += '#SBATCH -C knl\n'
    script += '#SBATCH --ntasks-per-node={}\n'.format(ntasks)

    script += """
module load python/3.7-anaconda-2019.07
export PYTHONPATH=$HOME/.local/cori/3.7-anaconda-2019.07/lib/python3.7/site-packages:$PYTHONPATH
module load vasp-tpc/5.4.4-knl

#cd {excpath}
export VASP_SCRIPT=./run_vasp.py
export VASP_PP_PATH=/project/projectdirs/{account}/vasp-psp/pseudo52

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
""".format(excpath=excpath, account=account)

    script += """
echo "import os" > run_vasp.py
echo "exitcode = os.system('srun -n {nproc} -c 4 --cpu_bind=cores vasp_std')" >> run_vasp.py

echo $VASP_SCRIPT
echo $VASP_PP_PATH

python {excpath}/model.py""".format(nproc=nodes * ntasks, excpath=excpath)

    return script
