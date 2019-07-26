import os
import json
import subprocess
import time
import numpy as np
import copy
import ase
from ase.io import read

from protosearch.build_bulk.build_bulk import BuildBulk
from protosearch.build_bulk.classification import get_classification

from protosearch.utils import get_basepath
from protosearch.utils.standards import VaspStandards
from protosearch.calculate.dummy_calc import DummyCalc
from protosearch.calculate.submit import TriSubmit
from .prototype_db import PrototypeSQL


class Workflow(PrototypeSQL):
    """Submits calculations with TriSubmit, and tracks the calculations
    in an ASE db.
    """

    def __init__(self,
                 calculator='vasp',
                 db_filename=None,
                 basepath_ext=None):

        self.basepath = get_basepath(calculator=calculator,
                                     ext=basepath_ext)
        if not db_filename:
            db_filename = self.basepath + '/prototypes.db'

        super().__init__(filename=db_filename)
        self._connect()
        self.collected = False

    def _collect(self):
        if self.collected:
            return
        subprocess.call('trisync', cwd=self.basepath)
        self.collected = True

    def recollect(self):
        subprocess.call('trisync', cwd=self.basepath)
        self.collected = True

    def submit_atoms_batch(self, atoms_list, ncpus=1, calc_parameters=None):
        """Submit a batch of calculations. Takes a list of atoms
        objects as input"""
        batch_no = self.get_next_batch_no()
        print('Batch no {}'.format(batch_no))
        for atoms in atoms_list:
            self.submit_atoms(atoms, ncpus, batch_no, calc_parameters)

    def submit_atoms(self, atoms, ncpus=1, batch_no=None, calc_parameters=None):
        """Submit a calculation for an atoms object"""
        print('SUBMIT!')
        prototype, parameters = get_classification(atoms)

        Sub = TriSubmit(atoms=atoms,
                        ncpus=ncpus,
                        calc_parameters=calc_parameters,
                        basepath=self.basepath)

        Sub.submit_calculation()

        key_value_pairs = {'p_name': prototype['p_name'],
                           'path': Sub.excpath,
                           'spacegroup': prototype['spacegroup'],
                           'wyckoffs': json.dumps(prototype['wyckoffs']),
                           'species': json.dumps(prototype['species'])}
        if batch_no:
            key_value_pairs.update({'batch': batch_no})

        self.write_submission(key_value_pairs)

    def submit_batch(self, prototypes, ncpus=1, calc_parameters=None):
        """Submit a batch of calculations. Takes a list of prototype
        dicts as input"""
        batch_no = self.get_next_batch_no()
        for prototype in prototypes:
            self.submit(prototype, ncpus, batch_no, calc_parameters)

    def submit(self, prototype, ncpus=1, batch_no=None, calc_parameters=None):
        """Submit a calculation for a prototype, generating atoms
        with build_bulk and enumerator"""
        cell_parameters = prototype.get('parameters', None)

        BB = BuildBulk(prototype['spacegroup'],
                       prototype['wyckoffs'],
                       prototype['species'],
                       cell_parameters=cell_parameters
                       )

        atoms = BB.get_atoms_from_poscar()
        p_name = BB.get_prototype_name()
        formula = atoms.get_chemical_formula()

        if self.is_calculated(formula=formula,
                              p_name=p_name):
            return

        Sub = TriSubmit(atoms=atoms,
                        ncpus=ncpus,
                        calc_parameters=calc_parameters,
                        basepath=self.basepath)

        Sub.submit_calculation()

        key_value_pairs = {'p_name': BB.prototype_name,
                           'path': Sub.excpath,
                           'spacegroup': BB.spacegroup,
                           'wyckoffs': json.dumps(BB.wyckoffs),
                           'species': json.dumps(BB.species)}

        if batch_no:
            key_value_pairs.update({'batch': batch_no})

        self.write_submission(key_value_pairs)

    def submit_enumerated(self, map_species, selection={}):
        """
        Submit a collection of enumeated prototypes.
        Parameters:
        ----------
        map_species: dict
          map A0, A1, etc to species. F.ex: {'A0': 'Fe', 'A1': 'O'}
        selection: dict
          ase db selection. F.ex: {'spacegroup': 166, natom=3}
        """

        prototypes = self.select(**selection)

        for prototype in prototypes:
            self.submit(prototype)

    def submit_id_batch(self, calc_ids, ncpus=1, calc_parameters=None):
        """Submit a batch of calculations. Takes a list of atoms
        objects db ids as input"""
        batch_no = self.get_next_batch_no()
        for calc_id in calc_ids:
            self.submit_id(calc_id, ncpus, batch_no, calc_parameters)

    def submit_id(self, calc_id, ncpus=1, batch_no=None, calc_parameters=None):
        """
        Submit an atomic structure by id
        """
        row = self.ase_db.get(id=int(calc_id))
        atoms = row.toatoms()

        formula = row.formula
        p_name = row.p_name

        if self.is_calculated(formula=formula,
                              p_name=p_name):
            return

        Sub = TriSubmit(atoms=atoms,
                        ncpus=ncpus,
                        calc_parameters=calc_parameters,
                        basepath=self.basepath)

        Sub.submit_calculation()

        key_value_pairs = {'path': Sub.excpath,
                           'submitted': 1}

        if batch_no is not None:
            key_value_pairs.update({'batch': batch_no})

        self.ase_db.update(int(calc_id), **key_value_pairs)

    def write_submission(self, key_value_pairs):
        """Track submitted job in database"""
        con = self.connection or self._connect()
        self._initialize(con)

        atoms = read(key_value_pairs['path'] + '/initial.POSCAR')

        key_value_pairs.update({'relaxed': 0,
                                'completed': 0,
                                'submitted': 1})

        self.ase_db.write(atoms, key_value_pairs)

    def check_submissions(self):
        """Check for completed jobs"""
        if not self.collected:
            self._collect()
        con = self.connection or self._connect()
        self._initialize(con)

        status_count = {'completed': 0,
                        'running': 0,
                        'errored': 0}
        completed_ids = []
        failed_ids = []
        running_ids = []
        for d in self.ase_db.select(submitted=1, completed=0):
            path = d.path + '/simulation'
            calcid0 = d.id
            status, calcid = self.check_job_status(path, calcid0)
            status_count[status] += 1
            if status == 'completed':
                completed_ids += [calcid]
            elif status == 'errored':
                failed_ids += [calcid]
            elif status == 'running':
                running_ids += [calcid]

        print('Status for calculations:')
        for status, value in status_count.items():
            print('  {} {}'.format(value, status))
        return completed_ids, failed_ids, running_ids

    def check_job_status(self, path, calcid):
        status = 'running'
        for root, dirs, files in os.walk(path):
            if 'monitor.sh' in files and not 'finalized' in files:
                status = 'running'
                break
            elif 'monitor.sh' in files and 'completed' in files:
                # Calculation completed - now save everything
                try:  # Sometimes outcar is corrupted
                    atoms = ase.io.read(root + '/OUTCAR')
                    status = 'completed'
                    calcid = self.save_completed_calculation(atoms,
                                                             path,
                                                             root,
                                                             calcid)
                except:
                    print("Couldn't read OUTCAR")
                    status = 'errored'
                    self.save_failed_calculation(root, calcid)
                break
            elif 'monitor.sh' in files and 'err' in files:
                status = 'errored'
                self.save_failed_calculation(root, calcid)
                break

        return status, calcid

    def save_completed_calculation(self, atoms, path, runpath, calcid,
                                   read_params=True,
                                   ):

        self.ase_db.update(id=calcid,
                           completed=1)

        batch_no = self.ase_db.get(id=calcid).get('batch', None)
        prototype, cell_parameters = get_classification(atoms)

        key_value_pairs = {'relaxed': 1,
                           'completed': 1,
                           'submitted': 1,
                           'initial_id': calcid,
                           'path': path,
                           'runpath': runpath}
        if batch_no:
            key_value_pairs.update({'batch': batch_no})

        key_value_pairs.update(prototype)
        key_value_pairs.update(cell_parameters)

        param_dict = {}
        if read_params:
            param_dict = params2dict(runpath + '/param')
            key_value_pairs.update(param_dict)

        atoms = set_calculator_info(atoms, param_dict)

        key_value_pairs = clean_key_value_pairs(key_value_pairs)

        newcalcid = self.ase_db.write(atoms, key_value_pairs)
        self.ase_db.update(id=calcid,
                           final_id=newcalcid,
                           completed=1)

        return newcalcid

    def save_failed_calculation(self, runpath, calcid):

        key_value_pairs = {'completed': -1,
                           'runpath': runpath}
        param_dict = params2dict(runpath + '/param')
        atoms = ase.io.read(runpath + '/initial.POSCAR')
        prototype, cell_parameters = get_classification(atoms)

        key_value_pairs.update(prototype)
        key_value_pairs.update(cell_parameters)
        key_value_pairs.update(param_dict)

        key_value_pairs = clean_key_value_pairs(
            key_value_pairs)

        with open(runpath + '/err', 'r') as errorf:
            message = errorf.read().replace("'", '')

            self.ase_db.update(id=calcid,
                               **key_value_pairs,
                               data={'error': message})

    def rerun_failed_calculations(self):
        self._collect()
        con = self.connection or self._connect()
        self._initialize(con)

        for d in self.ase_db.select(completed=-1):
            # 'error' is undefined
            resubmit, handle = vasp_errors(error)
            if not resubmit:
                print('Job errored: {}'.format(handle))
                self.ase_db.update(id=d.id, completed=-1)
                continue
            if handle == 'ncpus':
                ncpus = 8
            else:
                ncpus = 1

            p_name = d.p_name
            path = d.path + '/simulation'

            calc_parameters = {}
            for param in VaspStandards.sorted_calc_parameters:
                if d.get(param, None):
                    calc_parameters.update({param: d.get(param)})

            self.submit({'spacegroup': d.spacegroup,
                         'wyckoffs': json.loads(d.wyckoffs),
                         'species': json.loads(d.species)},
                        ncpus=ncpus,
                        calc_parameters=calc_parameters)

            # Treat as completed, now it's resubmitted
            self.ase_db.update(id=d.id, completed=1)

    def get_completed_batch(self):
        ids = self.check_submissions()

    def reclassify(self, tolerance=1e-2):
        """Reclassify prototypes of all completed structures"""
        self._collect()
        con = self.connection or self._connect()
        self._initialize(con)

        for row in self.ase_db.select(relaxed=1):
            prototype, cell_parameters = \
                get_classification(row.toatoms())
            prototype.update(cell_parameters)
            prototype = clean_key_value_pairs(prototype)
            self.ase_db.update(row.id, **prototype)


class AWSWorkflow(Workflow):
    """
    """

    def __init__(self,
                 *args, **kwargs,
                 ):

        print("USING DUMMY WORKFLOW CLASS | NO DFT SUBMISSION")

        super().__init__(*args, **kwargs)


class DummyWorkflow(Workflow):
    """
    """

    def __init__(self,
                 job_complete_time=0.6,
                 *args, **kwargs,
                 ):
        print("USING DUMMY WORKFLOW CLASS | NO DFT SUBMISSION")

        super().__init__(*args, **kwargs)

        self.job_complete_time = job_complete_time

        # Pretend that instance is always in "collected" state
        self.collected = True

        # Will be dotted with fingerprints to produce dummy 'energy' output
        np.random.seed(0)
        self.random_vect = np.random.rand(1, 1000)[0]

    def recollect(self):
        # print("Not actually calling trisync here")
        return(None)

    def submit_id(self, calc_id, ncpus=1, batch_no=None, calc_parameters=None):
        """
        Submit an atomic structure by id
        """
        row = self.ase_db.get(id=int(calc_id))
        # atoms = row.toatoms()

        formula = row.formula
        p_name = row.p_name

        if self.is_calculated(formula=formula, p_name=p_name):
            return

        key_value_pairs = {
            "path": "TEMP",
            "submit_time": time.time(),
            "submitted": 1,
        }

        if batch_no is not None:
            key_value_pairs.update({'batch': batch_no})

        self.ase_db.update(int(calc_id), **key_value_pairs)

    def check_job_status(self, path, calcid):
        """
        """
        d = self.ase_db.get(id=calcid)

        status = 'running'
        atoms = d.toatoms()
        calcid = d.id

        time_submit = d.submit_time
        time_i = time.time()
        time_elapsed = time_i - time_submit

        if time_elapsed > self.job_complete_time:
            # Job completed
            status = 'completed'

            train_features, train_target = self.get_fingerprints([calcid])
            shape = train_features.shape
            random_vect = self.random_vect[0:shape[-1]]
            dummy_energy = random_vect.dot(train_features[0]) / 1000

            atoms = copy.deepcopy(atoms)
            calc = DummyCalc(energy_zero=dummy_energy)
            atoms.set_calculator(calc)

            path = d.path
            runpath = "TEMP"

            new_calcid = self.save_completed_calculation(
                atoms, path, runpath, calcid,
                read_params=False,
                # atoms, calcid,
            )
            calcid = new_calcid

        else:
            # Job not completed
            status = 'running'

        return(status, calcid)


def clean_key_value_pairs(key_value_pairs):
    for key, value in key_value_pairs.items():
        if isinstance(value, list):
            key_value_pairs[key] = json.dumps(value)
        try:
            value = int(value)
            key_value_pairs[key] = value
        except:
            pass
    return key_value_pairs


def set_calculator_info(atoms, parameters):

    atoms.calc.name = 'vasp.5.4.4.18Apr17-6-g9f103f2a35'
    atoms.calc.parameters = parameters

    return atoms


def vasp_errors(error):

    if 'lattice_constant = float(f.readline().split()[0])' in error:
        return False, 'Vasp failed'
    elif 'Vasp exited' in error:
        return True, 'ncpus'


def params2dict(paramfile):
    param_dict = {}
    with open(paramfile, 'r') as f:
        param = f.read().lstrip('/').rstrip('\n')
        param_values = param.split('/')
        for i, param_key in enumerate(VaspStandards.sorted_calc_parameters):
            param_dict[param_key] = param_values[i]

    return param_dict
