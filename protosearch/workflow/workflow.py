import os
import json
import subprocess
import time
import numpy as np
import copy
import ase
import ase.build
from ase.io import read

from protosearch.build_bulk.build_bulk import BuildBulk
from protosearch.build_bulk.classification import PrototypeClassification

from protosearch.utils.standards import VaspStandards, CrystalStandards
from protosearch.calculate.dummy_calc import DummyCalc
from protosearch.calculate.submit import get_submitter
from .prototype_db import PrototypeSQL

standard_lattice = CrystalStandards.standard_lattice_mp
all_elements = list(standard_lattice.keys()) + ['H', 'N', 'O']


class Workflow(PrototypeSQL):
    """Submits calculations on specified cluster, and tracks the calculations
    in an ASE db.
    Parameters:

    db_filename: str
       full path to db file to save calculations
    cluster: str
       Currently only 'tri' is working
    """
    def __init__(self,
                 db_filename,
                 calculator='vasp',
                 cluster='tri',
                 verbose=False):

        self.verbose = verbose

        super().__init__(filename=db_filename)

        self.Submitter = get_submitter(cluster)

        self.collected = False

    def _collect(self):
        if self.collected:
            return
        self.recollect()

    def recollect(self):
        for row in self.ase_db.select(completed=0, submitted=1):
            subprocess.call('trisync', cwd=row.path)

        self.collected = True

    def submit_atoms_batch(self, atoms_list, ncpus=None, calc_parameters=None,
                           **kwargs):
        """Submit a batch of calculations. Takes a list of atoms
        objects as input"""
        batch_no = self.get_next_batch_no()
        for atoms in atoms_list:
            self.submit_atoms(atoms, ncpus, batch_no,
                              calc_parameters, **kwargs)

    def submit_atoms(self, atoms, ncpus=None, batch_no=None,
                     calc_parameters=None, **kwargs):
        """Submit a calculation for an atoms object"""
        PC = PrototypeClassification(atoms)
        prototype, cell_parameters = PC.get_classification()

        if self.is_calculated(formula=atoms.get_chemical_formula(),
                              p_name=prototype['p_name']):
            return

        Sub = self.Submitter(atoms=atoms,
                             ncpus=ncpus,
                             calc_parameters=calc_parameters)

        Sub.submit_calculation()

        key_value_pairs = {'p_name': prototype['p_name'],
                           'path': Sub.excpath,
                           'spacegroup': prototype['spacegroup'],
                           'wyckoffs': json.dumps(prototype['wyckoffs']),
                           'species': json.dumps(prototype['species']),
                           'ncpus': Sub.ncpus}

        key_value_pairs.update(kwargs)

        if batch_no:
            key_value_pairs.update({'batch': batch_no})

        self.write_submission(key_value_pairs)

    def submit_batch(self, prototypes, ncpus=None,
                     calc_parameters=None, **kwargs):
        """Submit a batch of calculations. Takes a list of prototype
        dicts as input"""
        batch_no = self.get_next_batch_no()
        for prototype in prototypes:
            self.submit(prototype, ncpus, batch_no, calc_parameters, **kwargs)
        self.write_status(last_batch_no=batch_no)

    def submit(self, prototype, ncpus=None, batch_no=None, calc_parameters=None,
               **kwargs):
        """Submit a calculation for a prototype, generating atoms
        with build_bulk and enumerator"""
        cell_parameters = prototype.get('parameters', None)

        BB = BuildBulk(prototype['spacegroup'],
                       prototype['wyckoffs'],
                       prototype['species'],
                       )

        atoms = BB.get_atoms(cell_parameters=cell_parameters)
        p_name = BB.get_prototype_name(prototype['species'])
        formula = atoms.get_chemical_formula()

        Sub = self.Submitter(atoms=atoms,
                             ncpus=ncpus,
                             calc_parameters=calc_parameters)

        Sub.submit_calculation()

        key_value_pairs = {'p_name': p_name,
                           'path': Sub.excpath,
                           'spacegroup': BB.spacegroup,
                           'wyckoffs': json.dumps(BB.wyckoffs),
                           'species': json.dumps(BB.species),
                           'ncpus': Sub.ncpus}

        key_value_pairs.update(kwargs)

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

    def submit_id_batch(self, calc_ids, ncpus=None,
                        calc_parameters=None, **kwargs):
        """Submit a batch of calculations. Takes a list of atoms
        objects db ids as input"""
        batch_no = self.get_next_batch_no()
        for calc_id in calc_ids:
            self.submit_id(calc_id, ncpus, batch_no, calc_parameters)
        self.write_status(last_batch_no=batch_no)

    def submit_id(self, calc_id, ncpus=None, batch_no=None,
                  calc_parameters=None, **kwargs):
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

        Sub = self.Submitter(atoms=atoms,
                             ncpus=ncpus,
                             calc_parameters=calc_parameters)
        Sub.submit_calculation()

        key_value_pairs = {'path': Sub.excpath,
                           'submitted': 1,
                           'ncpus': Sub.ncpus}

        if batch_no is not None:
            key_value_pairs.update({'batch': batch_no})

        key_value_pairs.update(kwargs)

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

    def check_submissions(self, **kwargs):
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
        for d in self.ase_db.select(submitted=1, completed=0, **kwargs):
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

        if self.verbose:
            print('Status for calculations:')
            for status, value in status_count.items():
                print('  {} {}'.format(value, status))
        return completed_ids, failed_ids, running_ids

    def check_job_status(self, path, calcid):
        status = 'running'
        for root, dirs, files in os.walk(path):
            if 'monitor.sh' in files and not 'finalized' in files:
                status = 'running'
                self.ase_db.update(id=calcid, runpath=root)
                break
            elif 'monitor.sh' in files and 'completed' in files:
                # Calculation completed - now save everything
                try:  # Sometimes outcar is corrupted
                    atoms = ase.io.read(root + '/OUTCAR')
                    status = 'completed'
                    calcid = self.save_completed_calculation(atoms,
                                                             root,
                                                             calcid)
                except BaseException as e:
                    print("Couldn't read OUTCAR", calcid)
                    status = 'errored'
                    self.save_failed_calculation(root, calcid)
                break
            elif 'monitor.sh' in files and 'finalized' in files:
                status = 'errored'
                self.save_failed_calculation(root, calcid)
                break

        return status, calcid

    def save_completed_calculation(self, atoms, runpath, calcid,
                                   read_params=True):

        batch_no = self.ase_db.get(id=calcid).get('batch', None)

        PC = PrototypeClassification(atoms)
        prototype, cell_parameters = PC.get_classification()

        key_value_pairs = self.ase_db.get(id=calcid).get('key_value_pairs', {})

        key_value_pairs.update({'relaxed': 1,
                                'completed': 1,
                                'initial_id': calcid,
                                'runpath': runpath})
        key_value_pairs.update(prototype)
        key_value_pairs.update(cell_parameters)

        if batch_no:
            key_value_pairs.update({'batch': batch_no})

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

        PC = PrototypeClassification(atoms)
        prototype, cell_parameters = PC.get_classification()

        key_value_pairs.update(prototype)
        key_value_pairs.update(cell_parameters)
        key_value_pairs.update(param_dict)

        key_value_pairs = clean_key_value_pairs(
            key_value_pairs)
        data = {}
        if os.path.isfile(runpath + '/err'):
            with open(runpath + '/err', 'r') as errorf:
                data.update({'error': errorf.read().replace("'", '')})

        if os.path.isfile(runpath + '/err.relax'):
            with open(runpath + '/err.relax', 'r') as errorf:
                data.update({'error_relax': errorf.read().replace("'", '')})

        self.ase_db.update(id=calcid,
                           **key_value_pairs,
                           data=data)

    def rerun_failed_calculations(self):
        self._collect()
        con = self.connection or self._connect()
        self._initialize(con)

        for d in self.ase_db.select(completed=-1):
            error = d.data.get('error', None)
            failed_relax = False
            if error is None:
                failed_relax = True
                error = d.data.get('error_relax', None)

            if not error:
                continue
            resubmit, fail_reason = vasp_errors(error)
            if failed_relax:
                resubmit = True
            if not resubmit:
                if fail_reason == 'ase read':
                    atoms = ase.io.read(d.runpath + '/OUTCAR')
                    calcid = self.save_completed_calculation(atoms,
                                                             d.runpath,
                                                             d.id)
                else:
                    print('Job not resubmitted: {}'.format(fail_reason))
                continue
            ncpus = None
            if fail_reason == 'ncpus':
                ncpus = d.get('ncpus', 1) * 2
            elif fail_reason == 'ase read':
                atoms = read(d.runpath + '/OUTCAR.relax')
                self.ase_db.update(id=d.id, atoms=atoms)
            p_name = d.p_name
            path = d.path + '/simulation'

            calc_parameters = {}
            for param in VaspStandards.sorted_calc_parameters:
                if d.get(param, None):
                    calc_parameters.update({param: d.get(param)})
            if fail_reason == 'Symmetry error':
                calc_parameters.update({'kgamma': True})
            self.submit_id(d.id,
                           ncpus=ncpus,
                           calc_parameters=calc_parameters)

            rerun = (d.get('rerun') or 0) + 1
            self.ase_db.update(id=d.id, completed=0, rerun=rerun)

    def get_completed_batch(self):
        ids = self.check_submissions()

    def reclassify(self, tolerance=0.5e-3):
        """Reclassify prototypes of all completed structures"""
        self._collect()
        con = self.connection or self._connect()
        self._initialize(con)

        for row in self.ase_db.select(relaxed=1):

            PC = PrototypeClassification(row.toatoms())
            prototype, cell_parameters = PC.get_classification()
            prototype.update(cell_parameters)
            prototype = clean_key_value_pairs(prototype)
            self.ase_db.update(row.id, **prototype)

    def submit_standard_states(self, elements=None, batch_no=None):
        elements = elements or all_elements
        for e in elements:
            if e in ['H', 'N', 'O', 'F', 'Cl']:
                atoms = ase.build.molecule(e + '2')
                atoms.set_cell(10 * np.identity(3))
                atoms.center()
                calc_parameters = VaspStandards.molecule_calc_parameters
                self.submit_atoms(atoms, batch_no=batch_no,
                                  calc_parameters=calc_parameters,
                                  standard_state=1)
            else:
                prototype = standard_lattice[e]
                self.submit(prototype, batch_no=batch_no, standard_state=1)


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

    def submit_id(self, calc_id, ncpus=None,
                  batch_no=None, calc_parameters=None):
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
                read_params=False)
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
    elif 'Vasp exited with exit code:' in error:
        return True, 'ncpus'
    elif "local variable 'forces' referenced before assignment" in error:
        return True, 'Symmetry error'
    elif 'could not convert string to float' in error:
        return True, 'ase read'
    elif 'local variable forces referenced before assignment' in error:
        return True, 'ase read'
    elif 'invalid literal for float()' in error:
        return True, 'ase read'
    else:
        return False, ''


def params2dict(paramfile):
    param_dict = {}
    with open(paramfile, 'r') as f:
        param = f.read().lstrip('/').rstrip('\n')
        param_values = param.split('/')
        for i, param_key in enumerate(VaspStandards.sorted_calc_parameters):
            param_dict[param_key] = param_values[i]

    return param_dict
