import os
import json
import subprocess
import time
import ase
from ase.io import read
# import bulk_enumerator as be

from protosearch.utils import get_basepath
# from protosearch.build_bulk.build_bulk import BuildBulk
from protosearch.build_bulk.classification import get_classification
from protosearch.utils.standards import VaspStandards
# from protosearch.calculate.submit import TriSubmit
from .prototype_db import PrototypeSQL


class DummyWorkflow(PrototypeSQL):
    """Submits calculations with TriSubmit, and tracks the calculations
    in an ASE db.
    """
    #| - DummyWorkflow
    def __init__(self,
                 calculator='vasp',
                 db_filename=None,
                 basepath_ext=None,
                 job_complete_time=0.8,
                 ):
        """
        Args:
            job_complete_time: Artificial time it takes for a job to complete
        """
        #| - __init__
        print("USING DUMMY WORKFLOW CLASS | NO DFT SUBMISSION")

        self.job_complete_time = job_complete_time

        self.basepath = get_basepath(calculator=calculator,
                                     ext=basepath_ext)
        if not db_filename:
            db_filename = self.basepath + '/prototypes.db'

        super().__init__(filename=db_filename)
        self._connect()
        self.collected = False
        #__|



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

        # Sub = TriSubmit(atoms=atoms,
        #                 ncpus=ncpus,
        #                 calc_parameters=calc_parameters,
        #                 basepath=self.basepath)
        #
        # Sub.submit_calculation()

        key_value_pairs = {
            # 'path': Sub.excpath,
            "submit_time": time.time(),
            'submitted': 1,
            }

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
        # if not self.collected:
        #     self._collect()

        con = self.connection or self._connect()
        self._initialize(con)

        # Only keeps track of jobs in current iteration, resets after each batch
        status_count = {'completed': 0,
                        'running': 0,
                        'errored': 0}

        completed_ids = []
        failed_ids = []
        running_ids = []
        for d in self.ase_db.select(submitted=1, completed=0):
            # path = d.path + '/simulation'
            # calcid0 = d.id

            status, calcid = self.check_job_status(d)

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

    def check_job_status(self, d):
        status = 'running'

        atoms = d.toatoms()
        calcid = d.id

        # print("DISJFIDJ")
        # print(d)
        # print("DISJFIDJ")

        time_submit = d.submit_time
        time_i = time.time()

        time_elapsed = time_i - time_submit

        # print("time_elapsed: ")
        # print(time_elapsed)

        if time_elapsed > self.job_complete_time:
            # Job completed
            status = 'completed'


            import copy
            from protosearch.utils.dummy_calc import DummyCalc

            atoms = copy.deepcopy(atoms)
            calc = DummyCalc(energy_zero=42.0)
            atoms.set_calculator(calc)
            # calc.get_potential_energy()

            calcid = self.save_completed_calculation(atoms, calcid)

        else:
            # Job not completed
            status = 'running'


        return(status, calcid)

        #| - out of sight
        # for root, dirs, files in os.walk(path):
        #     if 'monitor.sh' in files and not 'finalized' in files:
        #         status = 'running'
        #         break
        #     elif 'monitor.sh' in files and 'completed' in files:
        #         # Calculation completed - now save everything
        #         try:  # Sometimes outcar is corrupted
        #             atoms = ase.io.read(root + '/OUTCAR')
        #             status = 'completed'
        #             calcid = self.save_completed_calculation(atoms,
        #                                                      path,
        #                                                      root,
        #                                                      calcid)
        #         except:
        #             print("Couldn't read OUTCAR")
        #             status = 'errored'
        #             self.save_failed_calculation(root, calcid)
        #         break
        #     elif 'monitor.sh' in files and 'err' in files:
        #         status = 'errored'
        #         self.save_failed_calculation(root, calcid)
        #         break
        #
        # return status, calcid
        #__|

    def save_completed_calculation(self,
        atoms,
        # path,
        # runpath,
        calcid,
        ):

        self.ase_db.update(id=calcid,
                           completed=1)

        # param_dict = params2dict(runpath + '/param')
        prototype, cell_parameters = get_classification(atoms)

        key_value_pairs = {'relaxed': 1,
                           'completed': 1,
                           'submitted': 1,
                           'initial_id': calcid,
                           'energy_dummy': 42,
                           # 'path': path,
                           # 'runpath': runpath,
                           }

        key_value_pairs.update(prototype)
        key_value_pairs.update(cell_parameters)
        # key_value_pairs.update(param_dict)

        key_value_pairs = clean_key_value_pairs(key_value_pairs)

        # atoms = set_calculator_info(atoms, param_dict)

        # atoms.energy = 42  # TEMP

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
        # self._collect()
        con = self.connection or self._connect()
        self._initialize(con)

        for d in self.ase_db.select(completed=-1):
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

    #__|


#| - Methods
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
#__|
