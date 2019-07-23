import json
import time
import copy
import numpy as np
from protosearch.utils import get_basepath
from protosearch.build_bulk.classification import get_classification
from .prototype_db import PrototypeSQL
from protosearch.utils.dummy_calc import DummyCalc

# from protosearch.utils.standards import VaspStandards


class DummyWorkflow(PrototypeSQL):
    """Submits calculations with TriSubmit, and tracks the calculations
    in an ASE db.
    """
    #| - DummyWorkflow
    def __init__(self,
                 calculator='vasp',
                 db_filename=None,
                 basepath_ext=None,
                 job_complete_time=0.6,
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

        # Will be dotted with fingerprints to produce dummy 'energy' output
        np.random.seed(0)
        self.random_vect = np.random.rand(1, 1000)[0]
        #__|


    def recollect(self):
        print("Not actually calling trisync here")

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
        # atoms = row.toatoms()

        formula = row.formula
        p_name = row.p_name

        if self.is_calculated(formula=formula,
                              p_name=p_name):
            return

        key_value_pairs = {
            "submit_time": time.time(),
            'submitted': 1,
            }

        if batch_no is not None:
            key_value_pairs.update({'batch': batch_no})

        self.ase_db.update(int(calc_id), **key_value_pairs)

    def check_submissions(self):
        """Check for completed jobs"""

        con = self.connection or self._connect()
        self._initialize(con)

        # Only keeps track of jobs in current iteration
        # resets after each batch
        status_count = {'completed': 0,
                        'running': 0,
                        'errored': 0}

        completed_ids = []
        failed_ids = []
        running_ids = []
        for d in self.ase_db.select(submitted=1, completed=0):
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
        print(calcid)  # TEMP

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

            new_calcid = self.save_completed_calculation(atoms, calcid)
            calcid = new_calcid

        else:
            # Job not completed
            status = 'running'


        return(status, calcid)

    def save_completed_calculation(self, atoms, calcid):

        self.ase_db.update(id=calcid,
                           completed=1)

        prototype, cell_parameters = get_classification(atoms)

        key_value_pairs = {'relaxed': 1,
                           'completed': 1,
                           'submitted': 1,
                           'initial_id': calcid,
                           }

        key_value_pairs.update(prototype)
        key_value_pairs.update(cell_parameters)

        key_value_pairs = clean_key_value_pairs(key_value_pairs)

        newcalcid = self.ase_db.write(atoms, key_value_pairs)
        self.ase_db.update(id=calcid,
                           final_id=newcalcid,
                           completed=1)

        return newcalcid
    #__|


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
