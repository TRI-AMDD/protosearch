from protosearch.workflow.workflow_master import WorkflowMaster
from protosearch.workflow.workflow import Workflow

import json
import time
import copy
import numpy as np
from protosearch.utils import get_basepath
from protosearch.build_bulk.classification import get_classification
from .prototype_db import PrototypeSQL
from protosearch.utils.dummy_calc import DummyCalc
from protosearch.workflow.workflow import Workflow
# from protosearch.utils.standards import VaspStandards

class DummyWorkflow_TEST(Workflow):
    """
    """

    #| - DummyWorkflow
    def __init__(self,
        job_complete_time=0.6,
        *args, **kwargs,
        ):
        #| - __init__
        print("USING DUMMY WORKFLOW CLASS | NO DFT SUBMISSION")

        super().__init__(*args, **kwargs)

        self.job_complete_time = job_complete_time

        # Pretend that instance is always in "collected" state
        self.collected = True

        # Will be dotted with fingerprints to produce dummy 'energy' output
        np.random.seed(0)
        self.random_vect = np.random.rand(1, 1000)[0]
        #__|

    def recollect(self):
        print("Not actually calling trisync here")

    def submit_id(self, calc_id, ncpus=1, batch_no=None, calc_parameters=None):
        """
        Submit an atomic structure by id
        """
        #| - submit_id
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
        #__|

    def check_job_status(self, path, calcid):
        """
        """
        #| - check_job_status
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
        #__|

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


#| - __old__
    # def submit_id_batch(self, calc_ids, ncpus=1, calc_parameters=None):
    #     """Submit a batch of calculations. Takes a list of atoms
    #     objects db ids as input"""
    #     #| - submit_id_batch
    #     batch_no = self.get_next_batch_no()
    #     for calc_id in calc_ids:
    #         self.submit_id(calc_id, ncpus, batch_no, calc_parameters)
    #     #__|

    # def save_completed_calculation(self, atoms, calcid):
    #     """
    #     """
    #     #| - save_completed_calculation
    #     self.ase_db.update(id=calcid,
    #                        completed=1)
    #
    #     prototype, cell_parameters = get_classification(atoms)
    #
    #     key_value_pairs = {'relaxed': 1,
    #                        'completed': 1,
    #                        'submitted': 1,
    #                        'initial_id': calcid,
    #                        }
    #
    #     key_value_pairs.update(prototype)
    #     key_value_pairs.update(cell_parameters)
    #
    #     key_value_pairs = clean_key_value_pairs(key_value_pairs)
    #
    #     newcalcid = self.ase_db.write(atoms, key_value_pairs)
    #     self.ase_db.update(id=calcid,
    #                        final_id=newcalcid,
    #                        completed=1)
    #
    #     return newcalcid
    #     #__|



#__|
