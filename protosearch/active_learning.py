import time
import numpy as np
import pandas as pd

from protosearch.build_bulk.enumeration import (
    Enumeration, AtomsEnumeration, get_stoich_from_formulas)
from protosearch.workflow.prototype_db import PrototypeSQL
from protosearch.workflow.workflow import Workflow as WORKFLOW
from protosearch.ml_modelling.fingerprint import FingerPrint, clean_features
from protosearch.ml_modelling.regression_model import get_regression_model


class ActiveLearningLoop:

    def __init__(self,
                 chemical_formulas,
                 Workflow=WORKFLOW,
                 source='prototypes',
                 batch_size=10,
                 max_atoms=None,
                 check_frequency=60.,
                 frac_jobs_limit=0.7,
                 stop_mode="job_fraction_limit",
                 ):
        """
        Module to run active learning loop

        Parameters:
        ----------
        Workflow:
            Workflow class to use
        chemical_formulas: list of strings
            chemical formulas to investigate, such as:
            ['IrO2', 'IrO3']
        source: str
            Specify how to generate the structures. Options are:
            'oqmd_icsd': Experimental OQMD entries (not implemented)
            'oqmd_all': All OQMD structures (not implemented)
            'prototypes': Enumerate all prototypes.
        batch_size: int
            number of DFT jobs to submit simultaneously.
        check_frequency: float
            Frequency in (s) that the AL checks on the job state
        frac_jobs_limit: float
            Upper limit on the fraction of of jobs to be processed before
            stoping the loop
        stop_mode: str
            Method by which the stop criteria is defined for the ALL
            'job_fraction_limit'
        max_atoms: int
            Max number of atoms to allow in the unit cell
        """
        if isinstance(chemical_formulas, str):
            chemical_formulas = [chemical_formulas]
        self.chemical_formulas = chemical_formulas
        self.source = source
        self.batch_size = batch_size
        self.max_atoms = max_atoms
        self.check_frequency = check_frequency
        self.frac_jobs_limit = frac_jobs_limit
        self.stop_mode = stop_mode

        self.db_filename = '_'.join(chemical_formulas) + '.db'
        self.Workflow = Workflow(db_filename=self.db_filename)
        self.DB = PrototypeSQL(self.db_filename)
        self.DB.write_status(
            chemical_formulas=chemical_formulas, batch_size=batch_size)

    def run(self):
        """Run actice learning loop"""

        WF = self.Workflow
        self.status = self.DB.get_status()

        if not self.status['initialized']:
            self.batch_no = 1
            self.initialize()
            WF.submit_id_batch(self.batch_ids)
            self.DB.write_status(last_batch_no=self.batch_no)
            self.DB.write_status(initialized=1)
        else:
            self.batch_no = self.status['last_batch_no']

        happy = False
        while not happy:
            # get completed calculations from WorkFlow
            completed_ids, failed_ids, running_ids = WF.check_submissions()
            t0 = time.time()
            while len(running_ids) > self.batch_size // 2:
                WF.recollect()
                completed_ids0, failed_ids0, running_ids = WF.check_submissions()
                completed_ids += completed_ids0
                failed_ids += failed_ids0
                t = time.time() - t0
                print('{} job(s) completed in {:.2f} min'.format(len(completed_ids),
                                                                 t / 60))
                time.sleep(self.check_frequency)

            self.DB.write_job_status()

            # Make sure the number or running jobs doesn't blow up
            self.corrected_batch_size = self.batch_size - len(running_ids)

            # get formation energy of completed jobs and save to db
            self.save_formation_energies()

            # save regenerated fingerprints to db (changes after relaxation)
            self.generate_fingerprints(completed=True)

            happy = self.evaluate()

            # retrain ML model
            self.train_ids = self.DB.get_completed_structure_ids()
            self.test_ids = self.DB.get_completed_structure_ids(completed=0)

            self.train_ml()
            all_ids = self.test_ids + self.train_ids
            all_energies = list(self.energies) + list(self.targets)
            all_uncertainties = list(self.uncertainties) + \
                list(np.zeros_like(self.targets))
            self.DB.write_predictions(
                self.batch_no, all_ids, all_energies, all_uncertainties)
            # acqusition -> select next batch
            self.acquire_batch()

            self.batch_no += 1
            # submit structures with WorkFLow
            WF.submit_id_batch(self.batch_ids)
            self.DB.write_status(last_batch_no=self.batch_no)

    def test_run(self):
        """Use already completed calculations to test the loop """
        WF = self.Workflow
        self.status = self.DB.get_status()
        self.corrected_batch_size = self.batch_size

        self.batch_no = 1
        self.batch_ids = []
        self.test_ids = self.DB.get_completed_structure_ids()
        self.train_ids = self.test_ids[:self.batch_size]
        self.test_ids = [
            t for t in self.test_ids if not t in self.train_ids]

        while len(self.test_ids) > 0:
            self.train_ml()
            self.acquire_batch()

            self.train_ids += self.batch_ids
            self.test_ids = [
                t for t in self.test_ids if not t in self.train_ids]

            all_ids = self.test_ids + self.train_ids
            all_energies = np.array(
                list(self.energies) + list(self.targets))
            all_uncertainties = np.array(list(self.uncertainties) +
                                         list(np.zeros_like(self.targets)))

            prediction = {'batch_no': self.batch_no,
                          'ids': all_ids,
                          'energies': all_energies,
                          'vars': all_uncertainties}

            self.batch_no += 1
            yield prediction

    def initialize(self):
        if not self.status['enumerated']:
            self.enumerate_structures()
            self.DB.write_status(enumerated=1)
        if not self.status['fingerprinted']:
            self.generate_fingerprints()
            self.DB.write_status(fingerprinted=1)

        # Run standard states?

        self.acquire_batch(method='random', batch_size=self.batch_size)

    def get_frac_of_systems_processed(self):
        """
        Get the fraction of structures that have been processed, including
        completed as well as errored jobs.
        """
        n_unrelaxed = self.DB.ase_db.count(relaxed=0)
        n_errored = self.DB.ase_db.count(relaxed=0, completed=-1)
        n_relaxed = self.DB.ase_db.count(relaxed=1)

        frac = (n_relaxed + n_errored) / n_unrelaxed

        return frac

    def evaluate(self):
        happy = False

        # Upper limit on Systems/Jobs processed
        if self.stop_mode == "job_fraction_limit":
            frac_i = self.get_frac_of_systems_processed()
            print("fraction_i: ", frac_i)
            if frac_i >= self.frac_jobs_limit:
                print("HAPPY! Upper fraction of jobs procesed limit reached")
                happy = True

        return happy

    def enumerate_structures(self):
        # Map chemical formula to elements

        stoichiometries, elements =\
            get_stoich_from_formulas(self.chemical_formulas)

        if self.source == 'prototypes':
            for stoichiometry in stoichiometries:
                npoints = sum([int(s) for s in stoichiometry.split('_')])
                E = Enumeration(stoichiometry, num_start=1, num_end=npoints,
                                SG_start=1, SG_end=230, num_type='wyckoff')

                E.store_enumeration(filename=self.db_filename)

            AE = AtomsEnumeration(elements, self.max_atoms)
            AE.store_atom_enumeration(filename=self.db_filename,
                                      multithread=True)
        else:
            raise NotImplementedError  # OQMD interface not implemented

    def expand_structures(self, chemical_formula=None, max_atoms=None):
        """Add additional structures to enumerated space, by specifying new chemical
        formula or allowing more atoms per unit cell"""
        if chemical_formula:
            self.chemical_fomulas += [chemical_formula]
        stoichiometries, elements =\
            get_stoich_from_formulas(self.chemical_formulas)
        if max_atoms:
            self.max_atoms = max_atoms

        for stoichiometry in stoichiometries:
            npoints = sum([int(s) for s in stoichiometry.split('_')])
            E = Enumeration(stoichiometry, num_start=1, num_end=npoints,
                            SG_start=1, SG_end=230, num_type='wyckoff')

            E.store_enumeration(filename=self.db_filename)

        AE = AtomsEnumeration(elements, self.max_atoms)
        AE.store_atom_enumeration(filename=self.db_filename,
                                  multithread=True)

        self.generate_fingerprints()

    def generate_fingerprints(self,
                              feature_methods=['voronoi'],
                              completed=False):
        self.DB._connect()
        ase_db = self.DB.ase_db
        atoms_list = []
        target_list = {}

        ids = self.DB.get_new_fingerprint_ids(completed=completed)
        if not ids:
            print('Fingerprints up do date!')
            return
        atoms_df = pd.DataFrame(columns=['atoms'])
        targets_df = pd.DataFrame(columns=['id', 'Ef'])

        for id in ids:
            row = ase_db.get(id=id)
            atoms_df = atoms_df.append({'atoms': row.toatoms()},
                                       ignore_index=True)
            if row.get('Ef', None):
                targets_df = targets_df.append({'id': id, 'Ef': row.Ef},
                                               ignore_index=True)

        targets_df = targets_df.astype({'id': int})
        FP = FingerPrint(feature_methods=feature_methods,
                         input_data=atoms_df,
                         input_index=['atoms'])

        fingerprints_df = FP.generate_fingerprints()['voronoi'].assign(id=ids)
        fingerprints_df = fingerprints_df.astype({'id': int})

        self.DB.write_dataframe(table='fingerprint',
                                df=fingerprints_df)

        self.DB.write_dataframe(table='target',
                                df=targets_df)

    def save_formation_energies(self):
        ase_db = self.DB.ase_db

        # Should be updated to only run over new completed ids
        for row in self.DB.ase_db.select('-Ef', relaxed=1):
            energy = row.energy
            references = [0, 0]  # Update
            formation_energy = (energy - sum(references)) / row.natoms

            ase_db.update(id=row.id, Ef=formation_energy)

    def train_ml(self, model='catlearn'):
        train_features = self.DB.load_dataframe(
            'fingerprint', ids=self.train_ids)
        ids_train = train_features.pop('id')
        targets = self.DB.load_dataframe('target', ids=self.train_ids)
        ids_targets = targets.pop('id')
        test_features = self.DB.load_dataframe(
            'fingerprint', ids=self.test_ids)
        ids_test = test_features.pop('id')

        assert np.all(ids_train == ids_targets)
        assert np.all(ids_test == self.test_ids)

        self.targets = targets.values

        train_features, test_features = clean_features(train_features.values,
                                                       test_features.values)

        Model = get_regression_model(model)(train_features, self.targets)
        predictions = Model.predict(test_features)

        self.energies = predictions['prediction']
        index = [i for i in range(len(self.test_ids))
                 if np.isfinite(self.energies[i])]
        self.energies = self.energies[index, 0]
        self.test_ids = list(np.array(self.test_ids)[index])
        self.uncertainties = predictions['uncertainty'][index]

    def acquire_batch(self, kappa=1.5, method='mix', batch_size=None):
        if not batch_size:
            batch_size = self.corrected_batch_size

        if method == 'random':
            noncompleted_ids = self.DB.get_completed_structure_ids(
                completed=False)
            indices = np.random.randint(len(noncompleted_ids), size=batch_size)
            self.batch_ids = np.array(noncompleted_ids)[indices]
            return

        if method == 'energy':
            n_u = 0
        elif method == 'mix':
            n_u = batch_size // 3

        n_e = batch_size - n_u

        # Simple acquisition function
        values = self.energies - kappa * self.uncertainties
        self.acqu = values
        indices_e = np.argsort(values)
        indices_u = np.argsort(-self.uncertainties)

        indices = list(indices_e[:n_e]) + list(indices_u[:n_u])
        self.batch_ids = list(np.array(self.test_ids)[indices])[
            :batch_size]


if __name__ == "__main__":
    from protosearch.workflow.workflow_dummy import DummyWorkflow

    ALL = ActiveLearningLoop(
        chemical_formulas=['Cu2O'],
        max_atoms=10,
        # Workflow=DummyWorkflow,
        batch_size=1,
        check_frequency=0.6,
        frac_jobs_limit=0.4,
        stop_mode="job_fraction_limit")
    ALL.run()
