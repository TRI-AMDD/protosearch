import time
import numpy as np
import pandas as pd
from ase.symbols import string2symbols
from catlearn.active_learning import acquisition_functions

from protosearch.build_bulk.enumeration import (
    Enumeration, AtomsEnumeration, get_stoich_from_formulas)
from protosearch.workflow.prototype_db import PrototypeSQL
from protosearch.workflow.workflow import Workflow as WORKFLOW
from protosearch.ml_modelling.fingerprint import FingerPrint, clean_features
from protosearch.ml_modelling.regression_model import get_regression_model
from protosearch.utils.standards import CrystalStandards


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

    def _initialize(self):
        if not self.status['enumerated']:
            self.enumerate_structures()
            self.DB.write_status(enumerated=1)
        if not self.status['fingerprinted']:
            self.generate_fingerprints()
            self.DB.write_status(fingerprinted=1)

        self.batch_no = (self.status.get('last_batch_no', None) or 0) + 1

        elements = []
        for formula in self.chemical_formulas:
            elements += string2symbols(formula)
        elements = list(set(elements))
        self.Workflow.submit_standard_states(elements, batch_no=self.batch_no)
        self.DB.write_status(last_batch_no=self.batch_no)

        failed_ids = self.monitor_submissions(batch_size=0,
                                              standard_state=1)
        if len(failed_ids) > 0:
            raise RuntimeError(
                'Standard state calculation failed for one or more species')

        self.DB.write_status(initialized=1)

    def monitor_submissions(self, batch_size, **kwargs):
        WF = self.Workflow
        completed_ids, failed_ids, running_ids =\
            WF.check_submissions(**kwargs)
        t0 = time.time()
        min_running_jobs = batch_size
        while len(running_ids) > min_running_jobs:
            WF.recollect()
            temp_completed_ids, temp_failed_ids, running_ids =\
                WF.check_submissions(**kwargs)
            failed_ids += temp_failed_ids
            completed_ids += temp_completed_ids
            t = time.time() - t0

            print('Status for calculations at {:.2f} min:'
                  .format(t / 60))
            print('  {} completed'.format(len(completed_ids)))
            print('  {} running'.format(len(running_ids)))
            print('  {} errored'.format(len(failed_ids)))

            if temp_failed_ids:
                print('\nOne or more jobs failed:')
                for i in failed_ids:
                    row = self.DB.ase_db.get(id=i)
                    message = row.data.get('error', 'No message').split('\n')
                    if len(message) > 1:
                        message = message[-2]
                    else:
                        message = message[0]
                    print('  Job {}:'.format(i))
                    print('    ' + message)
            time.sleep(self.check_frequency)

        # Make sure the number or running jobs stays constant
        self.corrected_batch_size = min_running_jobs + \
            self.batch_size - len(running_ids)

        self.save_formation_energies()

        return failed_ids

    def run(self):
        """Run actice learning loop"""

        WF = self.Workflow
        self.status = self.DB.get_status()

        if not self.status['initialized']:
            self._initialize()
            self.batch_no += 1
            self.acquire_batch(method='random', batch_size=self.batch_size)
            WF.submit_id_batch(self.batch_ids)
        else:
            self.batch_no = self.status['last_batch_no']

        happy = False
        while not happy:
            self.monitor_submissions(batch_size=self.batch_size)
            self.DB.write_job_status()
            self.generate_fingerprints(completed=True)

            happy = self.evaluate()

            self.train_ids = self.DB.get_completed_structure_ids()
            self.test_ids = self.DB.get_uncompleted_structure_ids()
            self.get_ml_prediction()

            all_ids = np.append(self.test_ids, self.train_ids)
            all_energies = np.append(self.energies, self.targets)
            all_uncertainties = np.append(self.uncertainties,
                                          np.zeros_like(self.targets))
            self.DB.write_predictions(
                self.batch_no, all_ids, all_energies, all_uncertainties)

            self.acquire_batch()

            self.batch_no += 1

            WF.submit_id_batch(self.batch_ids)

    def test_run(self, acquire='random', kappa=None):
        """Use already completed calculations to test the loop """
        WF = self.Workflow
        self.status = self.DB.get_status()
        self.corrected_batch_size = self.batch_size

        self.batch_no = 1
        self.batch_ids = []
        self.test_ids = self.DB.get_initial_structure_ids(completed=True)
        completed_ids = np.array(self.DB.get_completed_structure_ids())
        self.train_ids = completed_ids[:self.batch_size]

        completed_initial = [self.DB.ase_db.get(
            int(t)).initial_id for t in self.train_ids]

        self.test_ids = np.array([
            t for t in self.test_ids if not t in completed_initial])

        while len(self.test_ids) > 0:
            self.get_ml_prediction()
            all_ids = np.append(self.test_ids, self.train_ids)
            all_energies = np.append(self.energies, self.targets)
            all_uncertainties = np.append(self.uncertainties,
                                          np.zeros_like(self.targets))

            prediction = {'batch_no': self.batch_no,
                          'ids': all_ids,
                          'energies': all_energies,
                          'vars': all_uncertainties}

            yield prediction

            if acquire == 'random':
                indices = np.random.randint(len(self.test_ids),
                                            size=self.batch_size)
                self.batch_ids = np.array(self.test_ids)[indices]
            else:
                self.acquire_batch(method=acquire, kappa=kappa)

            completed_ids = [self.DB.ase_db.get(int(t)).final_id
                             for t in self.batch_ids]
            self.train_ids = np.sort(np.append(self.train_ids, completed_ids))

            self.test_ids = np.sort(np.array([
                t for t in self.test_ids if not t in self.batch_ids]))

            self.batch_no += 1

    def get_frac_of_systems_processed(self):
        """
        Get the fraction of structures that have been processed, including
        completed as well as errored jobs.
        """
        n_nonrelaxed = self.DB.ase_db.count(relaxed=0)
        n_errored = self.DB.ase_db.count(relaxed=0, completed=-1)
        n_relaxed = self.DB.ase_db.count(relaxed=1)

        frac = (n_relaxed + n_errored) / n_nonrelaxed

        return frac

    def evaluate(self):
        happy = False

        # Upper limit on Systems/Jobs processed
        if self.stop_mode == "job_fraction_limit":
            frac_i = self.get_frac_of_systems_processed()
            print("Structures processed: {:.2f} %".format(frac_i))
            if frac_i >= self.frac_jobs_limit:
                print("Ending Loop! Upper limit of processed jobs reached")
                happy = True

        return happy

    def enumerate_structures(self, spacegroups=None):
        # Map chemical formula to elements

        stoichiometries, self.elements =\
            get_stoich_from_formulas(self.chemical_formulas)

        if self.source == 'prototypes':
            for stoichiometry in stoichiometries:
                self.enumerate_prototypes(stoichiometry,
                                          spacegroups)

            AE = AtomsEnumeration(self.elements, self.max_atoms)
            AE.store_atom_enumeration(filename=self.db_filename,
                                      multithread=False)
        else:
            raise NotImplementedError  # OQMD interface not implemented

    def enumerate_prototypes(self, stoichiometry, spacegroups=None):
        npoints = sum([int(s) for s in stoichiometry.split('_')])

        if spacegroups is not None:
            SG_start_end = [[s, s] for s in spacegroups]
        else:
            SG_start_end = [[1, 230]]

        for SG_start, SG_end in SG_start_end:
            E = Enumeration(stoichiometry,
                            num_start=1,
                            num_end=npoints,
                            SG_start=SG_start,
                            SG_end=SG_end,
                            num_type='wyckoff')
            E.store_enumeration(filename=self.db_filename)

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
            if row.get('Ef', None) is not None:
                targets_df = targets_df.append({'id': id, 'Ef': row.Ef},
                                               ignore_index=True)

        targets_df = targets_df.astype({'id': int, 'Ef': float})
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

        for row in self.DB.ase_db.select('-Ef', relaxed=1):
            energy = row.energy
            elements, counts = np.unique(
                row.toatoms().symbols, return_counts=True)
            ref_energies = []
            for i, e in enumerate(elements):
                ref_row = list(self.DB.ase_db.select(e,
                                                     relaxed=1,
                                                     standard_state=1,
                                                     limit=1))
                if len(ref_row) == 0:
                    print('Standard state for {} not found'.format(row.formula))
                else:
                    energy_atom = ref_row[0].energy / ref_row[0].natoms
                    ref_energies += [counts[i] * energy_atom]

            formation_energy = (energy - sum(ref_energies)) / row.natoms
            ase_db.update(id=row.id, Ef=formation_energy)

    def get_ml_prediction(self, model='catlearn'):
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

        features, bad_indices = \
            clean_features({'train': train_features.values,
                            'test': test_features.values})

        self.train_ids = np.delete(self.train_ids, bad_indices['train'])
        self.targets = np.delete(self.targets, bad_indices['train'])
        self.test_ids = np.delete(self.test_ids, bad_indices['test'])

        Model = get_regression_model(model)(features['train'], self.targets)
        predictions = Model.predict(features['test'])

        self.energies = predictions['prediction']
        index = [i for i in range(len(self.test_ids))
                 if np.isfinite(self.energies[i])]
        self.test_ids = np.array(self.test_ids)[index]
        self.uncertainties = predictions['uncertainty'][index]

    def acquire_batch(self, method='UCB', kappa=0.5, batch_size=None):
        if not batch_size:
            batch_size = self.corrected_batch_size

        if method == 'random':
            uncompleted_ids = self.DB.get_uncompleted_structure_ids()
            acquistision_values = np.zeros(len(uncompleted_ids))
            indices = np.random.randint(len(uncompleted_ids), size=batch_size)
            self.batch_ids = np.array(uncompleted_ids)[indices]
            return

        if method == 'UCB':
            """
            Upper confidence bound:
            Trade-off between low energy and high uncertainty
            """
            acquisition_values = self.energies - kappa * self.uncertainties
            indices = np.argsort(acquisition_values)
        elif method in ['optimistic', 'proximity', 'optimistic_proximity',
                        'probability_density']:
            AF = getattr(acquisition_functions, method)
            acquisition_values = AF(y_best=-np.min(self.targets),
                                    predictions=-self.energies,
                                    uncertainty=self.uncertainties)

            indices = np.argsort(acquisition_values)[::-1]

        elif method in ['EI', 'PI']:
            AF = getattr(acquisition_functions, method)
            acquisition_values = AF(y_best=np.min(self.targets),
                                    predictions=self.energies,
                                    uncertainty=self.uncertainties,
                                    objective='min'
                                    )

            indices = np.argsort(acquisition_values)[::-1]

        elif method == 'mix':
            """
            Choose two different batches - one targeting low energy,
            the other targeting high uncertainty
            """
            n_high_u = batch_size // 3
            n_high_e = batch_size - n_high_u
            indices_energy = np.argsort(self.energies)
            indices_u = np.argsort(-self.uncertainties)
            indices = np.append(indices_energy[:n_high_e],
                                indices_u[:n_high_u])
            acquisition_values = -self.uncertainties

        self.batch_ids = np.array(self.test_ids)[indices][
            :batch_size]
        self.acquisition_values = acquisition_values


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
