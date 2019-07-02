import sys
import time
import numpy as np
import pylab as p

from protosearch.build_bulk.enumeration import Enumeration, AtomsEnumeration, get_stoich_from_formulas
from protosearch.workflow.prototype_db import PrototypeSQL
from protosearch.workflow.workflow import Workflow
from protosearch.ml_modelling.catlearn_interface import get_voro_fingerprint, predict


class ActiveLearningLoop:

    def __init__(self,
                 chemical_formulas,
                 source='prototypes',
                 batch_size=10):
        """
        Class to run the active learning loop

        Parameters:
        ----------
        chemical_formulas: list of strings
            chemical formulas to investigate, such as:
            ['IrO2', 'IrO3']
        source: str
            Specify how to generate the structures. Options are:
            'oqmd_icsd': Experimental OQMD entries
            'oqmd_all': All OQMD structures
            'prototypes': Enumerate all prototypes.
        """
        if isinstance(chemical_formulas, str):
            chemical_formulas = [chemical_formulas]
        self.chemical_formulas = chemical_formulas
        self.source = source
        self.batch_size = batch_size
        self.db_filename = '_'.join(chemical_formulas) + '.db'
        self.DB = PrototypeSQL(self.db_filename)
        self.DB.write_status(
            chemical_formulas=chemical_formulas, batch_size=batch_size)

    def run(self):
        """Run actice learning loop"""

        WF = Workflow(db_filename=self.db_filename)
        self.status = self.DB.get_status()

        if self.status['initialized'] is None:
            self.batch_no = 1
            self.initialize()
            self.DB.write_status(initialized=1)
            WF.submit_id_batch(self.batch_ids)
            self.DB.write_status(last_batch_no=self.batch_no)
        else:
            self.batch_no = self.status['last_batch_no']
        happy = False
        while not happy:
            # Wait a while - time.sleep?
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
                time.sleep(60)

            self.DB.write_job_status()

            # Make sure the number or running jobs doesn't blow up
            self.corrected_batch_size = self.batch_size - len(running_ids)

            # get formation energy of completed jobs and save to db
            self.save_formation_energies(completed_ids)

            happy = self.evaluate()

            # save formation energies + regenerated fingerprints to db
            self.generate_fingerprints(completed=True)

            # retrain ML model
            self.train_ml()

            # acqusition -> select batch
            self.acquire_batch()

            self.plot_predictions()
            self.batch_no += 1
            # submit structures with WorkFLow
            WF.submit_id_batch(self.batch_ids)
            self.DB.write_status(last_batch_no=self.batch_no)

    def initialize(self):
        self.enumerate_structures()
        self.DB.write_status(enumerated=1)

        self.generate_fingerprints()
        self.DB.write_status(fingerprinted=1)

        # Run standard states?
        self.batch_ids = self.DB.get_structure_ids(n_ids=self.batch_size)

    def evaluate(self):
        happy = False
        # function to determine when the prediction ends
        return happy

    def enumerate_structures(self):
        # Map chemical formula to elements

        stoichiometries, elements =\
            get_stoich_from_formulas(self.chemical_formulas)

        if self.source == 'prototypes':
            for stoichiometry in stoichiometries:
                E = Enumeration(stoichiometry, num_start=1, num_end=3,
                                SG_start=1, SG_end=230, num_type='wyckoff')
                E.store_enumeration(filename=self.db_filename)

            AE = AtomsEnumeration(elements)
            AE.store_atom_enumeration(filename=self.db_filename)
        else:
            raise NotImplementedError  # OQMD interface not implemented

    def generate_fingerprints(self, code='catlearn', completed=False):
        self.DB._connect()
        ase_db = self.DB.ase_db
        atoms_list = []
        output_list = {}

        if completed:
            ids = self.DB.get_new_fingerprint_ids()
            for id in ids:
                row = ase_db.get(id=id)
                atoms_list += [row.toatoms()]
                Ef = row.get('Ef', None)
                if Ef:
                    output_list[str(id)] = {'Ef': Ef}
        else:
            ids = []
            for row in ase_db.select():
                ids += [row.id]
                atoms_list += [row.toatoms()]
                Ef = row.get('Ef', None)
                if Ef:
                    output_list[str(row.id)] = {'Ef': Ef}
        fingerprint_data = get_voro_fingerprint(atoms_list)
        for i, id in enumerate(ids):
            output_data = output_list.get(str(id), None)
            self.DB.save_fingerprint(id, input_data=fingerprint_data[i],
                                     output_data=output_data)

    def save_formation_energies(self, ids):
        ase_db = self.DB.ase_db

        # Should be updated to only run over new completed ids
        for row in self.DB.ase_db.select('-Ef', completed=1, relaxed=1):
            energy = row.energy
            # MENG
            references = [0, 0]  # Update
            formation_energy = (energy - sum(references)) / row.natoms

            ase_db.update(id=row.id, Ef=formation_energy)

    def train_ml(self):
        train_ids = []
        # replace by direct SQL
        for row in self.DB.ase_db.select(completed=1, relaxed=1):
            train_ids += [row.id]
        test_ids = []
        # replace by direct SQL
        for row in self.DB.ase_db.select(completed=0, submitted=0):
            test_ids += [row.id]

        train_features, train_target = self.DB.get_fingerprints(ids=train_ids)

        test_features, test_target = self.DB.get_fingerprints(ids=test_ids)

        self.train_ids = train_ids
        self.train_target = train_target

        predictions = predict(train_features, train_target, test_features)
        self.energies = predictions['prediction']
        self.uncertainties = predictions['uncertainty']
        self.test_ids = test_ids

    def acquire_batch(self, kappa=1.5):

        # Simple acquisition function
        values = self.energies - kappa * self.uncertainties

        self.acqu = values

        indices = np.argsort(values)
        self.batch_ids = list(np.array(self.test_ids)[indices])[
            :self.corrected_batch_size]

    def plot_predictions(self):
        all_ids = np.array(self.test_ids + self.train_ids)
        energies = np.array(list(self.energies) + list(self.train_target))
        uncertainties = np.array(
            list(self.uncertainties) + [0 for t in self.train_target])

        acqu = np.array(list(self.acqu) + list(self.train_target))

        idx = np.argsort(energies)
        all_ids = all_ids[idx]

        p.plot(range(len(idx)), energies[idx], label='prediction')
        p.plot(range(len(idx)), energies[idx] - uncertainties[idx]/2, 'k--')
        p.plot(range(len(idx)), energies[idx] + uncertainties[idx]/2, 'k--')

        p.plot(range(len(idx)),
               acqu[idx], label='acquisition', linestyle='--', color='0.6')
        p.plot([0, len(idx)], [min(energies), min(energies)],
               linestyle='--', color='0.4')

        for b, bi in enumerate(self.train_ids):
            i = list(all_ids).index(bi)
            if b == 0:
                p.plot(i, energies[idx][i], 'bo', label='calculated')
            else:
                p.plot(i, energies[idx][i], 'bo')

        for b, bi in enumerate(self.batch_ids):
            i = list(all_ids).index(bi)
            if b == 0:
                p.plot(i, energies[idx][i], 'ro', label='next batch')
            else:
                p.plot(i, energies[idx][i], 'ro')

        p.xlabel('Structure id')
        p.ylabel('Energy(eV)')
        p.legend()
        p.savefig('prediction_batch_{}.png'.format(self.batch_no))


if __name__ == "__main__":
    ALL = ActiveLearningLoop(chemical_formulas=['IrO2'])
    ALL.run()
