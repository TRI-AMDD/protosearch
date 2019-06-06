import time
import numpy as np
from protosearch.build_bulk.enumeration import Enumeration, AtomsEnumeration
from protosearch.workflow.prototype_db import PrototypeSQL
from protosearch.workflow.workflow import Workflow
from protosearch.ml_modelling.catlearn_interface import get_voro_fingerprint, train


class ActiveLearningLoop:

    def __init__(self,
                 chemical_formulas,
                 source='oqmd_icsd',
                 batch_size=1):
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
        self.db_filename = 'testloop.db'
        self.DB = PrototypeSQL(self.db_filename)

    def run(self):
        """Run loop"""
        # Initialize:
        # enumerate structures with Build bulk
        # generate fingerprints and save to db
        # aqcusition -> select random batch
        self.initialize()

        # LOOP:
        happy = False
        self.batch_no = 1
        while not happy:
            # submit structures with WorkFLow
            WF = Workflow(db_filename='testloop.db')

            WF.submit_atoms_batch(self.batch_atoms)

            # Wait a while - time.sleep?
            # get completed calculations from WorkFlow
            completed_ids = []
            t0 = time.time()
            while len(completed_ids) < max(1, self.batch_size // 2):
                completed_ids = WF.check_submissions()
                t = time.time() - t0
                print('{} jobs completed in {} sec'.format(len(completed_ids), t))
                time.sleep(10)

            # get formation energy of completed jobs and save to db
            self.save_formation_energies(completed_ids)

            happy = self.evaluate()

            # save formation energies + regenerated fingerprints to db
            self.generate_fingerprints(ids=completed_ids)

            # retrain ML model
            self.train_ml()

            # acqusition -> select batch
            self.acquire_batch()

            self.batch_no += 1

    def restart(self):
        """restart broken loop"""
        pass

    def initialize(self):
        self.enumerate_structures()
        self.generate_fingerprints()
        # Run standard states?

        batch_ids = list(range(1, self.batch_size + 1))
        self.batch_atoms = self.DB.get_atoms_list(batch_ids)

    def evaluate(self):
        happy = False
        # function to determine when the prediction ends
        return happy

    def enumerate_structures(self):

        # Map chemical formula to elements
        # stoichiometries, elements =\
        # get_stoich_from_formulas(self.chemical_formulas)

        stoichiometries = ['1_2']

        elements = {'A': ['Ir'],
                    'B': ['O']}

        for stoichiometry in stoichiometries:
            E = Enumeration(stoichiometry, num_start=1, num_end=4,
                            SG_start=76, SG_end=80)
            E.store_enumeration(filename=self.db_filename)

        AE = AtomsEnumeration(elements)
        AE.store_atom_enumeration(filename=self.db_filename)

    def generate_fingerprints(self, code='catlearn', ids=None):
        self.DB._connect()
        ase_db = self.DB.ase_db
        atoms_list = []
        output_list = {}
        if ids:
            for id in ids:
                row = ase_db.get(id)
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
                    output_list[str(id)] = {'Ef': Ef}

        fingerprint_data = get_voro_fingerprint(atoms_list)
        for i, id in enumerate(ids):
            output_data = output_list.get(str(id), None)
            self.DB.save_fingerprint(id, input_data=fingerprint_data[i],
                                     output_data=output_data)

    def save_formation_energies(self, ids):
        ase_db = self.DB.ase_db

        for id in ids:
            row = ase_db.get(id)
            energy = row.energy

            # MENG
            references = [0, 0]  # Update
            formation_energy = energy - sum(references)

            ase_db.update(id=id, Ef=formation_energy)

    def train_ml(self):
        train_ids = []
        for row in self.DB.ase_db.select(completed=1):  # replace by direct SQL
            train_ids += [row.id]
        test_ids = []
        for row in self.DB.ase_db.select(submitted=0):  # replace by direct SQL
            test_ids += [row.id]

        # Raul
        #energies, uncertainties = Ml_model(...).get_prediction(train_ids)

        self.test_ids = test_ids
        self.energies = np.random.random(len(test_ids))
        self.uncertainties = np.random.random(len(test_ids)) / 5

        self.DB.save_predictions(train_ids, self.energies, self.uncertainties)

    def acquire_batch(self, kappa=1.5):

        # Simple acquisition function
        # ids, formation_energies, uncertainties = \
        #    self.DB.read_predictions()

        values = self.energies - kappa * self.uncertainties

        indices = np.argsort(values)
        batch_ids = list(np.array(self.test_ids)[indices])
        print(batch_ids)
        self.batch_atoms = self.DB.get_atoms_list(batch_ids)


if __name__ == "__main__":
    ALL = ActiveLearningLoop(chemical_formulas=['IrO2'])
    ALL.run()
