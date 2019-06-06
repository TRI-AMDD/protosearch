from protosearch.build_bulk.enumeration import Enumeration, AtomsEnumeration
from protosearch.workflow.prototype_db import PrototypeSQL
from protosearch.workflow.workflow import Workflow
from protosearch.ml_modelling.catlearn_interface import get_voro_fingerprint, train


class ActiveLearningLoop:

    def __init__(self,
                 chemical_formulas,
                 source='oqmd_icsd',
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

    def run(self):

        # Initialize:
        # enumerate structures with Build bulk
        # generate fingerprints and save to db
        # aqcusition -> select random batch
        self.initialize()

        # LOOP:
        happy = False
        while not happy:
            # submit structures with WorkFLow
            WF = Workflow()
            WF.submit_atoms_batch(self.batch_atoms)

            # get completed calculations from WorkFlow
            completed_ids = WF.check_submissions()
            # get formation energy of completed jobs and save to db
            self.get_formation_energies()

            # save formation energies + regenerated fingerprints to db
            self.generate_fingerprints(ids=completed_ids)

            # retrain ML model
            self.train_ml()

            # acqusition -> select batch
            self.acquire_batch()

    def initialize(self):
        self.enumerate_structures()
        self.generate_fingerprints()
        # Run standard states?

        batch_ids = list(range(self.batch_size))
        DB = PrototypeSQL()
        self.batch_atoms = DB.get_atoms_list(batch_ids)

    def enumerate_structures(self):

        # Map chemical formula to elements
        # stoichiometries, elements =\
        # get_stoich_from_formulas(self.chemical_formulas)

        stoichiometries = ['1_2']

        elements = {'A': ['Fe'],
                    'B': ['O']}

        for stoichiometry in stoichiometries:
            E = Enumeration(stoichiometry)
            E.store_enumeration()

        AE = AtomsEnumeration()
        AE.store_atom_enumeration()

    def generate_fingerprint(self, code='catlearn', ids=None):
        DB = PrototypeSQL()
        ase_db = db.ase_db
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
            DB.save_fingerprint(id, input_data=fingerprint_data[i],
                                output_data=output_data)

    def save_formation_energies(self, ids):
        DB = PrototypeSQL()
        ase_db = db.ase_db

        for id in ids:
            row = ase_db.get(id)
            energy = row.energy

            # MENG
            # references = ?
            formation_energy = energy - sum(references)

            ase_db.update(id=id, Ef=formation_energy)

    def train_ml(self):

        # Raul
        #ids, energies, uncertainties = Ml_model(...).get_prediction()

        DB = PrototypeSQL()

        DB.save_predictions(ids, energies, uncertainties)

    def acquire_batch(self):

        # TODO
        # Acqusition functions
        # batch_ids = acquire (...)

        DB = PrototypeSQL()
        self.batch_atoms = DB.get_atoms_list(batch_ids)
