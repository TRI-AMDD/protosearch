import sys
import shutil
import os
import numpy as np
import tempfile
import unittest
import ase
from ase.db import connect

from protosearch.build_bulk.oqmd_interface import OqmdInterface
from protosearch.build_bulk.classification import get_classification
from protosearch.build_bulk.cell_parameters import CellParameters


class BuildBulkTest(unittest.TestCase):
    def setUp(self):
        self.pwd = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.pwd)
        shutil.rmtree(self.tempdir)

    def test_unique_prototypes(self):
        path = sys.path[0]
        O = OqmdInterface(dbfile=path + '/oqmd_ver3.db')
        p = O.get_distinct_prototypes(source='icsd',
                                      formula='ABC2',
                                      repetition=2)

        print(p)
        print('{} distinct prototypes found'.format(len(p)))

    def test_create_proto_dataset(self):
        path = sys.path[0]
        O = OqmdInterface(dbfile=path + '/oqmd_ver3.db')
        atoms_list = O.create_proto_data_set(source='icsd',
                                             chemical_formula='FeO6',
                                             repetition=1)
        # This test currently fails. There is 6 Fe instead of 6 O.
        # Need to fix atom substitution part.
        for atoms in atoms_list["atoms"][:5]:
            assert atoms.get_number_of_atoms() == 7
            assert atoms.get_chemical_symbols().count('Fe') == 1
            assert atoms.get_chemical_symbols().count('O') == 6
            ase.visualize.view(atoms)

    def test_lattice_parameters(self, id=63872):
        path = sys.path[0]
        db = connect(path + '/oqmd_ver3.db')
        atoms = db.get(id=id).toatoms()
        prototype, parameters = get_classification(atoms)

        for p in ['a', 'b', 'c']:
            if p in parameters:
                del parameters[p]

        CP = CellParameters(prototype['spacegroup'],
                            prototype['wyckoffs'],
                            prototype['species'])

        parameters = CP.get_parameter_estimate(
            master_parameters=parameters)
        atoms = CP.get_atoms(fix_parameters=parameters)
        assert np.isclose(atoms.get_volume(), 1594.64, rtol=1e-4)

        atoms = CP.get_atoms(fix_parameters=parameters,
                             primitive=True)
        assert np.isclose(atoms.get_volume(), 797.32, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
