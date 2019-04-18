import sys
import shutil
import os
import numpy as np
import tempfile
import unittest
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


    def test_lattice_parameters(self, id=63872):
        path = sys.path[0]
        db = connect(path + '/oqmd_ver3.db')
        atoms = db.get(id=id).toatoms()
        prototype, parameters = get_classification(atoms)

        for p in ['a','b','c']:
            if p in parameters:
                del parameters[p]

        CP = CellParameters(prototype['spacegroup'],
                            prototype['wyckoffs'],
                            prototype['species'])

        parameters = CP.get_parameter_estimate(
                            master_parameters=parameters)
        atoms = CP.get_atoms(fix_parameters=parameters)

        assert np.isclose(atoms.get_volume(), 943.65, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
