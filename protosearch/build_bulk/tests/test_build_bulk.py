import shutil
import os
import tempfile
import unittest
import numpy as np

from protosearch.build_bulk.build_bulk import BuildBulk


class BuildBulkTest(unittest.TestCase):
    def setUp(self):
        self.pwd = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.pwd)
        shutil.rmtree(self.tempdir)

    def test_no_parameters(self):
        BB = BuildBulk(225, ['a'], ["Cu"])
        BB.get_poscar()
        atoms = BB.get_atoms_from_poscar()
        assert np.isclose(atoms.cell[0, 2], 1.28, rtol=0.01)

    def test_set_parameters(self):
        BB = BuildBulk(225, ['a'], ["Cu"],
                                 cell_parameters={'a': 5})
        BB.get_poscar()
        atoms = BB.get_atoms_from_poscar()
        assert atoms.cell[0, 2] == 2.5
        

if __name__ == '__main__':
    unittest.main()
