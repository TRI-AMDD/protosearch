import shutil
import os
import tempfile
import unittest

from protosearch.build_bulk.build_bulk import BuildBulk
from protosearch.calculate.submit import TriSubmit


class TriSubmitTest(unittest.TestCase):
    def setUp(self):
        self.bb_iron = BuildBulk(225, ['a', 'c'], ['Mn', 'O'])
        atoms = self.bb_iron.get_atoms_from_poscar()
        self.submitter = TriSubmit(atoms,
                                   basepath_ext='tests',
                                   calc_parameters={'encut': 300,
                                                    'kspacing': 30}
                                   )
        self.pwd = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.pwd)
        shutil.rmtree(self.tempdir)

    def test_write_poscar(self):
        self.submitter.write_poscar('.')

    def test_write_model(self):
        self.submitter.write_model('.')
        with open('model.py') as f:
            model_text = f.readlines()

    def test_submit(self):
        self.submitter.submit_calculation()


if __name__ == '__main__':
    unittest.main()
