import shutil
import os
import tempfile
import unittest

from protosearch.protosearch.build_bulk.build_bulk import BuildBulk


class BuildBulkTest(unittest.TestCase):
    def setUp(self):
        self.bb_iron = BuildBulk(255, ['l'], "Cu")
        self.pwd = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.pwd)
        shutil.rmtree(self.tempdir)

    def test_write_model(self):
        self.bb_iron.write_model('.')
        with open('model.py') as f:
            model_text = f.readlines()
        pass

    def test_get_nbands(self):
        nbands_default = self.bb_iron.get_nbands()
        self.assertEqual(nbands_default, 12)

    def test_get_poscar(self):
        pass


if __name__ == '__main__':
    unittest.main()
