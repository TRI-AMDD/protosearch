import shutil
import os
import tempfile
import unittest

from protosearch.build_bulk.build_bulk import BuildBulk
from protosearch.calculate.submit import TriSubmit
from protosearch.calculate.vasp import VaspModel


class TriSubmitTest(unittest.TestCase):
    def test_write_simpel_model(self):
        self.bb_iron = BuildBulk(225, ['a', 'c'], ['Mn', 'O'])
        self.submitter = TriSubmit(self.bb_iron.atoms,
                                   basepath_ext='tests')

        self.submitter.write_simple_model('.')

    def test_write_model(self):
        self.bb_iron = BuildBulk(225, ['a', 'c'], ['Mn', 'O'])
        self.submitter = TriSubmit(self.bb_iron.atoms,
                                   basepath_ext='tests')

        self.submitter.write_model('.')

    def test_write_model_source(self):
        Model = VaspModel(calc_parameters=None,
                          symbols=['Mn', 'O'])
        modelstr = Model.get_model()
        with open('model_clean_source.py', 'w') as f:
            f.write(modelstr)

        
if __name__ == '__main__':
    unittest.main()
