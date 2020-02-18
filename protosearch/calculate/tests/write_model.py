import shutil
import os
import tempfile
import unittest

from protosearch.build_bulk.build_bulk import BuildBulk
from protosearch.calculate.submit import TriSubmit, NerscSubmit
from protosearch.calculate.vasp import VaspModel


class SubmitTest(unittest.TestCase):

    def test_write_model(self):
        bb_iron = BuildBulk(225, ['a', 'c'], ['Mn', 'O'])
        atoms = bb_iron.get_atoms()
        self.submitter = TriSubmit(atoms,
                                   basepath_ext='tests')

        self.submitter.write_model('.')

    def test_write_model_nersc(self):
        bb_iron = BuildBulk(225, ['a', 'c'], ['Mn', 'O'])
        atoms = bb_iron.get_atoms()
        submitter = NerscSubmit(atoms,
                                account='projectname',
                                basepath='.')

        submitter.set_execution_path(strict_format=False)
        submitter.write_submission_files()
        submitter.write_submit_script()


if __name__ == '__main__':
    unittest.main()
