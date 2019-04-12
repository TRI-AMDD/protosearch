import shutil
import os
import tempfile
import unittest
from protosearch.build_bulk.enumeration import Enumeration


class EnumerationTest(unittest.TestCase):
    def setUp(self):
        self.pwd = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.pwd)
        shutil.rmtree(self.tempdir)

    def test_enumeration(self):
        enumeration = Enumeration("Fe2O3", num_start=1, num_end=20)


if __name__ == '__main__':
    unittest.main()
