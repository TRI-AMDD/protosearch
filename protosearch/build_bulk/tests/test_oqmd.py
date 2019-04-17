import sys
import shutil
import os
import tempfile
import unittest

from protosearch.build_bulk.oqmd_interface import OqmdInterface


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


if __name__ == '__main__':
    unittest.main()
