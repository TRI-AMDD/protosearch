import shutil
import os
import tempfile
import unittest
import time
import numpy as np

from protosearch.build_bulk.cell_parameters import CellParameters


class CellParametersTest(unittest.TestCase):
    def setUp(self):
        self.pwd = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)
        self.reference_output = {'a': 2.8512,
                                 'b/a': 0.890999894217854,
                                 'c/a': 0.7806562271349045,
                                 'alpha': 69.93615023959542,
                                 'beta': 88.3269216779091,
                                 'gamma': 80.00000121741787,
                                 'xa0': -2.1469986566223507,
                                 'ya0': -0.749693073805286,
                                 'za0': 2.2156831161918893,
                                 'xa1': -0.7075949700128609,
                                 'ya1': 0.6902736084156266,
                                 'za1': -0.3128838868072398}

    def tearDown(self):
        os.chdir(self.pwd)
        shutil.rmtree(self.tempdir)

    def test_cell_parameters(self):
        CP = CellParameters(1, ['a', 'a'],
                            ['Fe', 'O'])
        # CP = CellParameters(14,
        #                    ['e', 'e', 'e', 'e', 'e', 'e', 'e' , 'e', 'e',
        #                     'e', 'e', 'e', 'e', 'e', 'e', 'e'],
        #                    ['Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se',
        #                     'Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se'])

        t0 = time.time()
        parameters = CP.get_parameter_estimate()
        t = time.time() - t0
        print('optimized small cell in {} sec'.format(t))

        for key, value in self.reference_output.items():
            assert key in parameters

        atoms = CP.get_atoms(parameters, primitive=True)

    def test_cell_parameters_big(self):
        CP = CellParameters(14,
                            ['e', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e',
                             'e', 'e', 'e', 'e', 'e', 'e', 'e'],
                            ['Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se',
                             'Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se'])

        t0 = time.time()
        parameters = CP.get_parameter_estimate()
        t = time.time() - t0
        print('optimized big cell in {} sec'.format(t))

        atoms = CP.get_atoms(parameters, primitive=True)
        import ase
        ase.visualize.view(atoms)

    def test_fix_wyckoffs(self):
        wyckoff_coor = self.reference_output
        for p in ['a', 'b/a', 'c/a', 'alpha', 'beta', 'gamma']:
            del wyckoff_coor[p]

        CP = CellParameters(1, ['a', 'a'],
                            ['Fe', 'O'])

        parameters = CP.get_parameter_estimate(
            master_parameters=wyckoff_coor)

        for key, value in wyckoff_coor.items():
            assert value == parameters[key], '{}: {} != {}'.\
                format(key, value, parameters[key])


if __name__ == '__main__':
    unittest.main()
