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
                                 'b': 2.85 * 0.890999894217854,
                                 'c': 2.85 * 0.7806562271349045,
                                 'alpha': 69.93615023959542,
                                 'beta': 88.3269216779091,
                                 'gamma': 80.00000121741787,
                                 'xa0':  0.012189615713195129,
                                 'ya0': 0.914497412930739,
                                 'za0': 0.5340268669248445,
                                 'xa1': 0.498548108759491,
                                 'ya1': 0.8925062992580316,
                                 'za1': 0.9605024862250223}

    def tearDown(self):
        os.chdir(self.pwd)
        shutil.rmtree(self.tempdir)

    def test1_cell_parameters(self):
        CP = CellParameters(spacegroup=1,
                            wyckoffs=['a', 'a'],
                            species=['Fe', 'O'])

        t0 = time.time()
        parameters = CP.get_parameter_estimate()[0]
        t = time.time() - t0
        print('optimized small cell in {} sec'.format(t))

        for key, value in self.reference_output.items():
            assert key in parameters

    def test2_fix_wyckoffs(self):
        wyckoff_coor = self.reference_output
        for p in ['a', 'b', 'c', 'alpha', 'beta', 'gamma']:
            del wyckoff_coor[p]
        CP = CellParameters(spacegroup=1,
                            wyckoffs=['a', 'a'],
                            species=['Fe', 'O'])

        parameters = CP.get_parameter_estimate(
            master_parameters=wyckoff_coor)[0]

        for key, value in wyckoff_coor.items():
            assert np.isclose(value,  parameters[key]), '{}: {} != {}'.\
                format(key, value, parameters[key])

    def test3_anatase(self):
        """ Several discinct structures exists for the anatase prototype
        with anatase corresponding to ze1=0.2.
        Make sure several candidates are returned, including anatase.
        """
        CP = CellParameters(141, ['a', 'e'], ['Ti', 'O'])

        parameters = CP.get_parameter_estimate(max_candidates=None)

        zes = [p['ze1'] for p in parameters]

        zes = [min([z, 1-z]) for z in zes]

        assert len(parameters) > 3

        assert np.any(np.isclose(zes, 0.2, atol=0.04))

    def test4_cromium_oxide(self):

        CP = CellParameters(167, ['c', 'e'], ['Cr', 'O'])

        parameters = CP.get_parameter_estimate(max_candidates=None)

        params = [[p['zc0'], p['xe1']] for p in parameters]

        assert len(params) > 2


if __name__ == '__main__':
    unittest.main()
