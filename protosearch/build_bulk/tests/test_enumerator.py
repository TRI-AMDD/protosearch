import shutil
import os
import sys
import tempfile
import unittest
import numpy as np
from ase.io import read

from protosearch.build_bulk.enumeration import Enumeration
from protosearch.build_bulk.enumeration import AtomsEnumeration
from protosearch.build_bulk.classification import get_classification


class EnumerationTest(unittest.TestCase):
    def setUp(self):
        self.pwd = os.getcwd()
        self.tempdir = tempfile.mkdtemp()
        os.chdir(self.tempdir)

    def tearDown(self):
        os.chdir(self.pwd)
        shutil.rmtree(self.tempdir)

    def test1_enumeration(self):
        E = Enumeration("1_2", num_start=1, num_end=5, SG_start=20, SG_end=22)
        E.store_enumeration('1_2.db')

    def test2_atoms_enumeration(self):
        E = AtomsEnumeration(elements={'A': ['Fe', 'Ru', 'Ir'],
                                       'B': ['O']})  # ,
        # 'C': ['C']})
        E.store_atom_enumeration('1_2.db')

    def test3_classification(self):
        path = sys.path[0]
        atoms = read(path + '/Se_mp-570481_conventional_standard.cif')
        result, param = get_classification(atoms)
        assert result == {'p_name': 'A_64_e16_14',
                          'spacegroup': 14,
                          'wyckoffs': ['e', 'e', 'e', 'e', 'e', 'e',
                                       'e', 'e', 'e', 'e', 'e', 'e',
                                       'e', 'e', 'e', 'e'],
                          'species': ['Se', 'Se', 'Se', 'Se', 'Se',
                                      'Se', 'Se', 'Se', 'Se', 'Se',
                                      'Se', 'Se', 'Se', 'Se', 'Se', 'Se']}

        param_ref = {'a': 15.739181,
                     'b': 0.9726696706772735,
                     'c': 0.6451328742335684,
                     'beta': 94.0428681461934,
                     'xe0': 0.986572,
                     'ye0': 0.593959,
                     'ze0': 0.23546,
                     'xe1': 0.984712,
                     'ye1': 0.214049,
                     'ze1': 0.120869,
                     'xe2': 0.913585,
                     'ye2': 0.675372,
                     'ze2': 0.828421,
                     'xe3': 0.906751,
                     'ye3': 0.009713,
                     'ze3': 0.847639,
                     'xe4': 0.814616,
                     'ye4': 0.142437,
                     'ze4': 0.451916,
                     'xe5': 0.806848,
                     'ye5': 0.690537,
                     'ze5': 0.978908,
                     'xe6': 0.769548,
                     'ye6': 0.047426,
                     'ze6': 0.277226,
                     'xe7': 0.76865,
                     'ye7': 0.215177,
                     'ze7': 0.888981,
                     'xe8': 0.761848,
                     'ye8': 0.508143,
                     'ze8': 0.274501,
                     'xe9': 0.695855,
                     'ye9': 0.589422,
                     'ze9': 0.437045,
                     'xe10': 0.691437,
                     'ye10': 0.736288,
                     'ze10': 0.364628,
                     'xe11': 0.646212,
                     'ye11': 0.524822,
                     'ze11': 0.842267,
                     'xe12': 0.644476,
                     'ye12': 0.185345,
                     'ze12': -0.000248,
                     'xe13': 0.52639,
                     'ye13': 0.213948,
                     'ze13': 0.849520,
                     'xe14': 0.525862,
                     'ye14': 0.645501,
                     'ze14': 0.102871,
                     'xe15': 0.524356,
                     'ye15': 0.04622,
                     'ze15': 0.241639}

        for key, value in param_ref.items():
            assert np.isclose(value, param[key], rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
