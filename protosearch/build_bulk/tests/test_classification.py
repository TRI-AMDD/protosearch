import shutil
import os
import tempfile
import unittest
import time
import numpy as np
from ase.io import read

from protosearch.build_bulk import tests
from protosearch.build_bulk.classification import PrototypeClassification

path = list(tests.__path__)[0]


class PrototypeClassificationTest(unittest.TestCase):

    def test1_cromium_oxide(self):
        filename = path + '/Cr2O3_mp-19399_conventional_standard.cif'
        atoms = read(filename)

        PC = PrototypeClassification(atoms)

        prototype, parameters = PC.get_classification()

        prototype_ref = {'p_name': 'A2B3_6_c_e_167',
                         'structure_name': '167_Cr_c_O_e',
                         'spacegroup': 167,
                         'wyckoffs': ['c', 'e'],
                         'species': ['Cr', 'O']}

        parameters_ref = {'a': 5.09424496,
                          'b': 5.09424496,
                          'c': 13.78071749,
                          'alpha': 90,
                          'beta': 90,
                          'gamma': 120,
                          'zc0':  0.150269,
                          'xe1': 0.300642}

        for key, value in prototype_ref.items():
            assert prototype[key] == value

        for key, value in parameters_ref.items():
            assert np.isclose(parameters[key], value, rtol=1e-5)

    def test2_anatase(self):

        filename = path + '/TiO2_mp-390_conventional_standard.cif'
        atoms = read(filename)

        PC = PrototypeClassification(atoms)

        prototype, parameters = PC.get_classification()

        prototype_ref = {'p_name': 'AB2_4_a_e_141',
                         'structure_name': '141_Ti_a_O_e',
                         'spacegroup': 141,
                         'wyckoffs': ['a', 'e'],
                         'species': ['Ti', 'O']}

        parameters_ref = {'a': 3.80270954,
                          'b': 3.80270954,
                          'c': 9.74775206,
                          'alpha': 90,
                          'beta': 90,
                          'gamma': 90,
                          'ze1': 0.206163}

        for key, value in prototype_ref.items():
            assert prototype[key] == value

        for key, value in parameters_ref.items():
            assert np.isclose(parameters[key], value, rtol=1e-5)

    def test3_manganese_oxide(self):
        filename = path + '/Mn3O4_mp-18759_conventional_standard.cif'
        atoms = read(filename)

        PC = PrototypeClassification(atoms)

        prototype, parameters = PC.get_classification()

        prototype_ref = {'p_name': 'A3B4_4_bc_h_141',
                         'structure_name': '141_Mn_b_Mn_c_O_h',
                         'spacegroup': 141,
                         'wyckoffs': ['b', 'c', 'h'],
                         'species': ['Mn', 'Mn', 'O']}

        parameters_ref = {'a': 5.87521306,
                          'b': 5.87521306,
                          'c': 9.58660398,
                          'alpha': 90.0,
                          'beta': 90.0,
                          'gamma': 90.0,
                          'yh2': 0.221099,
                          'zh2': 0.88255}

        for key, value in prototype_ref.items():
            print(prototype[key], value)
            assert prototype[key] == value

        for key, value in parameters_ref.items():
            assert np.isclose(parameters[key], value, rtol=1e-5)

    def test4_nickel_oxide(self):
        filename = path + '/NiO_mp-19009_conventional_standard.cif'
        atoms = read(filename)

        PC = PrototypeClassification(atoms)

        prototype, parameters = PC.get_classification()

        prototype_ref = {'p_name': 'AB_4_a_b_225',
                         'structure_name': '225_Ni_a_O_b',
                         'spacegroup': 225,
                         'wyckoffs': ['a', 'b'],
                         'species': ['Ni', 'O']}

        parameters_ref = {'a': 4.2042588,
                          'b': 4.2042588,
                          'c': 4.2042588,
                          'alpha': 90.0,
                          'beta': 90.0,
                          'gamma': 90.0}

        for key, value in prototype_ref.items():
            assert prototype[key] == value

        for key, value in parameters_ref.items():
            assert np.isclose(parameters[key], value, rtol=1e-5)

    def test5_ternary_alloy(self):

        filename = path + '/Mn2CrCo_mp-864955_conventional_standard.cif'
        atoms = read(filename)

        PC = PrototypeClassification(atoms)

        prototype, parameters = PC.get_classification()

        prototype_ref = {'p_name': 'ABC2_4_a_b_c_225',
                         'structure_name': '225_Cr_a_Co_b_Mn_c',
                         'spacegroup': 225,
                         'wyckoffs': ['a', 'b', 'c'],
                         'species': ['Cr', 'Co', 'Mn']}

        parameters_ref = {'a': 5.74961404,
                          'b': 5.74961404,
                          'c': 5.74961404,
                          'alpha': 90.0,
                          'beta': 90.0,
                          'gamma': 90.0}

        for key, value in prototype_ref.items():
            assert prototype[key] == value

        for key, value in parameters_ref.items():
            assert np.isclose(parameters[key], value, rtol=1e-5)

    def test6_primitive(self):
        filename = path + '/Cr2O3_mp-19399_primitive.cif'
        atoms = read(filename)

        PC = PrototypeClassification(atoms)

        prototype, parameters = PC.get_classification()

        prototype_ref = {'p_name': 'A2B3_6_c_e_167',
                         'structure_name': '167_Cr_c_O_e',
                         'spacegroup': 167,
                         'wyckoffs': ['c', 'e'],
                         'species': ['Cr', 'O']}

        parameters_ref = {'a': 5.09424496,
                          'b': 5.09424496,
                          'c': 13.78071749,
                          'alpha': 90,
                          'beta': 90,
                          'gamma': 120,
                          'zc0':  0.150269 + 0.5,
                          'xe1': 0.300642}

        for key, value in prototype_ref.items():
            assert prototype[key] == value

        for key, value in parameters_ref.items():
            assert np.isclose(parameters[key], value, rtol=1e-5)

    def test7_weird_cell(self):

        filename = path + '/IrO_167.cif'

        atoms = read(filename)
        PC = PrototypeClassification(atoms)

        prototype_ref, parameters_ref = PC.get_classification()

        filename = path + '/000__id-unique_8p8evt9pcg__id-short_207.cif'

        atoms = read(filename)
        PC = PrototypeClassification(atoms, tolerance=0.09)

        prototype, parameters = PC.get_classification()

        for key, value in prototype_ref.items():
            assert prototype[key] == value


if __name__ == '__main__':
    #P = PrototypeClassificationTest()
    # P.test1_cromium_oxide()
    # P.test2_anatase()
    # P.test3_manganese_oxide()
    # P.test4_nickel_oxide()
    # P.test5_ternary_alloy()
    # P.test6_primitive()
    # P.test7_weird_cell()
    unittest.main()
