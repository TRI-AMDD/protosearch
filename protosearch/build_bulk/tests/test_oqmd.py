import sys
import unittest
import ase

from protosearch.build_bulk.oqmd_interface import OqmdInterface


class OqmdTest(unittest.TestCase):
    def test_unique_prototypes(self):
        O = OqmdInterface()
        p = O.get_distinct_prototypes(chemical_formula='NiCuCN',
                                      max_atoms=8)

        assert(len(p) == 19)
        print('{} distinct prototypes found'.format(len(p)))

    def test_create_proto_dataset(self):
        O = OqmdInterface()
        atoms_list = O.create_proto_data_set(chemical_formula='FeO6',
                                             max_atoms=7)

        assert len(atoms_list) == 5

        for atoms in atoms_list["atoms"][:5]:
            assert atoms.get_number_of_atoms() == 7
            assert atoms.get_chemical_symbols().count('Fe') == 1
            assert atoms.get_chemical_symbols().count('O') == 6

    def test_get_same_formula(self):
        # Should get same formula if exists
        O = OqmdInterface()

        atoms_data = O.get_atoms_for_prototype(chemical_formula='TiO2',
                                               proto_name='AB2_2_a_f_136',
                                               max_candidates=1)[0]
        assert atoms_data['chemical_formula'] == atoms_data['original_formula']

    def test_unique(self):
        # AB == BA  - one structure
        O = OqmdInterface()
        atoms_data = O.get_atoms_for_prototype(chemical_formula='TiMo',
                                               proto_name='AB_4_ab_ab_186',
                                               max_candidates=1)
        atoms = [a['atoms'] for a in atoms_data]

        assert len(atoms) == 1

        # ABC3 != BAC3  - two structures
        atoms_data = O.get_atoms_for_prototype(chemical_formula='TiMoO3',
                                               proto_name='ABC3_1_a_a_b_160',
                                               max_candidates=1)
        atoms = [a['atoms'] for a in atoms_data]

        assert len(atoms) == 2

    def test_substitute_atoms(self):
        O = OqmdInterface()

        atoms_data = O.get_atoms_for_prototype(chemical_formula='TiMoO2',
                                               proto_name='ABC2_2_a_c_f_194',
                                               max_candidates=1)[0]
        atoms = atoms_data['atoms']

        atoms_list = O.substitute_atoms(
            atoms, new_symbols=['Nb', 'V', 'O', 'O'])

        assert len(atoms_list) == 2

    def test_store_enumeration(self):
        O = OqmdInterface()
        O.store_enumeration(filename='test.db',
                            chemical_formula='FeO6',
                            max_atoms=7)


if __name__ == '__main__':
    unittest.main()
