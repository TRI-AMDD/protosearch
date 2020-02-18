import shutil
import os
import tempfile
import unittest
import time
import numpy as np
from ase.io import read

from protosearch.build_bulk import tests
from protosearch.build_bulk.classification import PrototypeClassification
from protosearch.build_bulk.build_bulk import BuildBulk
from protosearch.build_bulk.wyckoff_symmetries import get_wyckoff_letters
from protosearch.build_bulk.enumeration import metals


path = list(tests.__path__)[0]


class SpacegroupClassificationTest(unittest.TestCase):
    def test6_all_spacegroups(self):

        images = []
        spacegroups = range(195, 230)

        spacegroups, spacegroup_letters = get_wyckoff_letters(spacegroups)

        for sg in spacegroups:
            wyckoffs = spacegroup_letters[sg-1][::-1][:5]
            species = metals[:len(wyckoffs)]
            print('----------', sg, '------------')

            BB = BuildBulk(sg, wyckoffs, species)
            atoms = BB.get_atoms(primitive_cell=False)
            
            PC = PrototypeClassification(atoms)

            prototype = PC.get_classification(include_parameters=False)

            if not prototype['spacegroup'] == sg:
                print('  change in spacegroup', sg,
                      '----->', prototype['spacegroup'])
                print('    ', prototype['p_name'])
            else:
                assert prototype['wyckoffs'] == wyckoffs, \
                    '{} --->> {}'.format(prototype['wyckoffs'], wyckoffs)


if __name__ == '__main__':
    unittest.main()
