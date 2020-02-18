import shutil
import os
import tempfile
import unittest
import numpy as np

from ase.visualize import view

from protosearch.build_bulk.build_bulk import BuildBulk


class BuildBulkTest(unittest.TestCase):

    def test1_no_parameters(self):
        BB = BuildBulk(225, ['a'], ["Cu"])
        atoms = BB.get_atoms(proximity=1)
        unit_cell_lengths = np.linalg.norm(atoms.cell, axis=1)
        assert np.all(np.isclose(unit_cell_lengths, 2.629, rtol=0.01))

    def test2_set_parameters(self):
        BB = BuildBulk(225, ['a'], ["Cu"])

        atoms = BB.get_atoms(cell_parameters={'a': 5})
        unit_cell_lengths = np.linalg.norm(atoms.cell, axis=1)

        assert np.all(np.isclose(unit_cell_lengths, 5 / np.sqrt(2)))

        atoms = BB.get_atoms(cell_parameters={'a': 5}, primitive_cell=False)
        unit_cell_lengths = np.linalg.norm(atoms.cell, axis=1)

        assert np.all(np.isclose(unit_cell_lengths, 5))


if __name__ == '__main__':
    unittest.main()
