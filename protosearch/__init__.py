import sys
import os

from protosearch.build_bulk.oqmd_interface import OqmdInterface

__author__ = "TRI Materials"
__version__ = '0.0.1'


def start_learning(formula, max_atoms=None, source='oqmd'):

    if source == 'oqmd':
        StructureMaster = OqmdInterface
        atoms_list = StructureMaster.get_structures(
            formula, source='icsd', max_atoms=None)

    else:
        # Enumerate from prototypes
        atoms_list = []
