import string
import numpy as np
import bulk_enumerator as be

from protosearch.workflow.prototype_db import PrototypeSQL


class Enumeration():

    def __init__(self,
                 stoichiometry,
                 num_start,
                 num_end,
                 SG_start=1,
                 SG_end=230,
                 num_type='atom'):
        """
        Parameters

        stoichiometry: str
            Ratio bewteen elements separated by '_'. 
            For example: '1_2' or '1_2_3'

        num_start: int
            minimum number of atoms or wyckoff sites
        num_end: int
            maximum number of atoms or wyckoff sites
        SG_start: int
           minimum spacegroup number
        SG_end: int
           maximum spacegroup number
        num_type: str 
            'atom' or 'wyckoff'
        """

        self.stoichiometry = stoichiometry
        self.num_start = num_start
        self.num_end = num_end
        self.SG_start = SG_start
        self.SG_end = SG_end
        self.num_type = num_type

    def set_spacegroup(self, spacegroup):
        self.SG_start = spacegroup
        self.SG_end = spacegroup + 1

    def set_stoichiometry(self, stoichiometry):
        self.stoichiometry = stoichiometry

    def set_natoms(self, natoms):
        self.num_start = natoms
        self.num_end = natoms

    def get_enumeration(self):
        E = be.enumerator.ENUMERATOR()
        enumerations = E.get_bulk_enumerations(self.stoichiometry,
                                               self.num_start,
                                               self.num_end,
                                               self.SG_start,
                                               self.SG_end,
                                               self.num_type)

        return enumerations

    def store_enumeration(self, filename=None):

        enumerations = self.get_enumeration()

        with PrototypeSQL(filename=filename) as DB:
            for entry in enumerations:
                DB.write_prototype(entry=entry)


all_elements = [
    'H', 'He',
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn']

non_metals = ['H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
              'Si', 'P', 'S', 'Cl', 'Ar',
              'Ge', 'As', 'Se', 'Br', 'Kr',
              'Sb', 'Te', 'I', 'Xe',
              'Po', 'At', 'Rn']

metals = [
    'Li', 'Be',
    'Na', 'Mg', 'Al',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
]


class OqmdEnumeration():
    """
    Enumerator class to obtain all possible formulas for a
    supplied set of elements.
    """

    def __init__(self):
        pass

    def get_formulas(self,
                     elements,
                     stoichiometries=None,
                     max_atoms=12):
        """
        elements: dict
            {'A': 'metals', 'B': ['O', 'F'], 'C': 'all'}
        stoichiometry:  str or None
            f.ex. '1_2', '1_3' or '1_2_2'
        max_atoms: int
        """

        if not stoichiometries:
            """Get all possible combinations where N_A <= N_B etc. and
            sum(N) <= max_atoms"""

            N_species = len(list(elements.keys()))
            n_max = max_atoms - N_species
            s_dict = {'0': [[i] for i in range(1, n_max // N_species + 1)]}
            dim = 1
            while dim < N_species:
                s_dict.update({str(dim): []})
                for s_list in enum_dict[str(dim - 1)]:
                    n_atoms = sum(s_list)
                    i = s_list[-1]
                    for j in [j for j in range(1, max_atoms) if j >= i]:
                        if n_atoms + j > max_atoms:
                            continue
                        s_dict[str(dim)] += [s_list + [j]]
                dim += 1

            stoichiometries = []
            for s_list in s_dict[str(dim - 1)]:
                stoichiometries += ['_'.join([str(s) for s in s_list])]

        alph = list(string.ascii_uppercase)

        all_formulas = []
        for stoichiometry in stoichiometries:
            assert len(elements.keys()) == \
                len(stoichiometry.split('_'))
            formulas = np.array([])
            for i, value in enumerate(stoichiometry.split('_')):
                element_list = map_elements(elements[alph[i]])
                if value == '1':
                    value = ''
                element_list = np.char.array([e + value for e in element_list])
                if len(formulas) == 0:
                    formulas = element_list
                else:
                    formulas = np.expand_dims(formulas, axis=i).\
                        repeat(len(element_list), axis=i)
                    for k in range(i):
                        append_formulas = np.expand_dims(element_list, axis=k).\
                            repeat(formulas.shape[k], axis=k)
                    formulas = formulas + append_formulas
            all_formulas += list(formulas.flatten())
        return all_formulas


def map_elements(key):
    if isinstance(key, list):
        assert np.all([k in all_elements for k in key])
        return key

    key_to_elements = {'metals': metals,
                       'non_metals': non_metals,
                       'all': all_elements}

    return key_to_elements[key]
