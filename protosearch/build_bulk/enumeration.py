import sys
import string
import json
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import bulk_enumerator as be
from ase.symbols import string2symbols

from protosearch.build_bulk.build_bulk import BuildBulk
from protosearch.build_bulk.cell_parameters import CellParameters
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
        for SG in range(self.SG_start, self.SG_end + 1):
            with PrototypeSQL(filename=filename) as DB:
                enumerated = [DB.is_enumerated(self.stoichiometry,
                                               SG, num, self.num_type)
                              for num in range(self.num_start, self.num_end + 1)]
                if np.all(enumerated):
                    print('spacegroup={} already enumerated'.format(SG))
                    continue
            E = be.enumerator.ENUMERATOR()
            try:
                enumerations = E.get_bulk_enumerations(
                    self.stoichiometry,
                    self.num_start,
                    self.num_end,
                    SG,
                    SG,
                    self.num_type)

                print('Found {} prototypes for spacegroup={}'.format(
                    len(enumerations), SG))
            except:
                print('Found 0 prototypes for spacegroup={}'.format(SG))

            with PrototypeSQL(filename=filename) as DB:
                for entry in enumerations:
                    DB.write_prototype(entry=entry)

                for num in range(self.num_start, self.num_end + 1):
                    DB.write_enumerated(self.stoichiometry, SG, num,
                                        self.num_type)


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


class AtomsEnumeration():
    """
    Enumeratate atomic structures based on prototype and elements

    elements: dict
        {'A': 'metals', 'B': ['O', 'F'], 'C': 'all'}
    """

    def __init__(self,
                 elements,
                 max_atoms=None):
        self.elements = elements
        self.max_atoms = max_atoms

        for key, value in self.elements.items():
            self.elements[key] = map_elements(value)

    def store_atom_enumeration(self, filename=None, multithread=False):
        self.filename = filename
        DB = PrototypeSQL(filename=filename)
        DB._connect()
        N0 = DB.ase_db.count()

        if self.max_atoms:
            prototypes = DB.select(max_atoms=self.max_atoms)
        else:
            prototypes = DB.select()
        Nprot = len(prototypes)
        pool = Pool()

        import time
        t0 = time.time()
        if multithread:
            res = pool.amap(self.store_atoms_for_prototype, prototypes)
            while not res.ready():
                N = DB.ase_db.count() - N0
                t = time.time() - t0
                N_per_t = N / t
                if N > 0:
                    print('---------------------------------')
                    print(
                        "{}/{} structures generated in {:.2f} sec".format(N, Nprot, t))
                    print("{} sec / structure".format(t / N))
                    print('Estimated time left: {:.2f} min'.format(
                        Nprot / N_per_t / 60))
                print('---------------------------------')
                time.sleep(10)
            res = res.get()
        else:
            for prototype in prototypes:
                self.store_atoms_for_prototype(prototype)

    def store_atoms_for_prototype(self, prototype):

        species_lists = self.get_species_lists(
            prototype['species'], prototype['permutations'])

        cell_parameters = prototype.get('cell_parameters', None)
        if cell_parameters:
            cell_parameters = json.load(cell_parameters)
        for species in species_lists:
            structure_name = str(prototype['spacegroup'])
            for spec, wy_spec in zip(species, prototype['wyckoffs']):
                structure_name += '_{}_{}'.format(spec, wy_spec)
            with PrototypeSQL(filename=self.filename) as DB:
                if DB.ase_db.count(structure_name=structure_name) > 0:
                    continue

                for row in DB.ase_db.select(p_name=prototype['name'], limit=1):
                    cell_parameters = json.loads(row.cell_parameters)

            BB = BuildBulk(prototype['spacegroup'],
                           prototype['wyckoffs'],
                           species,
                           )
            atoms_list, parameters = \
                BB.get_wyckoff_candidate_atoms(return_parameters=True)

            key_value_pairs = {'p_name': prototype['name'],
                               'spacegroup': prototype['spacegroup'],
                               'wyckoffs':
                               json.dumps(prototype['wyckoffs']),
                               'species': json.dumps(species),
                               'structure_name': structure_name,
                               'relaxed': 0,
                               'completed': 0,
                               'submitted': 0}

            for i, atoms in enumerate(atoms_list):

                key_value_pairs.update(
                    {'cell_parameters': json.dumps(parameters[i])})
                BB.get_atoms(cell_parameters)
                atoms = CP.construct_atoms(fix_parameters=parameters,
                                           primitive_cell=True)
                if atoms:
                    with PrototypeSQL(filename=self.filename) as DB:
                        DB.ase_db.write(atoms, key_value_pairs)
                else:
                    print('no atoms?')
            sys.exit()

    def get_species_lists(self, gen_species, permutations):
        elements = self.elements

        alph = list(string.ascii_uppercase)

        name_map = {'A': 'A0',
                    'B': 'A1',
                    'C': 'A2',
                    'D': 'A3',
                    'E': 'A4'}

        N_unique = len(set(gen_species))

        N_species = len(list(elements.keys()))

        c_list = [[]]
        for n in range(N_unique):
            c_list_0 = c_list.copy()
            c_list = []
            for c in c_list_0:
                for e in elements[alph[n]]:
                    if not e in c:
                        c_list += [c + [e]]

        """
        permutations_list = [p.split('_') for p in permutations]

        perm0 = permutations_list[0]
        duplicate_list = []
        for perm in permutations_list[1:]:
            indices = [perm.index(p) for p in perm0]            
            for i, c in enumerate(c_list):
                c_perm = list(np.array(c)[indices])
                print(c, c_perm)
                if c_perm in c_list[:i]:
                    duplicate_list += [i]
            
        c_list = [c for i, c in enumerate(c_list) if i not in duplicate_list]
        """

        species_list = []

        for c in c_list:
            temp = []
            for g in gen_species:
                for i, cc in enumerate(c):
                    g = g.replace(name_map[alph[i]], cc)
                temp += [g]
            species_list += [temp]

        return species_list

    def get_formulas(self,
                     stoichiometries=None,
                     max_atoms=12):
        """
        obtain all possible formulas for a
        supplied set of elements.

        stoichiometries:  list or None
            f.ex. ['1_2', '1_3', '1_2_2']
        max_atoms: int
        """
        elements = self.elements
        if not stoichiometries:
            """Get all possible combinations where N_A <= N_B etc. and
            sum(N) <= max_atoms"""

            N_species = len(list(elements.keys()))
            n_max = max_atoms - N_species
            s_dict = {'0': [[i] for i in range(1, n_max // N_species + 1)]}
            dim = 1
            while dim < N_species:
                s_dict.update({str(dim): []})
                for s_list in s_dict[str(dim - 1)]:
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
                element_list = elements[alph[i]]
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


def get_stoich_from_formulas(formulas):
    alphabet = ['A', 'B', 'C', 'D', 'E']
    stoichs = []
    elements = {}
    for formula in formulas:
        stoich, unique_symbols = formula2stoich(formula)
        stoichs += [stoich]
        for i, us in enumerate(unique_symbols):
            if not alphabet[i] in elements:
                elements[alphabet[i]] = []
            elements[alphabet[i]] += [us]
    stoichs = list(set(stoichs))
    return stoichs, elements


def formula2stoich(formula):
    symbols = string2symbols(formula)
    unique_symbols = list(set(symbols))
    count = [symbols.count(us) for us in unique_symbols]
    idx = np.argsort(count)
    count = [count[i] for i in idx]
    unique_symbols = [unique_symbols[i] for i in idx]
    stoich = '_'.join([str(c) for c in count])
    return stoich, unique_symbols
