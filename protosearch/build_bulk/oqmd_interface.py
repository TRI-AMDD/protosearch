import numpy as np
import json
import string
from ase import Atoms
from ase.db import connect
from ase.formula import Formula
from ase.symbols import string2symbols
import pandas as pd

from protosearch.utils.data import metal_numbers
from protosearch.workflow.prototype_db import PrototypeSQL
from protosearch import build_bulk
from .fitness_function import get_connections
from .spglib_interface import SpglibInterface
from .cell_parameters import CellParameters


path = build_bulk.__path__[0]


class OqmdInterface:
    """Interace to OQMD data to create structurally unique atoms objects"""

    def __init__(self, source='icsd'):
        if source == 'icsd':
            self.dbfile = path + '/oqmd_icsd.db'
        else:
            raise NotImplementedError('Only ICSD structure source impemented')

    def create_proto_data_set(self,
                              chemical_formula,
                              max_atoms=None):
        """Create a dataset of unique prototype structures.

        Creates a unique set of structures of uniform stoicheometry and
        composition by substituting the desired elemetns into the dataset of
        unique OQMD structures.

        Parameters
        ----------
        chemical_formula: str
            desired chemical formula with elements inserted (ex. 'Al2O3')

        max_atoms: int
          maximum number of atoms in the primitive unit cell
        """

        distinct_protonames = \
            self.get_distinct_prototypes(
                chemical_formula=chemical_formula,
                max_atoms=max_atoms)

        data_list = []

        for proto_name in distinct_protonames:
            data_list += self.get_atoms_for_prototype(chemical_formula,
                                                      proto_name)

        df = pd.DataFrame(data_list)

        return df

    def store_enumeration(self, filename, chemical_formula, max_atoms=None):
        """ Saves the enumerated prototypes and atomic species in a database"""

        distinct_protonames = \
            self.get_distinct_prototypes(
                chemical_formula=chemical_formula,
                max_atoms=max_atoms)

        for proto_name in distinct_protonames:
            data_list = self.get_atoms_for_prototype(chemical_formula,
                                                     proto_name)

            DB = PrototypeSQL(filename=filename)

            for d in data_list:
                atoms = d.pop('atoms')

                formula = d.pop('chemical_formula')
                original_formula = d.pop('original_formula')

                # Same format as Enumerator output
                entry = d.copy()
                entry['spaceGroupNumber'] = entry.pop('spacegroup')
                entry['name'] = entry.pop('p_name')
                entry['parameters'] = {}

                structure_name = d['structure_name']

                print('Writing structure:', structure_name)

                # Save prototype
                DB.write_prototype(entry=entry)

                if DB.ase_db.count(structure_name=structure_name) > 0:
                    continue

                key_value_pairs = \
                    {'p_name': d['p_name'],
                     'spacegroup': d['spacegroup'],
                     'permutations': json.dumps(d['specie_permutations']),
                     'wyckoffs': json.dumps(d['wyckoffs']),
                     'species': json.dumps(d['species']),
                     'structure_name': structure_name,
                     'relaxed': 0,
                     'completed': 0,
                     'submitted': 0}

                # Save structure
                DB.ase_db.write(atoms, key_value_pairs)

    def get_atoms_for_prototype(self,
                                chemical_formula,
                                proto_name,
                                fix_metal_ratio=False,
                                must_contain_nonmetal=False,
                                max_candidates=1):

        oqmd_db = connect(self.dbfile)

        atoms0 = Atoms(chemical_formula)
        numbers0 = atoms0.numbers
        symbols0 = atoms0.get_chemical_symbols()

        metal_count = len([n for n in numbers0 if n in metal_numbers])
        metal_ratio_0 = metal_count / len(numbers0)
        nonmetals_idx = [i for i, n in enumerate(numbers0)
                         if not n in metal_numbers]
        nonmetals_symbols = ''.join(np.array(symbols0)[nonmetals_idx])

        if must_contain_nonmetal:
            structures = list(oqmd_db.select(nonmetals_symbols,
                                             proto_name=proto_name))
        else:
            structures = list(oqmd_db.select(proto_name=proto_name))

        if len(structures) == 0:
            return []

        formulas = [Formula(s.formula).reduce()[0].format('metal')
                    for s in structures]
        chemical_formula = Formula(chemical_formula).format('metal')
        if chemical_formula in formulas:
            idx = formulas.index(chemical_formula)
            structures = structures[idx:] + structures[:idx]

        atoms_data = []

        for s in structures:
            atoms = s.toatoms()
            old_species = np.array(s.data['species'], dtype='U2')
            old_symbols = atoms.get_chemical_symbols()
            orig_formula = Formula(atoms.get_chemical_formula()).reduce()[
                0].format('metal')
            metal_count = len([n for n in atoms.numbers if n in metal_numbers])
            metal_ratio = metal_count / len(atoms)
            if fix_metal_ratio and not metal_ratio == metal_ratio_0:
                continue

            atoms_sub_list = self.substitute_atoms(atoms, symbols0)
            graphs = []
            for atoms in atoms_sub_list:
                new_symbols = atoms.get_chemical_symbols()
                new_species = old_species.copy()
                for i in range(len(new_symbols)):
                    os = old_symbols[i]
                    idx = [i for i, o in enumerate(old_species) if o == os]
                    if idx:
                        new_species[idx] = [new_symbols[i]] * len(idx)


                graphs += [get_connections(atoms, decimals=1)]
                # Get spacegroup and conventional atoms
                SPG = SpglibInterface(atoms)
                atoms = SPG.get_conventional_atoms()
                spacegroup = SPG.get_spacegroup()

                structure_name = str(spacegroup)
                for spec, wy_spec in zip(new_species, s.data['wyckoffs']):
                    structure_name += '_{}_{}'.format(spec, wy_spec)

                # Set new lattice constants
                CP = CellParameters(spacegroup=spacegroup)
                atoms = CP.optimize_lattice_constants(atoms)

                # Get primitive atoms
                atoms = SPG.get_primitive_atoms(atoms)

                atoms_data += [{'spacegroup': spacegroup,
                                'wyckoffs': s.data['wyckoffs'],
                                'species': list(new_species),
                                'structure_name': structure_name,
                                'natom': len(atoms),
                                'specie_permutations': s.data['permutations'],
                                'p_name': proto_name,
                                'chemical_formula': chemical_formula,
                                'original_formula': orig_formula,
                                'atoms': atoms.copy()}]

            if len(atoms_data) >= max_candidates:
                break

        #graphs = [get_connections(data['atoms'], decimals=1)
        #          for data in atoms_data]

        indices = [i for i in range(len(atoms_data))
                   if not np.any(graphs[i] in graphs[:i])]

        graphs = [graphs[i] for i in indices]
        atoms_data = [atoms_data[i] for i in indices]

        return atoms_data

    def substitute_atoms(self, atoms, new_symbols):
        """ Substitute new elements into atoms object"""

        formula = atoms.get_chemical_formula()
        rep = Formula(formula).reduce()[1]

        chemical_symbols = np.array(atoms.get_chemical_symbols(), dtype='U2')

        unique_old, counts_old = np.unique(
            chemical_symbols, return_counts=True)
        counts_old = counts_old / rep

        idx = np.argsort(counts_old)
        counts_old = counts_old[idx]
        unique_old = unique_old[idx]
        unique_new, counts_new = np.unique(new_symbols, return_counts=True)

        idx = np.argsort(counts_new)
        counts_new = counts_new[idx]
        unique_new = unique_new[idx]

        new_perm_temp = [[]]

        for i, c in enumerate(counts_new):
            perm_temp = []
            same_count = np.where(counts_new == c)[0]

            new_perm = []
            for temp in new_perm_temp:
                for i2 in same_count:
                    sym = unique_new[i2]
                    if not sym in temp:
                        new_perm += [temp + [sym]]

            new_perm_temp = new_perm.copy()

        old_symbols = chemical_symbols.copy()

        atoms_list = []
        for unique_new in new_perm:
            atoms_temp = atoms.copy()

            for i, old_s in enumerate(unique_old):
                loc_symbols = [i for i, s in enumerate(
                    old_symbols) if s == old_s]
                chemical_symbols[loc_symbols] = np.repeat(
                    [unique_new[i]], len(loc_symbols))

            atoms_temp.set_chemical_symbols(chemical_symbols)
            atoms_list += [atoms_temp.copy()]

        return atoms_list

    def ase_db(self):
        self.db = connect(self.dbfile)

    def get_distinct_prototypes(self,
                                chemical_formula=None,
                                max_atoms=None):
        """ Get list of distinct prototype strings given certain filters.

        Parameters
        ----------
        formula: str
          stiochiometry of the compound, f.ex. 'AB2' or AB2C3
        repetition: int
          repetition of the stiochiometry
        """
        db = connect(self.dbfile)
        con = db.connection or db._connect()
        cur = con.cursor()

        sql_comm = \
            "select distinct value from text_key_values where key='proto_name'"

        if chemical_formula is None:
            cur.execute(sql_comm)
            prototypes_rep = cur.fetchall()
            prototypes += [p[0] for p in prototypes_rep]

            return prototypes

        # Transform formula to general 'AB2C3' format
        formula, elements = get_stoich_formula(chemical_formula)

        repetitions = [0]
        if max_atoms is not None:
            natoms_formula = len(Atoms(chemical_formula))
            r = 1
            while (repetitions[-1] + 1) * natoms_formula <= max_atoms:
                repetitions += [r]
                r += 1
            del repetitions[0]

        prototypes = []
        for r in repetitions:
            if r == 0:  # all repetitions
                match_formula = formula
            else:
                match_formula = formula + '\_{}'.format(r)
            full_sql_comm = sql_comm + \
                " and value like '{}\_%' ESCAPE '\\'".format(match_formula)

            cur.execute(full_sql_comm)

            prototypes_rep = cur.fetchall()
            prototypes += [p[0] for p in prototypes_rep]

        return prototypes


def get_stoich_formula(formula, return_elements=True):
    """Get element independent stoichiometry formula
    i.e. TiO2 -> AB2"""
    alphabet = list(string.ascii_uppercase)
    elem_list = string2symbols(formula)

    unique_symbols, counts = np.unique(elem_list, return_counts=True)

    sorted_idx = np.argsort(counts)

    formula = ''
    for i, j in enumerate(sorted_idx):
        if counts[j] > 1:
            formula += alphabet[i] + str(counts[j])
        else:
            formula += alphabet[i]
    if return_elements:
        return formula, unique_symbols
    else:
        return formula
