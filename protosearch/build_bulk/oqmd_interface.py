"""Interace to OQMD data to create structurally unique atoms objects.


Author(s): Raul A. Flores; Kirsten Winther; Meng Zhao
"""

# Import Modules
from ase.db import connect
from tqdm import tqdm
from pymatgen.core.composition import Composition
from ase.symbols import string2symbols
import string
import copy
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from ast import literal_eval
from protosearch.build_bulk.cell_parameters import CellParameters


class OqmdInterface:

    def __init__(self,
        dbfile,
        verbose=False,
        ):
        """Set up OqmdInterface.

        Parameters
        ----------

        dbfile: str (filepath)
            filepath of oqmd databae file (available upon request from S3)

        verbose: bool
            Switches verbosity of module (not implemented fully now)
        """
        self.dbfile = dbfile
        self.verbose = verbose

    def create_proto_data_set(self,
        chemical_formula=None,
        source=None,
        repetition=None):
        """Create a dataset of unique prototype structures.

        Creates a unique set of structures of uniform stoicheometry and
        composition by substituting the desired elemetns into the dataset of
        unique OQMD structures.

        Parameters
        ----------

        chemical_formula: str
            desired chemical formula with elements inserted (ex. 'Al2O3')

        source: str
          oqmd project name, such as 'icsd'

        repetition: int
          repetition of the stiochiometry
        """
        dbfile = self.dbfile
        verbose = self.verbose

        # Argument Checker
        if chemical_formula is not None:
            mess_i = "Formula must be given as a string"
            assert type(chemical_formula) == str, mess_i
            elem_list, compos, stoich_formula, elem_list_ordered = formula2elem(chemical_formula)
            formula = stoich_formula
            elements = elem_list_ordered
        else:
            raise ValueError("ERROR: Couldn't correctly parse input")

        relev_id_list = self.__get_relevant_ids__(formula, source, repetition)

        db = connect(dbfile)
        data_list = []
        for id_i in relev_id_list:
            row_i = db.get(selection=id_i)

            # Procesing data dict since objects are stored as strings
            # (literal_eval needed)
            data_dict_0 = {}
            for key_i, value_i in row_i.data.items():
                data_dict_0[key_i] = literal_eval(value_i)

            data_dict_1 = {
                "id": row_i.id,
                "protoname": row_i.proto_name,
                "formula": row_i.formula,
                "spacegroup": row_i.spacegroup}

            data_dict_out = {**data_dict_0, **data_dict_1}
            data_list.append(data_dict_out)
        df = pd.DataFrame(data_list)

        data_list = []
        groups = df.groupby("protoname")
        for protoname_i, group_i in tqdm(groups):
            struct_in_db_i = self.__structure_in_database__(
                group_i, chemical_formula, "bool")
            if struct_in_db_i:
                row_i = self.__structure_in_database__(
                    group_i, chemical_formula, "return_structure")

                db_row_i = db.get(selection=int(row_i["id"]))
                atoms_i = db_row_i.toatoms()
            else:
                # Just returning the 'first' structure in the group and doing
                # an atom replacement
                row_i = group_i.iloc[0]
                atoms_i = self.__create_atoms_object_with_replacement__(
                    row_i, user_elems=elements)

            sys_dict_i = {
                "proto_name": row_i["protoname"],
                "atoms": atoms_i,
                "existing_structure": struct_in_db_i}
            data_list.append(sys_dict_i)

        df_out = pd.DataFrame(data_list)

        return(df_out)

    def __structure_in_database__(self, group_i, chemical_formula, mode):
        """Looks for existing structure in the db"""
        group_i["pymatgen_comp"] = group_i.apply(
            lambda x: Composition(x["formula"]), axis=1)
        same_formula = group_i[
            group_i["pymatgen_comp"] == Composition(chemical_formula)]

        if len(same_formula) != 0:
            structure_exists = True
        else:
            structure_exists = False

        if len(same_formula) > 1:
            print("There is more than 1 structure in the database for the",
                " given prototype and chemical formula")
            print("Just using the 'first' one for now")

        if mode == "bool":
            return(structure_exists)
        elif mode == "return_structure":
            return(same_formula.iloc[0])

    def __get_relevant_ids__(self,
        formula,
        source,
        repetition,
        ):
        """
        """
        distinct_protonames = self.get_distinct_prototypes(
            formula=formula,
            source=source,
            repetition=repetition)

        if self.verbose:
            print("Number of unique prototypes: ")
            print(len(distinct_protonames))

        str_tmp = ""
        for i in distinct_protonames:
            str_tmp += "'" + i + "', "
        str_tmp = str_tmp[0:-2]

        db = connect(self.dbfile)

        con = db.connection or db._connect()

        sql_comm = "SELECT value,id FROM text_key_values WHERE key='proto_name' and value in (" + str_tmp + ")"
        df_text_key_values = pd.read_sql_query(
            sql_comm,
            con)

        relev_id_list = df_text_key_values["id"].tolist()
        return(relev_id_list)

    def __create_atoms_object_with_replacement__(self,
        indiv_data_tmp_i,
        user_elems=None):
        """

        Parameters
        ----------
        indiv_data_tmp_i: pandas.Series or dict
          Row of dataframe that contains the following keys:
            spacegroup
            wyckoffs
            species
            param

        user_elems: list
        """
        spacegroup_i = indiv_data_tmp_i["spacegroup"]
        prototype_wyckoffs_i = indiv_data_tmp_i["wyckoffs"]
        prototype_species_i = indiv_data_tmp_i["species"]
        init_params = indiv_data_tmp_i["param"]

        # Atom type replacement
        def CountFrequency(my_list):
            """
            Python program to count the frequency of
            elements in a list using a dictionary
            """
            freq = {}
            for item in my_list:
                if (item in freq):
                    freq[item] += 1
                else:
                    freq[item] = 1

            return(freq)


        elem_count_freq = CountFrequency(prototype_species_i)

        freq_data_list = []
        for key_i, value_i in elem_count_freq.items():
            freq_data_list.append(
                {
                    "element": key_i,
                    "frequency": value_i,
                    }
                )

        elem_mapping_dict = dict(zip(
            pd.DataFrame(freq_data_list).sort_values(
                by=["frequency"])["element"].tolist(),
            user_elems,
            ))

        # Preparing new atom substituted element list
        new_elem_list = []
        for i in prototype_species_i:
            new_elem_list.append(
                elem_mapping_dict.get(i, i))

        # #####################################################################
        # Preparing Initial Wyckoff parameters to pass to the
        # CellParameters Code

        # Removing lattice constant parameters, so CellParameters will only
        # optimize the parameters that aren't included in this list
        # lattice angles and wyckoff positions
        non_wyck_params = [
            "a", "b", "c",
            "b/a", "c/a",
            # "alpha", "beta", "gamma",
            ]

        init_wyck_params = copy.copy(init_params)
        for param_i in non_wyck_params:
            if param_i in list(init_wyck_params.keys()):
                del init_wyck_params[param_i]

        try:
            # Using CellParameters Code
            CP = CellParameters(
                spacegroup=spacegroup_i,
                wyckoffs=prototype_wyckoffs_i,
                species=new_elem_list,
                # species=prototype_species_i,
                verbose=False,
                )

            parameters = CP.get_parameter_estimate(
                master_parameters=init_wyck_params)
            atoms_opt = CP.get_atoms(fix_parameters=parameters)
            out = atoms_opt
        except RuntimeError:
            out = None

        return(out)

    def get_distinct_prototypes(self,
                                source=None,
                                formula=None,
                                repetition=None):
        """ Get list of distinct prototype strings given certain filters.

        Parameters
        ----------
        source: str
          oqmd project name, such as 'icsd'
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
        if formula:
            if repetition:
                formula += '\_{}'.format(repetition)
            sql_comm += " and value like '{}\_%' ESCAPE '\\'".format(formula)

        if source:
             sql_comm += " and id in (select id from text_key_values where key='source' and value='icsd')"
        cur.execute(sql_comm)

        prototypes = cur.fetchall()
        prototypes = [p[0] for p in prototypes]

        return prototypes


# Atom type replacement
def CountFrequency(my_list):
    """
    Python program to count the frequency of
    elements in a list using a dictionary
    """
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    return(freq)


def formula2elem(formula):
    '''
    convert plain chemical formula to element list with their composition and
    the element agnostic chemical formula.

    arg: formula(str)
    return:
    elem_list(list): element list
    compos(dict): composition for elements
    stoich_formula(str): Stoicheometric representation of the chemical formula
    '''
    elem_list = string2symbols(formula)

    compos = {}
    uniq = set(elem_list)
    for symbol in uniq:
        compos.update({symbol: elem_list.count(symbol)})


    # Creating stoicheometric repr of formula (i.e. AB2 not FeO2)
    data_list = []
    for key_i, value_i in compos.items():
        data_list.append(
            {"element": key_i,
             "stoich": value_i})

    df_sorted = pd.DataFrame(data_list).sort_values("stoich")
    df_sorted["fill_symbol"] = list(string.ascii_uppercase[0:len(df_sorted)])

    # List of elements ordered by highest stoich to lowest
    # This method is getting a bit redundant but it's just to assure
    # compatability with the way I wrote the module, can clean up later
    # This includes the stoich_formula variable that I'm creating as well,
    # it's also needed for my method
    elem_list_ordered = list(reversed(df_sorted["element"].tolist()))

    stoich_formula = ""
    for i_ind, row_i in df_sorted.iterrows():
        stoich_formula += str(row_i["fill_symbol"])  # + str(row_i["stoich"])
        if row_i["stoich"] == 1:
            pass
        else:
            stoich_formula += str(row_i["stoich"])

    return(elem_list, compos, stoich_formula, elem_list_ordered)
