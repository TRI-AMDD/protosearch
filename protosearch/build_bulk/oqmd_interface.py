"""Interace to OQMD data to create structurally unique atoms objects.


Author(s): Raul A. Flores; Kirsten Winther
"""

#| - Import Modules
from ase.db import connect
import sqlite3

import copy
import pandas as pd
import numpy as np

from ast import literal_eval
from protosearch.build_bulk.cell_parameters import CellParameters
#__|

class OqmdInterface:

    def __init__(self, dbfile):
        """Set up OqmdInterface."""
        #| - __init__
        self.dbfile = dbfile

        #__|

    def create_proto_data_set(self,
        formula,
        elements,
        verbose=False,
        ):
        """Create a dataset of unique prototype structures.

        Creates a unique set of structures of uniform stoicheometry and
        composition by substituting the desired elemetns into the dataset of
        unique OQMD structures.

        Parameters
        ----------

        formula: str
          desired stoicheometry of data set(ex. 'AB2' or 'Ab2C3')


        elements: list
          list of desired elements to substitute into the dataset

          NOTE: The order of the elements list must correspond to the ordered
          stoicheometry, such that the first element from the elements list
          will replace into the most frequenct term of the desired chemical
          formula.

          Ex.)

            B is the more frequent term, followed by A
            AB2 --> A1 B2

            Fe is the first element given in the user defined list, so it will
            replace the most frequent term in the formula (B in this case)
            elements = [Fe, C]

            A1 B2 -----> CF2


        verbose: bool
          Switches verbosity of module (not implemented fully now)
        """
        #| - create_proto_data_set
        dbfile = self.dbfile

        #| - Argument Checker
        if verbose:
            print("Checking arguments")

        assert type(formula) == str, "Formula must be given as a string"
        assert type(elements) == list, "elements must be given as a list"
        #__|

        if verbose:
            print("Reading text_key_values table from sql")

        con = sqlite3.connect(dbfile)
        # df_systems = pd.read_sql_query("SELECT * FROM systems", con)
        df_text_key_values = pd.read_sql_query(
            "SELECT * FROM text_key_values", con)

        if verbose:
            print("Generating protoname --> structure_id dataframe")

        df_protoid_to_strucid = self.__get_protoid_to_strucid__(
            df_text_key_values=df_text_key_values,
            user_stoich=formula,
            data_file_path=dbfile)

        struct_ids_of_interest = list(np.concatenate(
            df_protoid_to_strucid["id_list"].tolist()
            ).ravel())


        # #####################################################################
        # #####################################################################
        # #####################################################################

        if verbose:
            print("Reading systems table from sql")

        # Read table from sql
        dbfile = self.dbfile
        con = sqlite3.connect(dbfile)
        df_systems = pd.read_sql_query("SELECT * FROM systems", con)

        df_systems = df_systems.set_index("id", drop=True)


        #| - Drop unnecessary Columns
        cols_to_drop = [
            #  'id',
            #  'unique_id',
            'ctime',
            'mtime',
            'username',
            'numbers',
            'positions',
            'cell',
            'pbc',
            'initial_magmoms',
            'initial_charges',
            'masses',
            'tags',
            'momenta',
            'constraints',
            'calculator',
            'calculator_parameters',
            'energy',
            'free_energy',
            'forces',
            'stress',
            'dipole',
            'magmoms',
            'magmom',
            'charges',
            #  'key_value_pairs',
            #  'data',
            'natoms',
            'fmax',
            'smax',
            'volume',
            'mass',
            'charge',
            ]

        df_systems = df_systems.drop(cols_to_drop, 1)
        #__|

        # Drop rows that aren't of the user defined subset
        df_systems = df_systems.loc[struct_ids_of_interest]

        # TEMP Sampling DF down for testing purposes
        # df_systems = df_systems.sample(
        #     frac=0.3,
        #     # n=5,
        #     )

        if verbose:
            print("Processing systems dataframe to generate Enumerator info")

        data_list = []
        for i_cnt, row_i in df_systems.iterrows():

            #| - Processing 'key_value_pairs' column
            key_value_pairs_str = row_i["key_value_pairs"]
            key_value_pairs_str = key_value_pairs_str.replace(
                " NaN,",
                ' " NaN",')

            try:
                key_value_pairs_dict = literal_eval(key_value_pairs_str)

            except:
                print("This error shouldn't be happening!!!! 98ufs8")
                print(key_value_pairs_str)

                key_value_pairs_dict = {}

            keys_to_delete = [
                'name',
                'directory',
                'energy_pa',
                'volume',
                'bandgap',
                'delta_e',
                'stability',

                # 'proto_name',
                # 'spacegroup',
                # 'source',
                ]

            for key_i in keys_to_delete:
                key_value_pairs_dict.pop(key_i, None)
            #__|

            #| - Processing 'Data' column
            data_str = row_i["data"]
        #     key_value_pairs_str = key_value_pairs_str.replace(" NaN,", ' " NaN",')

            data_dict = literal_eval(data_str)

            prototype_params = literal_eval(data_dict["param"])
            prototype_species = literal_eval(data_dict["species"])
            prototype_wyckoffs = literal_eval(data_dict["wyckoffs"])

            data_dict = {
                "prototype_params": prototype_params,
                "prototype_species": prototype_species,
                "prototype_wyckoffs": prototype_wyckoffs,
                }
            #__|

            out_dict = {
                "id": row_i.name,
                # "id": row_i["id"],
                **key_value_pairs_dict,
                **data_dict,
                }
            data_list.append(out_dict)

        df_systems = pd.DataFrame(data_list)

        # #####################################################################
        # #####################################################################
        # #####################################################################

        if verbose:
            print("Generating atoms objects")

        data_list = []

        atoms_list_out = []
        groups = df_systems.groupby("proto_name")
        for protoname_i, group_i in groups:

            # COMBAK Here I'm just taking the prototype parameters from the
            # first structure in OQMD, this can be improved with further logic
            # and tests
            row_i = group_i.iloc[0]


            atoms_i = self.__create_atoms_object_with_replacement__(
                row_i,
                user_elems=elements)

            atoms_list_out.append(atoms_i)


            sys_dict_i = {
                "proto_name": row_i["proto_name"],
                "atoms": atoms_i,
                }

            data_list.append(sys_dict_i)


        prototype_atoms_dataframe = pd.DataFrame(data_list)

        return(prototype_atoms_dataframe)

        #__|

    def __get_protoid_to_strucid__(self,
        df_text_key_values=None,
        user_stoich=None,
        data_file_path=None):
        """
        """
        #| - __get_protoid_to_strucid__
        # Getting unique prototype IDs for given constraints
        # unique_prototype_names = self.get_distinct_prototypes(
        unique_protonames_for_user = self.get_distinct_prototypes(
            source=None,
            formula=user_stoich,
            repetition=None)

        print(
            "There are ",
            str(len(unique_protonames_for_user)),
            " unique prototypes for the",
            str(user_stoich), " stoicheometry",
            " and source of",
            # str(source),
            )

        # My method to get the unique prototype ids
        # Getting unique prototype names
        # unique_prototype_names = df_text_key_values[
        #     df_text_key_values["key"] == "proto_name"]["value"].unique()

        df_tmp = df_text_key_values[
            df_text_key_values["key"] == "proto_name"]
        df_tmp1 = df_tmp[df_tmp["value"].isin(unique_protonames_for_user)]

        data_list = []
        group = df_tmp1.groupby(["value"])
        for protoname_i, df_i in group:
            data_list.append({
                "id_list": df_i["id"].tolist(),
                "protoname": protoname_i,
                })

        out_df = pd.DataFrame(data_list)

        return(out_df)
        #__|

    def __create_atoms_object_with_replacement__(self,
        indiv_data_tmp_i,
        user_elems=None):
        """

        Parameters
        ----------
        indiv_data_tmp_i: pandas.Series
          Row of dataframe that contains the following keys:
            spacegroup
            prototype_wyckoffs
            prototype_species
            prototype_params

        user_elems: list
        """
        #| - __create_atoms_object_with_replacement__
        spacegroup_i = indiv_data_tmp_i["spacegroup"]
        prototype_wyckoffs_i = indiv_data_tmp_i["prototype_wyckoffs"]
        prototype_species_i = indiv_data_tmp_i["prototype_species"]
        init_params = indiv_data_tmp_i["prototype_params"]

        #| - Atom type replacement
        def CountFrequency(my_list):
            """
            Python program to count the frequency of
            elements in a list using a dictionary
            """
            #| - CountFrequency
            # Creating an empty dictionary
            freq = {}
            for item in my_list:
                if (item in freq):
                    freq[item] += 1
                else:
                    freq[item] = 1

            return(freq)
            #__|

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
        #__|

        #| - Preparing Initial Prototype Parameters
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
        #__|

        #| - Using CellParameters Code
        CP = CellParameters(
            spacegroup=spacegroup_i,
            wyckoffs=prototype_wyckoffs_i,
            species=new_elem_list,
            # species=prototype_species_i,
            )

        parameters = CP.get_parameter_estimate(
            master_parameters=init_wyck_params)
        atoms_opt = CP.get_atoms(fix_parameters=parameters)
        atoms_out = atoms_opt
        #__|

        return(atoms_out)
        #__|

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
        #| - get_distinct_prototypes
        db = connect(self.dbfile)

        con = db.connection or db._connect()
        cur = con.cursor()

        sql_command = \
            "select distinct value from text_key_values where key='proto_name'"
        if formula:
            if repetition:
                formula += '\_{}'.format(repetition)
            sql_command += " and value like '{}\_%' ESCAPE '\\'".format(formula)

        if source:
             sql_command += " and id in (select id from text_key_values where key='source' and value='icsd')"
        cur.execute(sql_command)

        prototypes = cur.fetchall()
        prototypes = [p[0] for p in prototypes]

        return prototypes

        #__|
