"""
190419 | Test change RF
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

        df_protoid_to_strucid = self.get_protoid_to_strucid(
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

        # TEMP
        df_systems = df_systems.sample(
            # frac=0.3,
            n=5,
            )

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

            spacegroup = key_value_pairs_dict["spacegroup"]
            proto_name = key_value_pairs_dict["proto_name"]


            keys_to_delete = [
                #  'proto_name',
                #  'spacegroup',
                 'name',
                 'directory',
                 'energy_pa',
                 'volume',
                 'bandgap',
                 'delta_e',
                 'stability',
                #  'source',
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
            row_i = group_i.iloc[0]

            print(row_i)


            #| - Try-except rationale
            # Try - Except is a hacky workaround some bugs in the Enumerator code

            # Error from Enumerator Code
            # ---------------------------------------------------------------------
            # Not able to find value for parameter: yc0
            # Lattice not set!
            # Lattice not set!

            # Full Trace-Traceback
            # ---------------------------------------------------------------------------
            # IndexError                                Traceback (most recent call last)
            # <ipython-input-4-59c86e447fcb> in <module>
            #       1 user_elems = ["O", "Al"]
            # ----> 2 df_m = DB_inter.create_proto_data_set("AB2", user_elems)
            #
            # /mnt/c/Users/raulf/github/protosearch/protosearch/build_bulk/oqmd_interface.py in create_proto_data_set(self, formula, elements)
            #     221             atoms_i = self.create_atoms_object_with_replacement_tmp(
            #     222                 row_i,
            # --> 223                 user_elems=elements)
            #     224
            #     225             atoms_list_out.append(atoms_i)
            #
            # /mnt/c/Users/raulf/github/protosearch/protosearch/build_bulk/oqmd_interface.py in create_atoms_object_with_replacement_tmp(self, indiv_data_tmp_i, user_elems)
            #     629         parameters = CP.get_parameter_estimate(
            #     630             master_parameters=init_wyck_params)
            # --> 631         atoms_opt = CP.get_atoms(fix_parameters=parameters)
            #     632         atoms_out = atoms_opt
            #     633         #__|
            #
            # /mnt/c/Users/raulf/github/protosearch/protosearch/build_bulk/cell_parameters.py in get_atoms(self, fix_parameters)
            #     154         self.b.set_parameter_values(self.parameters, parameter_guess_values)
            #     155         poscar = self.b.get_primitive_poscar()
            # --> 156         self.atoms = read_vasp(io.StringIO(poscar))
            #     157
            #     158         return self.atoms
            #
            # ~/anaconda2/envs/py36/lib/python3.6/site-packages/ase/io/vasp.py in read_vasp(filename)
            #     124     line1 = f.readline()
            #     125
            # --> 126     lattice_constant = float(f.readline().split()[0])
            #     127
            #     128     # Now the lattice vectors
            #
            # IndexError: list index out of range
            #__|

            try:
                atoms_i = self.create_atoms_object_with_replacement_tmp(
                    row_i,
                    user_elems=elements)

                atoms_list_out.append(atoms_i)


                sys_dict_i = {
                    "proto_name": row_i["proto_name"],
                    "atoms": atoms_i,
                    }

                data_list.append(sys_dict_i)

            except:
                pass

        prototype_atoms_dataframe = pd.DataFrame(data_list)

        return(prototype_atoms_dataframe)

        #__|




    def get_protoid_to_strucid(self,
        df_text_key_values=None,
        user_stoich=None,
        data_file_path=None):
        """
        """
        #| - get_protoid_to_strucid
        # Getting unique prototype IDs for given constraints
        # unique_prototype_names = self.get_distinct_prototypes(
        unique_protonames_for_user = self.get_distinct_prototypes(
            source=None,
            formula=user_stoich,
            repetition=None)

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

    def create_atoms_object_with_replacement_tmp(self,
        indiv_data_tmp_i,
        user_elems=None):
        """

        Parameters
        ----------
        indiv_data_tmp_i: str
        user_elems: list
        """
        #| - create_atoms_object_with_replacement_tmp
        spacegroup_i = indiv_data_tmp_i["spacegroup"]
        prototype_wyckoffs_i = indiv_data_tmp_i["prototype_wyckoffs"]
        prototype_species_i = indiv_data_tmp_i["prototype_species"]
        init_params = indiv_data_tmp_i["prototype_params"]

        #| - Atom type replacement
        # #############################################################
        # #############################################################
        # #############################################################
        prototype_species_i


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


        # #############################################################
        freq_data_list = []
        for key_i, value_i in elem_count_freq.items():
            tmp = {
                "element": key_i,
                "frequency": value_i,
                }
            freq_data_list.append(tmp)


        elem_mapping_dict = dict(zip(
            pd.DataFrame(
                freq_data_list).sort_values(
                    by=["frequency"])["element"].tolist(),
            user_elems,
            ))

        #__|

        # #############################################################
        new_elem_list = []
        for i in prototype_species_i:
            new_elem_list.append(
                elem_mapping_dict.get(i, i)
                )
        # #############################################################

        # Preparing Initial Wyckoff parameters to pass to the
        # CellParameters Code
        init_params_cpy = copy.copy(init_params)

        non_wyck_params = [
            "a", "b", "c",
            "b/a", "c/a",
            # "alpha", "beta", "gamma",
            ]

        for wyck_i in non_wyck_params:
            if wyck_i in list(init_params_cpy.keys()):
                del init_params_cpy[wyck_i]

        init_wyck_params = init_params_cpy

        #| - Using CellParameters Code
        # CP = CellParameters(
        #     spacegroup=spacegroup_i,
        #     wyckoffs=prototype_wyckoffs_i,
        #     species=new_elem_list,
        #     # species=prototype_species_i,
        #     )
        # atoms_init = CP.get_atoms(fix_parameters=init_params)
        # atoms_out = atoms_init

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



#| - __old__


    # def return_data_dict_for_id(self,
    #     id_i,
    #     df_systems=None):
    #     """
    #     """
    #     #| - return_data_dict_for_id
    #     #TODO add check
    #     # There should only be row
    #     data_string_i_tmp = df_systems[
    #         df_systems["id"] == id_i]["key_value_pairs"]
    #     data_string_i = data_string_i_tmp.iloc[0]
    #
    #
    #     # Error on literal_eval from NaN without quotes
    #     data_string_i_2 = data_string_i.replace(" NaN,", ' " NaN",')
    #
    #     try:
    #         data_0 = literal_eval(data_string_i_2)
    #     except:
    #         print(data_string_i_2)
    #         data_0 = {}
    #
    #     spacegroup = data_0["spacegroup"]
    #     proto_name = data_0["proto_name"]
    #
    #     # ######################################################################
    #     # ######################################################################
    #     # ######################################################################
    #
    #     data_string_1 = df_systems[
    #         df_systems["id"] == id_i]["data"].iloc[0]
    #     data_1 = literal_eval(data_string_1)
    #
    #     prototype_params = literal_eval(data_1["param"])
    #     prototype_species = literal_eval(data_1["species"])
    #     prototype_wyckoffs = literal_eval(data_1["wyckoffs"])
    #
    #
    #     data_dict_i = {
    #         "id": id_i,
    #         "proto_name": proto_name,
    #         "spacegroup": spacegroup,
    #
    #         "prototype_params": prototype_params,
    #         "prototype_species": prototype_species,
    #         "prototype_wyckoffs": prototype_wyckoffs,
    #         }
    #
    #     return(data_dict_i)
    #     #__|
    #
    # def process_systems_table(self):
    #     """
    #     """
    #     #| - process_systems_table
    #     # # Read table from sql
    #     # dbfile = self.dbfile
    #     # con = sqlite3.connect(dbfile)
    #     # df_systems = pd.read_sql_query("SELECT * FROM systems", con)
    #     #
    #     #
    #     # #| - Drop unnecessary Columns
    #     # cols_to_drop = [
    #     #     #  'id',
    #     #     #  'unique_id',
    #     #     'ctime',
    #     #     'mtime',
    #     #     'username',
    #     #     'numbers',
    #     #     'positions',
    #     #     'cell',
    #     #     'pbc',
    #     #     'initial_magmoms',
    #     #     'initial_charges',
    #     #     'masses',
    #     #     'tags',
    #     #     'momenta',
    #     #     'constraints',
    #     #     'calculator',
    #     #     'calculator_parameters',
    #     #     'energy',
    #     #     'free_energy',
    #     #     'forces',
    #     #     'stress',
    #     #     'dipole',
    #     #     'magmoms',
    #     #     'magmom',
    #     #     'charges',
    #     #     #  'key_value_pairs',
    #     #     #  'data',
    #     #     'natoms',
    #     #     'fmax',
    #     #     'smax',
    #     #     'volume',
    #     #     'mass',
    #     #     'charge',
    #     #     ]
    #     #
    #     # df_systems = df_systems.drop(cols_to_drop, 1)
    #     # #__|
    #     #
    #     #
    #     # data_list = []
    #     # # for i_cnt, row_i in df_systems[0:20].iterrows():
    #     # for i_cnt, row_i in df_systems.iterrows():
    #     #
    #     #     #| - Processing 'key_value_pairs' column
    #     #     key_value_pairs_str = row_i["key_value_pairs"]
    #     #     key_value_pairs_str = key_value_pairs_str.replace(" NaN,", ' " NaN",')
    #     #
    #     #     try:
    #     #         key_value_pairs_dict = literal_eval(key_value_pairs_str)
    #     #
    #     #     except:
    #     #         print("This error shouldn't be happening!!!! 98ufs8")
    #     #         print(key_value_pairs_str)
    #     #
    #     #         key_value_pairs_dict = {}
    #     #
    #     #     spacegroup = key_value_pairs_dict["spacegroup"]
    #     #     proto_name = key_value_pairs_dict["proto_name"]
    #     #
    #     #
    #     #     keys_to_delete = [
    #     #         #  'proto_name',
    #     #         #  'spacegroup',
    #     #          'name',
    #     #          'directory',
    #     #          'energy_pa',
    #     #          'volume',
    #     #          'bandgap',
    #     #          'delta_e',
    #     #          'stability',
    #     #         #  'source',
    #     #          ]
    #     #
    #     #     for key_i in keys_to_delete:
    #     #         key_value_pairs_dict.pop(key_i, None)
    #     #     #__|
    #     #
    #     #     #| - Processing 'Data' column
    #     #     data_str = row_i["data"]
    #     # #     key_value_pairs_str = key_value_pairs_str.replace(" NaN,", ' " NaN",')
    #     #
    #     #     data_dict = literal_eval(data_str)
    #     #
    #     #     prototype_params = literal_eval(data_dict["param"])
    #     #     prototype_species = literal_eval(data_dict["species"])
    #     #     prototype_wyckoffs = literal_eval(data_dict["wyckoffs"])
    #     #
    #     #     data_dict = {
    #     #         "prototype_params": prototype_params,
    #     #         "prototype_species": prototype_species,
    #     #         "prototype_wyckoffs": prototype_wyckoffs,
    #     #         }
    #     #     #__|
    #     #
    #     #     out_dict = {
    #     #         "id": row_i["id"],
    #     #         **key_value_pairs_dict,
    #     #         **data_dict,
    #     #         }
    #     #     data_list.append(out_dict)
    #     #
    #     # df_sys = pd.DataFrame(data_list)
    #     # df_sys = df_sys.set_index("id", drop=True)
    #     #
    #     # return(df_sys)
    #     #__|
    #


#__|
