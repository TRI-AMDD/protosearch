"""
190419 | Test change RF
"""

#| - Import Modules
from ase.db import connect
import sqlite3

import copy
import pandas as pd

from ast import literal_eval
from protosearch.build_bulk.cell_parameters import CellParameters
#__|

class OqmdInterface:

    def __init__(self, dbfile):
        #| - __init__
        self.dbfile = dbfile

        #__|

    def create_proto_data_set(self, formula, elements):
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
        assert type(formula) == str, "Formula must be given as a string"
        assert type(elements) == list, "elements must be given as a list"
        #__|

        con = sqlite3.connect(dbfile)
        df_systems = pd.read_sql_query("SELECT * FROM systems", con)
        df_text_key_values = pd.read_sql_query(
            "SELECT * FROM text_key_values", con)

        #| - TEMP_0
        df_tmp3 = self.temp_get_df(
            df_text_key_values=df_text_key_values,
            user_stoich=formula,
            data_file_path=dbfile)

        number_rows_to_do = 10

        data_tmp3 = []
        # for i_ind, row_i in df_tmp3.iterrows():
        for i_ind, row_i in df_tmp3[0:number_rows_to_do].iterrows():
            protoname_i = row_i["protoname"]

            prototype_data_i = []
            for id_i in row_i["id_list"]:
                data_id_i = self.return_data_dict_for_id(
                    id_i,
                    df_systems=df_systems,
                    )

                prototype_data_i.append(data_id_i)
                # data_tmp3.append(data_id_i)

            data_tmp3.append({
                "prototype_name": protoname_i,
                "data": prototype_data_i,
                })

        data_out_tmp8 = pd.DataFrame(data_tmp3)
        #__|


        #| - Performing Atom Subtitution to Create the Atoms List
        atoms_list_out = []
        for i_cnt, row_i in data_out_tmp8.iterrows():

            # Just considering 1 (the first) entry for each unique prototype
            structure_first_i = row_i["data"][0]
            print(i_cnt)

            # TODO Hacky try/except because of error from Enumerator
            try:
                atoms_i = self.create_atoms_object_with_replacement_tmp(
                    structure_first_i,
                    user_elems=elements)

                atoms_list_out.append(atoms_i)

            except:
                pass
        #__|


        return(atoms_list_out)
        #__|







    #| - Methods
    def temp_get_df(self,
        df_text_key_values=None,
        user_stoich=None,
        data_file_path=None):
        """
        """
        #| - temp_get_df
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

        #| - __old__
        # df_unique_proto = pd.DataFrame(
        #     unique_prototype_names, columns=["proto_name"])
        #
        #
        #
        # def get_stoich_from_protoname(row_i):
        #     """
        #     """
        #     #| - TEMP
        #     stoich_i = row_i["proto_name"].split("_")[0]
        #     return(stoich_i)
        #     #__|
        #
        # df_unique_proto["stoich_i"] = df_unique_proto.apply(
        #     get_stoich_from_protoname,
        #     axis=1)
        #
        # df_unique_proto_user = df_unique_proto[
        #     df_unique_proto["stoich_i"] == user_stoich]
        #
        #
        # tmp1 = df_unique_proto_user["proto_name"].tolist()
        # unique_protonames_for_user = tmp1
        #__|

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

    def return_data_dict_for_id(self,
        id_i,
        df_systems=None):
        """
        """
        #| - return_data_dict_for_id
        #TODO add check
        # There should only be row
        data_string_i_tmp = df_systems[
            df_systems["id"] == id_i]["key_value_pairs"]
        data_string_i = data_string_i_tmp.iloc[0]


        # Error on literal_eval from NaN without quotes
        data_string_i_2 = data_string_i.replace(" NaN,", ' " NaN",')

        try:
            data_0 = literal_eval(data_string_i_2)
        except:
            print(data_string_i_2)
            data_0 = {}

        spacegroup = data_0["spacegroup"]
        proto_name = data_0["proto_name"]

        # ######################################################################
        # ######################################################################
        # ######################################################################

        data_string_1 = df_systems[
            df_systems["id"] == id_i]["data"].iloc[0]
        data_1 = literal_eval(data_string_1)

        prototype_params = literal_eval(data_1["param"])
        prototype_species = literal_eval(data_1["species"])
        prototype_wyckoffs = literal_eval(data_1["wyckoffs"])


        data_dict_i = {
            "id": id_i,
            "proto_name": proto_name,
            "spacegroup": spacegroup,

            "prototype_params": prototype_params,
            "prototype_species": prototype_species,
            "prototype_wyckoffs": prototype_wyckoffs,
            }

        return(data_dict_i)
        #__|

    def create_atoms_object_with_replacement_tmp(self,
        indiv_data_tmp_i,
        user_elems=None):
        """
        """
        #| - create_atoms_object_with_replacement_tmp
        spacegroup_i = indiv_data_tmp_i["spacegroup"]
        prototype_wyckoffs_i = indiv_data_tmp_i["prototype_wyckoffs"]
        prototype_species_i = indiv_data_tmp_i["prototype_species"]

        init_params = indiv_data_tmp_i["prototype_params"]

        # Atom type replacement
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

        # #############################################################
        new_elem_list = []
        for i in prototype_species_i:
            new_elem_list.append(
                elem_mapping_dict.get(i, i)
                )

        # print(new_elem_list)
        # Preparing Initial Wyckoff parameters to pass to the
        # CellParameters Code

        # #############################################################

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
