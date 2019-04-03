#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

#| - Import Modules
import sys
import copy

import pandas as pd

from catlearn.fingerprint.voro import VoronoiFingerprintGenerator

from catlearn.preprocess.clean_data import (
    clean_infinite,
    clean_variance,
    clean_skewness,
    )

from catlearn.preprocess.scaling import standardize
#__|

# Featurizing methods implemented, must have corresponding class
feature_methods_dict = {
    "voronoi": "VoronoiFingerprint",
    }

class FingerPrint:
    """General fingerprinting/featurizing class.

    Development Notes:
        * If the fingerprinting needs to know what the entire set of atoms
        that must be taken into account
        * Incorporate feature engineering
        * TEMP

    """

    #| - FingerPrint **********************************************************

    def __init__(self,
        feature_methods=None,
        input_objects=None,
        ):
        """__init__.

        Parameters
        ----------
        feature_methods : list
            List of featurizing methods to apply to inputs.
        inputs : list
            List of inputs to be featurized
        """
        #| - __init__

        #| - Setting Class Attributes
        self.__feature_methods__ = feature_methods
        self.input_objects = input_objects
        #__|

        self.__check_class_inputs__()

        self.__feature_instances__ = self.__instantiate_feature_classes__()

        # self.clean_features()
        #__|

    def __check_class_inputs__(self):
        """
        """
        #| - __checking_class_inputs__

        #| - __feature_methods__
        feature_methods = self.__feature_methods__


        err_mess_i = "feature_methods must be of type <list>"
        assert feature_methods is not None, err_mess_i

        err_mess_i = "feature_methods must be of type <list>"
        assert type(feature_methods) is list, err_mess_i

        for meth_i in feature_methods:
            err_mess_i = "feature methods must be in the following: \n"
            err_mess_i += str(feature_methods_dict.keys())

            assert meth_i in feature_methods_dict.keys(), err_mess_i

        #__|

        #__|

    def __check_input_objects__(self):
        """
        """
        #| - __check_input_objects__
        tmp = 42

        #__|

    def __instantiate_feature_classes__(self):
        """
        """
        #| - __instantiate_feature_classes__
        feature_methods = self.__feature_methods__
        input_objects = self.input_objects

        out_dict = {}
        for meth_i in feature_methods:
            feature_class_name_i = feature_methods_dict[meth_i]
            class_i = getattr(sys.modules[__name__], feature_class_name_i)
            instance_i = class_i(input_objects)

            out_dict[meth_i] = instance_i

        return(out_dict)
        #__|

    def clean_features(self):
        """
        """
        #| - clean_features
        df_features = self.fingerprints
        df_features_cpy = copy.deepcopy(df_features)

        train_features = df_features_cpy.values
        train_labels = list(df_features_cpy)

        #| - Clean variance
        output = clean_variance(
            train_features,
            test=None,
            labels=train_labels,
            mask=None,
            )
        train_features = output["train"]
        train_labels = output["labels"]
        #__|

        #| - Clean infinite
        output = clean_infinite(
            train_features,
            test=None,
            targets=None,
            labels=train_labels,
            mask=None,
            max_impute_fraction=0,
            strategy='mean',
            )
        train_features = output["train"]
        train_labels = output["labels"]
        #__|

        #| - Clean skewness
        output = clean_skewness(
            train_features,
            test=None,
            labels=train_labels,
            mask=None,
            skewness=3.,
            )
        train_features = output["train"]
        train_labels = output["labels"]

        column_labels = output["labels"]
        #__|


        #| - Standardize Data
        output = standardize(
            train_features,
            test_matrix=None,
            mean=None,
            std=None,
            local=True,
            )

        #__|


        #| - Reconstruct dataframe
        df_features_cleaned = pd.DataFrame(
            data=output["train"],
            # columns=output["labels"],
            )

        multi_index = pd.MultiIndex.from_tuples(
            [tuple(i) for i in column_labels],
            # names=("tmp1", "tmp2"),
            )

        df_features_cleaned.columns = multi_index
        #__|


        self.fingerprints_precleaned = df_features
        self.fingerprints = df_features_cleaned

        #| - __old__
        # return(df_features_out)
        # df_features = self.fingerprints
        #
        # columns_to_remove = []
        # for column in df_features:
        #     num_unique_vals = len(list(set(df_features[column].tolist())))
        #
        #     if num_unique_vals == 1:
        #         columns_to_remove.append(column)
        #
        # df_features = df_features.drop(columns_to_remove, axis=1)
        #__|

        #__|

    def generate_fingerprints(self):
        #| - generate_fingerprints
        feature_instances = self.__feature_instances__


        # Collecting fingerprint dataframes from fingerprint instances
        fingerprints = {}
        for name_i, feature_instance_i in feature_instances.items():
            feature_instance_i.generate_fingerprints()

            features_i = feature_instance_i.features
            fingerprints[name_i] = features_i

            #| - Checking type of fingerprints (must be pandas dataframe)
            # Fingerprints must be given as a pandas dataframe
            is_pd_df = isinstance(
                features_i,
                pd.DataFrame,
                )

            err_mess_i = "Fingerprint class must return a pandas dataframe"
            assert is_pd_df is True, err_mess_i
            #__|

        fingerprints_out = pd.concat(
            fingerprints.values(),
            axis=1,
            keys=fingerprints.keys())\



        self.fingerprints = fingerprints_out


        # return(fingerprints_out)
        #__|


    #__| **********************************************************************


# #############################################################################
# #############################################################################
# #############################################################################

class VoronoiFingerprint:
    """

    """

    #| - VoronoiFingerprint ***************************************************
    from catlearn.fingerprint.voro import VoronoiFingerprintGenerator

    def __init__(self,
        atoms_list,
        ):
        """Voronoi fingerprinting setup.

        Parameters
        ----------
        atoms_list : list
            A list of ase atoms objects to be featurized.


        Development Notes:
        ------------------
        * This featurizing method is specific to atoms objects, for the sake
        of generality lets just assume that inputs can be anything, therefore
        check that inputs are atoms objects

        """
        #| - __init__
        self.atoms_list = atoms_list

        self.Voro_inst = VoronoiFingerprintGenerator(
            self.atoms_list,
            delete_temp=False,
            )
        #__|

    def __check_inputs__(self):
        """
        """
        #| - __check_inputs__
        from ase import Atoms

        atoms_list = self.atoms_list

        type_check_list = []
        for atom_i in atoms_list:
            is_atoms_object = isinstance(
                atom_i,
                Atoms,
                )

            type_check_list.append(is_atoms_object)

        err_mess_i = "Inputs to Voronoi must be atom objects"
        assert all(type_check_list) is True, err_mess_i
        #__|


    def generate_fingerprints(self):
        """
        """
        #| - generate_fingerprints
        self.features = self.Voro_inst.generate()
        #__|



    #| - __out_of_sight__
    # # Setting Voronai index to those in the main dataframe
    # df_vor = df_vor.set_index(
    #     df_m.index.values,
    #     drop=True,
    #     append=False,
    #     inplace=False,
    #     verify_integrity=False,
    #     )
    #__|

    #__| **********************************************************************
