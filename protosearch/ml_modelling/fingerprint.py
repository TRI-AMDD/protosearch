import sys
import copy
import numpy as np
import pandas as pd

from catlearn.fingerprint.voro import VoronoiFingerprintGenerator
from catlearn.preprocess.clean_data import (
    clean_infinite,
    clean_variance,
    clean_skewness,
)

from catlearn.preprocess.scaling import standardize

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

    def __init__(self,
                 feature_methods=None,
                 input_data=None,
                 input_index=None):
        """
        Parameters
        ----------
        feature_methods: list
            List of featurizing methods to apply to inputs.
        input_data: list or pandas_dataframe
            Pandas dataframe containing a column corresponding to the input
            objects to be featurized.
        input_index: str
            must be defined to index the correct column in inpud_data
        clean: Bool
            whether to clean features or not
        """
        self.feature_methods = feature_methods
        self.input_index = input_index
        self.input_data = input_data

        self.__check_class_inputs__()

        self.__instantiate_feature_classes__()

    def __check_class_inputs__(self):
        """
        checking_class_inputs
        """

        # feature_methods
        err_mess = "feature_methods must be of type <list>"
        assert isinstance(self.feature_methods, list), err_mess

        for method in self.feature_methods:
            err_mess_i = "feature methods must be in the following: \n"
            err_mess_i += str(feature_methods_dict.keys())
            assert method in feature_methods_dict.keys(), err_mess_i

        # input_data
        is_pd_df = isinstance(self.input_data, pd.DataFrame)
        if not is_pd_df:
            msg = "Please give input_data as a pandas dataframe \n"
            msg += "At the least a dataframe with 1 column"
            raise TypeError("Please give input_data as a pandas dataframe")

        # input_index
        # Is this necessary?
        if isinstance(self.input_index, list):
            pass
            # self.input_index = tuple(input_index)
        elif isinstance(self.input_index, set):
            pass

        else:
            raise TypeError("input_index must be given as a <list> or <set>")

    def __instantiate_feature_classes__(self):
        out_dict = {}
        for method in self.feature_methods:
            feature_class_name = feature_methods_dict[method]
            class_i = getattr(sys.modules[__name__], feature_class_name)

            # COMBAK There were some issues caused here by the .loc method
            # returning either a pandas.DataFrame or a pandas.series
            # input_array = input_data.loc[:, input_index].tolist()
            input_array = \
                self.input_data.loc[:, self.input_index].iloc[:, 0].tolist()
            instance = class_i(input_array)
            out_dict[method] = instance

        self.feature_instances = out_dict

    def generate_fingerprints(self):
        # generate_fingerprints
        feature_instances = self.feature_instances
        input_data = self.input_data

        # Collecting fingerprint dataframes from fingerprint instances
        fingerprints = {}
        for name_i, feature_instance_i in feature_instances.items():
            feature_instance_i.generate_fingerprints()

            features_i = feature_instance_i.features

            # Checking type of fingerprints (must be pandas dataframe)
            # Fingerprints must be given as a pandas dataframe
            is_pd_df = isinstance(features_i, pd.DataFrame)
            err_mess_i = "Fingerprint class must return a pandas dataframe"
            assert is_pd_df, err_mess_i

            features_i = features_i.set_index(
                input_data.index,
                # np.array(rand_ids),
                drop=False, append=False,
                inplace=False, verify_integrity=True)

            fingerprints[name_i] = features_i

        fingerprints_out = pd.concat(fingerprints.values(),
                                     axis=1,
                                     keys=fingerprints.keys())\

        self.fingerprints = fingerprints_out
        return self.fingerprints

    def join_input_to_fingerprints(self):
        """Concancotate
        """
        # join_input_to_fingerprints
        input_data = self.input_data
        fingerprints = self.fingerprints

        df_out = pd.merge(input_data, fingerprints,
                          left_index=True,
                          right_index=True,
                          indicator=True,  # This was breaking the method for some reason
                          )

        # TODO
        # Check that operation was succesful
        # Merge command shouldn't be dropping any rows

        # print(len(input_data))
        # print(len(fingerprints))
        # print(len(df_out))

        if len(input_data) != len(fingerprints):
            print("MISTAKE iasdjfisj")
        if len(input_data) != len(df_out):
            print("MISTAKE iasdjfisj2")
        if len(fingerprints) != len(df_out):
            print("MISTAKE iasdjfisj3")

        self.fingerprints = df_out

        # return(input_data, fingerprints)
        # return(df_out)


class VoronoiFingerprint:
    """
    Uses the CatLearn interface to MagPie Voronoi Fingerprint

    """
    from catlearn.fingerprint.voro import VoronoiFingerprintGenerator

    def __init__(self,
                 atoms_list):
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
        self.check_inputs(atoms_list)
        self.Voro_inst = VoronoiFingerprintGenerator(
            atoms_list,
            delete_temp=False)

    def check_inputs(self, atoms_list):
        # check atoms list
        from ase import Atoms

        type_check_list = [isinstance(atoms_i, Atoms)
                           for atoms_i in atoms_list]
        err_mess = "Inputs to Voronoi must be atom objects"
        assert all(type_check_list), err_mess

    def generate_fingerprints(self):
        self.features = self.Voro_inst.generate()


def clean_features(features, scale=False):
    remove_indices = {'train': np.array([], dtype=int),
                      'test': np.array([], dtype=int)}
    for key, feature_set in features.items():
        if feature_set is None:
            continue
        bad_structure_indices = \
            np.where(np.isfinite(feature_set).all(axis=1)==False)
        for b in bad_structure_indices:
            if len(np.where(np.isfinite(feature_set[b])==False)) > 1:
                remove_indices[key] = np.append(remove_indices[key], b)

        features[key] = np.delete(feature_set, remove_indices[key], axis=0)
    if not 'test' in features:
        features['test'] = None

    # Finite features
    features = clean_infinite(features['train'],
                            features['test'])
    # Clean variance
    features = clean_variance(features['train'],
                              features['test'])

    #Clean skewness
    features = clean_skewness(features['train'],
                              features.get('test', None),
                              skewness=3)

    if scale:
        features = standardize(features['train'],
                            features.get('test', None))

    return features, remove_indices
