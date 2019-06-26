from catlearn.fingerprint.voro import VoronoiFingerprintGenerator
from catlearn.preprocess.clean_data import clean_infinite, clean_variance


def get_voro_fingerprint(atoms_list):

    voro = VoronoiFingerprintGenerator(atoms_list)
    data_frame = voro.generate()
    matrix = data_frame.values

    finite_numeric_data = clean_infinite(matrix)
    #data = clean_variance(finite_numeric_data['train'])

    return finite_numeric_data['train']


def train(self):
    # Regression model
    pass
