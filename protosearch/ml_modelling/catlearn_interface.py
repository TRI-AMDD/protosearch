from catlearn.fingerprint.voro import VoronoiFingerprintGenerator
from catlearn.preprocess.clean_data import clean_infinite, clean_variance
from catlearn.regression.gaussian_process import GaussianProcess


def get_voro_fingerprint(atoms_list):

    voro = VoronoiFingerprintGenerator(atoms_list)
    data_frame = voro.generate()
    matrix = data_frame.values

    finite_numeric_data = clean_infinite(matrix)
    #data = clean_variance(finite_numeric_data['train'])

    return finite_numeric_data['train']


def predict(train_features, train_target, test_features):
    kernel = [{'type': 'gaussian', 'width': 1., 'scaling': 1., 'dimension': 'single'}]

    GP = GaussianProcess(train_fp=train_features, train_target=train_target, kernel_list=kernel,
                         regularization=1e-2, optimize_hyperparameters=True, scale_data=True)

    pred = GP.predict(test_fp=test_features, uncertainty=True)

    return pred
