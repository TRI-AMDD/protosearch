from catlearn.fingerprint.voro import VoronoiFingerprintGenerator
from catlearn.preprocess.clean_data import clean_infinite, clean_variance
from catlearn.preprocess.scaling import normalize
from catlearn.regression.gaussian_process import GaussianProcess


def get_voro_fingerprint(atoms_list):

    voro = VoronoiFingerprintGenerator(atoms_list)
    data_frame = voro.generate()
    matrix = data_frame.values

    finite_numeric_data = clean_infinite(matrix)

    return finite_numeric_data['train']


def predict(train_features, train_target, test_features):

    data = clean_variance(train_features, test_features)
    data = normalize(data['train'], data['test'])
    train_features = data['train']
    test_features = data['test']

    kernel = [{'type': 'gaussian', 'width': 3, 'scaling': 1.}]

    GP = GaussianProcess(train_fp=train_features, train_target=train_target,
                         kernel_list=kernel, regularization=5e-2,
                         optimize_hyperparameters=True, scale_data=True)

    pred = GP.predict(test_fp=test_features, uncertainty=True)

    return pred
