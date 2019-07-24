from catlearn.fingerprint.voro import VoronoiFingerprintGenerator
from catlearn.preprocess.clean_data import clean_infinite, clean_variance
from catlearn.preprocess.scaling import normalize
from catlearn.regression.gaussian_process import GaussianProcess


def get_voro_fingerprint(atoms_list):

    voro = VoronoiFingerprintGenerator(atoms_list)
    data_frame = voro.generate()
    values = data_frame.values
    columns = data_frame.columns
    return columns, values


def predict(train_features, train_target, test_features):

    finite_numeric_data = clean_infinite(train_features,
                                         test=test_features,
                                         targets=train_target)
    train_features = finite_numeric_data['train']
    test_features = finite_numeric_data['test']
    train_target = finite_numeric_data['targets']
    data = clean_variance(train_features, test_features)
    data = normalize(data['train'], data['test'])
    train_features = data['train']
    test_features = data['test']

    kernel = [{'type': 'gaussian', 'width': 3, 'scaling': 1.}]

    GP = GaussianProcess(
        train_fp=train_features,
        train_target=train_target,
        kernel_list=kernel,
        regularization=3e-2,
        optimize_hyperparameters=True,
        scale_data=True, # True is breaking code sometimes
        )

    pred = GP.predict(test_fp=test_features, uncertainty=True)

    return pred
