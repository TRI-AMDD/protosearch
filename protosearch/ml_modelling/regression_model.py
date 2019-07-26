from catlearn.regression.gaussian_process import GaussianProcess
ml_methods_dict = {
    "catlearn": "CatLearnGP"}


def get_regression_model(name):
    if name == 'catlearn':
        from protosearch.ml_modelling.regression_model import CatLearnGP as Model
    return Model


class CatLearnGP:

    # CatLearn GP interface

    def __init__(self,
                 features,
                 targets,
                 kernel_type='gaussian',
                 kernel_width=3,
                 kernel_scaling=1.,
                 kernel_dimension='single',
                 regularization=3e-2,
                 optimize_hyperparameters=True,
                 scale_data=True):

        self.features = features
        self.targets = targets
        kernel_list = [{'type': kernel_type,
                        'width': kernel_width,
                        'scaling': kernel_scaling,
                        'dimension': kernel_dimension}]

        self.GP = GaussianProcess(
            train_fp=features,
            train_target=targets,
            kernel_list=kernel_list,
            regularization=regularization,
            optimize_hyperparameters=True,
            scale_data=True,
        )

    def predict(self, test_features, uncertainty=True):
        prediction = self.GP.predict(test_features,
                                     uncertainty=uncertainty)

        return prediction
