from .workflow import Workflow, MLinterf


class Acqusition(Workflow, MLInterface):
    """ TODO: Decide which calculations to do next. 
    Should interface MLInterface for predictions.
    """

    def __init__(self):
        super().__init__()

    def get_prediction(self):
        prototypes = None
        return prototypes

    def update_model(self):
        """ Should we call the ML model here, 
        to update with completed calculations? """

        self.collect()  # collect completed calculations
        return None

    def submit_batch(self, prototypes, species):
        """Call WorkFlow module to submit selected prototypes"""

        prototypes = self.get_prediction()
        for prototype in prototypes:
            self.submit(prototype)
