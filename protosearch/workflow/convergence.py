from .workflow import Workflow


class Convergence(Workflow):
    """TO DO: Automated convergence check """

    def __init__(self,
                 check_parameters=['encut', 'kspacing']):

        super().__init__()
