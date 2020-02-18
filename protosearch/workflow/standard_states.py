from protosearch.utils.standards import CrystalStandards
from protosearch.workflow.workflow import Workflow

standard_lattice = CrystalStandards.standard_lattice_mp
all_elements = list(standard_lattice.keys())


class StandardStates:

    def __init__(self):
        pass

    def submit_standard_states(self, elements=None):
        elements = elements or all_elements
        W = Workflow()
        print(elements)
        for e in elements:
            print(e)
            prototype = standard_lattice[e]
            W.submit(prototype)
