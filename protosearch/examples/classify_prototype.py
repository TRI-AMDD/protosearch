import pprint
import ase
from ase.io import read

from protosearch.workflow.classification import get_classification

atoms = read('examples/Se_mp-570481_conventional_standard.cif')
result, param = get_classification(atoms)

pprint.pprint(result)

print('')

pprint.pprint(param)
