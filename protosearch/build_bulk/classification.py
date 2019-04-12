import io
import ase
from ase.io.vasp import read_vasp, write_vasp
import bulk_enumerator as be


def get_classification(atoms):
    """ Get the prototype and cell parameters of an atomic structure
    in ASE Atoms format"""

    b = be.bulk.BULK()

    poscar = io.StringIO()
    write_vasp(filename=poscar, atoms=atoms, vasp5=True,
               long_format=False, direct=True)

    b.set_structure_from_file(poscar.getvalue())
    name = b.get_name()
    spacegroup = b.get_spacegroup()
    wyckoffs = b.get_wyckoff()
    species = b.get_species()
    parameters = b.get_parameter_values()

    prototype = {'p_name': name,
                 'spacegroup': spacegroup,
                 'wyckoffs': wyckoffs,
                 'species': species}

    parameter_dict = {}
    for param in parameters:
        if param['name'] in ['b/a', 'c/a']:
            param['name'] = param['name'][:1]
        parameter_dict.update({param['name']: param['value']})

    return prototype, parameter_dict
