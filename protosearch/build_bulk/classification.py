import io
import ase
import bulk_enumerator as be
from ase.io.vasp import write_vasp


def get_classification(atoms, tolerance=5e-3):
    """ Get the prototype and cell parameters of an atomic structure
    in ASE Atoms format"""

    b = be.bulk.BULK(tolerance=tolerance)

    poscar = io.StringIO()
    write_vasp(filename=poscar, atoms=atoms, vasp5=True,
               long_format=False, direct=True)

    poscar = poscar.getvalue()

    b.set_structure_from_file(poscar)
    name = b.get_name()
    spacegroup = b.get_spacegroup()
    wyckoffs = b.get_wyckoff()
    species = b.get_species()
    parameters = b.get_parameter_values()
    b.delete()
    structure_name = str(spacegroup)
    for spec, wy_spec in zip(species, wyckoffs):
        structure_name += '_{}_{}'.format(spec, wy_spec)
    prototype = {'p_name': name,
                 'structure_name': structure_name,
                 'spacegroup': spacegroup,
                 'wyckoffs': wyckoffs,
                 'species': species}

    parameter_dict = {}
    for param in parameters:
        if param['name'] in ['b/a', 'c/a']:
            param['name'] = param['name'][:1]
        parameter_dict.update({param['name']: param['value']})

    return prototype, parameter_dict
