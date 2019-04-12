import os
import io
from ase.io.vasp import read_vasp
import bulk_enumerator as be

from .cell_parameters import CellParameters


class BuildBulk(CellParameters):
    """
    Set up Vasp calculations on TRI-AWS for bulk structure created with the
    Bulk prototype enumerator developed by A. Jain described in:
    A. Jain and T. Bligaard, Phys. Rev. B 98, 214112 (2018)

    Parameters:

    spacegroup: int
        int between 1 and 230
    wyckoffs: list
        wyckoff positions, for example ['a', 'a', 'b', 'c']
    species: list
        atomic species, for example ['Fe', 'O', 'O', 'O']
    cell_parameters: dict
        Optional specification of cell parameters, 
        such as {'a': 3.7, 'alpha': 75}.
        Otherwise a fair guees for parameters will be provided by the
        CellParameters module.
    """

    def __init__(self,
                 spacegroup,
                 wyckoffs,
                 species,
                 cell_parameters=None
                 ):

        super().__init__(spacegroup=spacegroup,
                         wyckoffs=wyckoffs,
                         species=species)

        assert (0 < spacegroup < 231 and isinstance(spacegroup, int)), \
            'Spacegroup must be an integer between 1 and 230'

        self.poscar = None
        self.spacegroup = spacegroup
        self.wyckoffs = wyckoffs
        self.species = species

        TRI_PATH = os.environ['TRI_PATH']
        username = os.environ['TRI_USERNAME']

        master_parameters = cell_parameters or {}
        self.cell_parameters = self.get_parameter_estimate(master_parameters)

        if self.cell_parameters:
            self.cell_param_list = []
            self.cell_value_list = []

            for param in self.cell_parameters:
                self.cell_value_list += [self.cell_parameters[param]]
                self.cell_param_list += [param]

    def get_poscar(self):
        """Get POSCAR structure file from the Enumerator """
        if not self.cell_parameters:
            return None
        b = be.bulk.BULK()
        b.set_spacegroup(self.spacegroup)
        b.set_wyckoff(self.wyckoffs)
        b.set_species(self.species)

        b.set_parameter_values(self.cell_param_list, self.cell_value_list)
        self.prototype_name = b.get_name()

        self.poscar = b.get_primitive_poscar()

        return self.poscar

    def get_atoms_from_poscar(self):
        if not self.cell_parameters:
            return None
        if not self.poscar:
            poscar = self.get_poscar()

        atoms = read_vasp(io.StringIO(self.poscar))

        return atoms
