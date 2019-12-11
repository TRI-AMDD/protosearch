import os

from .cell_parameters import CellParameters


class BuildBulk(CellParameters):
    """
    Set up bulk structures for a given prototype specification.

    Prototypes can be enumeated with the Bulk prototype enumerator,
    developed by A. Jain described in:
    A. Jain and T. Bligaard, Phys. Rev. B 98, 214112 (2018)

    Parameters:

    spacegroup: int
        int between 1 and 230
    wyckoffs: list
        wyckoff positions, for example ['a', 'a', 'b', 'c']
    species: list
        atomic species, for example ['Fe', 'O', 'O', 'O']
    """

    def __init__(self,
                 spacegroup,
                 wyckoffs,
                 species
                 ):

        super().__init__(spacegroup=spacegroup,
                         wyckoffs=wyckoffs,
                         species=species)

        assert (0 < spacegroup < 231 and isinstance(spacegroup, int)), \
            'Spacegroup must be an integer between 1 and 230'

        self.spacegroup = spacegroup
        self.wyckoffs = wyckoffs
        self.species = species

    def get_atoms(self, proximity=1, cell_parameters=None, primitive_cell=True):
        """Get one atoms object pr prototype

        Parameters:

        proximity: float
            number close to 1, specifying the proximity of atoms in the 
            hard sphere model. r_spheres = proximity * r_covalent

        cell_parameters: dict
            Optional specification of cell parameters, 
             such as {'a': 3.7, 'alpha': 75}.
             Otherwise a guees for parameters will be provided by the
             CellParameters module."""

        master_parameters = cell_parameters or {}
        cell_parameters = self.get_parameter_estimate(master_parameters=master_parameters,
                                                      proximity=proximity,
                                                      max_candidates=1)

        return self.construct_atoms(cell_parameters[0],
                                    primitive_cell=primitive_cell)

    def get_wyckoff_candidate_atoms(self,
                                    proximity=1,
                                    cell_parameters=None,
                                    primitive_cell=True,
                                    return_parameters=False,
                                    max_candidates=None):
        """Returns a list of atomic structures with different wyckoff settings"""
        master_parameters = cell_parameters or {}

        cell_parameters = \
            self.get_parameter_estimate(master_parameters=master_parameters,
                                        proximity=proximity,
                                        max_candidates=max_candidates)

        atoms_list = []
        for c_p in cell_parameters:
            atoms_list += [self.construct_atoms(c_p,
                                                primitive_cell=primitive_cell)]

        if return_parameters:
            return atoms_list, cell_parameters

        return atoms_list
