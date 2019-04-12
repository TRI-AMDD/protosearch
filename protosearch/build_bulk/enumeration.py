import bulk_enumerator as be
from protosearch.workflow.prototype_db import PrototypeSQL


class Enumeration():

    def __init__(self,
                 stoichiometry,
                 num_start,
                 num_end,
                 SG_start=1,
                 SG_end=230,
                 num_type='atom'):
        """
        Parameters

        stoichiometry: str
            Ratio bewteen elements separated by '_'. 
            For example: '1_2' or '1_2_3'

        num_start: int
            minimum number of atoms or wyckoff sites
        num_end: int
            maximum number of atoms or wyckoff sites
        SG_start: int
           minimum spacegroup number
        SG_end: int
           maximum spacegroup number
        num_type: str 
            'atom' or 'wyckoff'
        """

        self.stoichiometry = stoichiometry
        self.num_start = num_start
        self.num_end = num_end
        self.SG_start = SG_start
        self.SG_end = SG_end
        self.num_type = num_type

    def set_spacegroup(self, spacegroup):
        self.SG_start = spacegroup
        self.SG_end = spacegroup + 1

    def set_stoichiometry(self, stoichiometry):
        self.stoichiometry = stoichiometry

    def set_natoms(self, natoms):
        self.num_start = natoms
        self.num_end = natoms

    def get_enumeration(self):
        E = be.enumerator.ENUMERATOR()
        enumerations = E.get_bulk_enumerations(self.stoichiometry,
                                               self.num_start,
                                               self.num_end,
                                               self.SG_start,
                                               self.SG_end,
                                               self.num_type)

        return enumerations

    def store_enumeration(self, filename=None):

        enumerations = self.get_enumeration()

        with PrototypeSQL(filename=filename) as DB:
            for entry in enumerations:
                DB.write_prototype(entry=entry)
