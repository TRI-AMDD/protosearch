import bulk_enumerator as be
from protosearch.workflow.prototype_db import PrototypeSQL


class Enumeration(PrototypeSQL):

    def __init__(self,
                 stoichiometry,
                 num_start,
                 num_end,
                 SG_start=1,
                 SG_end=230,
                 num_type='atom'):
        """
        # TODO: clarify num_start, num_end, num_type

        Args:
            stoichiometry (str): chemical formula
            num_start (int): minimum number of atoms or wyckoff sites?
            num_end (int): maximum number of atoms or wyckoff sites?
            SG_start (int): spacegroup number minimum
            SG_end (int): spacegroup number maximum
            num_type (int): whether the num is atoms or wyckoff?
        """

        super().__init__()
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

    def store_enumeration(self):

        enumerations = self.get_enumeration()

        with PrototypeSQL() as DB:
            for entry in enumerations:
                self.write_prototype(entry=entry)