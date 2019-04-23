from ase import Atoms
from ase.db import connect
import sqlite3


class OqmdInterface:

    def __init__(self, dbfile):
        self.dbfile = dbfile

    def ase_db(self):
        self.db = connect(self.dbfile)

    def get_distinct_prototypes(self,
                                source=None,
                                formula=None,
                                repetition=None):
        """
        Parameters
        ----------
        source: str
          oqmd project name, such as 'icsd'
        formula: str
          stiochiometry of the compound, f.ex. 'AB2' or AB2C3
        repetition: int
          repetition of the stiochiometry    
        """

        con = self.db.connection or db._connect()
        cur = con.cursor()

        sql_command = \
            "select distinct value from text_key_values where key='proto_name'"
        if formula:
            if repetition:
                formula += '\_{}'.format(repetition)
            sql_command += " and value like '{}\_%' ESCAPE '\\'".format(
                formula)

        if source:
            sql_command += " and id in (select id from text_key_values where key='source' and value='icsd')"
        cur.execute(sql_command)

        prototypes = cur.fetchall()
        prototypes = [p[0] for p in prototypes]

        return prototypes

    def get_structures(self, source, formula, max_atoms=None):
        """
        Parameters
        ----------
        source: str
          oqmd project name, such as 'icsd'
        formula: str
          chemical formula with atoms such as 'TiO2'
        max_atoms: int
           maximum number of atoms in cell 
        """

        # Get prototype AB formula from chemical formula

        # Get repetition from max_atoms

        prototypes = get_distinct_prototypes(source=source,
                                             formula=prototype_formula,
                                             repetition=repetition)

        # Query db - is this structure present?

        # If not - replace atoms in list and get new lattice parameters

        ### return all atoms in list ###
        # Dummy atoms to test the workflow
        atoms_list = []
        for dummy_atom in self.db.select(source='icsd', limit=3):
            atoms_list += [dummy_atoms]

        return atoms_list
