from ase.db import connect
import sqlite3


class OqmdInterface:

    def __init__(self, dbfile):
        self.dbfile = dbfile

    def get_distinct_prototypes(self,
                                formula=None,
                                repetition=None):
        """
        Parameters
        ----------
        formula: str
          stiochiometry of the compound, f.ex. 'AB2' or AB2C3
        repetition: int
          repetition of the stiochiometry    
        """

        db = connect(self.dbfile)

        con = db.connection or db._connect()
        cur = con.cursor()

        sql_command = \
            "select distinct value from text_key_values where key='proto_name'"
        if formula:
            if repetition:
                formula += '\_{}'.format(repetition)
            sql_command += "and value like '{}\_%' ESCAPE '\\'".format(formula)

        cur.execute(sql_command)

        prototypes = cur.fetchall()
        prototypes = [p[0] for p in prototypes]

        return prototypes
