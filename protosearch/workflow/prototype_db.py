import os
import sys
import subprocess
import json
import ase.db
from ase.db.sqlite import SQLite3Database
from ase.io import read
import sqlite3

from protosearch.utils import get_basepath
from protosearch.utils.standards import VaspStandards
from protosearch.build_bulk.classification import get_classification

init_commands = [
    """CREATE TABLE prototype (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name text UNIQUE,
    natom int,
    spacegroup int,
    npermutations int,
    permutations text,
    parameters text,
    species text,
    wyckoffs text);""",

    """CREATE TABLE jobs (
    id integer,
    parameters text,
    completed bool,
    FOREIGN KEY (id) REFERENCES prototype(id))
    """
]

columns = ['id', 'name', 'natom', 'spacegroup', 'npermutations', 'permutations',
           'parameters', 'species', 'wyckoffs']

json_columns = [5, 6, 7, 8]


class PrototypeSQL:
    def __init__(self,
                 filename=None,
                 stdin=sys.stdin,
                 stdout=sys.stdout):

        if filename:
            assert filename.endswith(
                '.db'), 'filename should have .db extension'
        else:
            basepath = get_basepath()
            filename = '/prototypes.db'
        self.filename = filename
        self.initialized = False
        self.default = 'NULL'
        self.connection = None
        self.stdin = stdin
        self.stdout = stdout

    def _connect(self):
        self.ase_db = ase.db.connect(self.filename)
        return sqlite3.connect(self.filename, timeout=600)

    def __enter__(self):
        """Set connection upon entry using with statement"""
        assert self.connection is None
        self.connection = self._connect()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        """Commit changes upon exit"""
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()
        self.connection = None

    def _initialize(self, con):
        """Set up tables in SQL"""
        if self.initialized:
            return

        SQLite3Database()._initialize(con)  # ASE db initialization

        cur = con.execute(
            'SELECT COUNT(*) FROM sqlite_master WHERE name="prototype"')

        if cur.fetchone()[0] == 0:  # no reaction table
            for init_command in init_commands:
                con.execute(init_command)  # Create tables
            con.commit()

        self.initialized = True

    def write_prototype(self, entry):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()

        species = []

        permutations = entry['specie_permutations']

        values = (entry['name'],
                  entry['natom'],
                  entry['spaceGroupNumber'],
                  len(permutations),
                  json.dumps(permutations),
                  json.dumps(entry['parameters']),
                  json.dumps(entry['species']),
                  json.dumps(entry['wyckoffs'])
                  )

        q = self.default + ',' + ', '.join('?' * len(values))
        cur.execute('INSERT or IGNORE INTO prototype VALUES ({})'.format(q),
                    values)
        if not self.connection:
            con.commit()
            con.close()

        return

    def select(self, **kwargs):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()

        query = ''
        for key, value in kwargs.items():
            query += ' {}={}'.format(key, value)

        cur.execute('SELECT * from prototype where {}'.format(query))
        data = cur.fetchall()

        result = []
        for l in range(len(data)):
            result += [{}]
            for i, k in enumerate(columns):
                if i in json_columns:
                    result[l].update({k: json.loads(data[l][i])})
                else:
                    result[l].update({k: data[l][i]})

        return result

    def write_result(self, path):
        atoms = read(path + '/OUTCAR')

    def get_prototype_id(self, p_name):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()
        cur.execute('SELECT id from prototype where name={};'.format(p_name))
        name = cur.fetchall()[0]

        return name

    def get_prototype_info(self, p_name):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()
        cur.execute(
            'SELECT spacegroup,wyckoffs,s from prototype where name={};'.format(p_name))
        name = cur.fetchall()[0]

        return name

    def is_calculated(self, formula, p_name):
        con = self.connection or self._connect()
        if self.ase_db.count(formula=formula,
                             p_name=p_name,
                             error=0) > 0:
            print('Allready calculated')
            return True
        else:
            return False
