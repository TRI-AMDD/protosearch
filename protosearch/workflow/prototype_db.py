import os
import sys
import subprocess
import json
import numpy as np
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

    """CREATE TABLE fingerprint (
    id integer PRIMARY KEY,
    input text,
    output text,
    FOREIGN KEY (id) REFERENCES systems(id));
    """,

    """CREATE TABLE prediction (
    batch_no,
    ids text,
    energies text,
    uncertainties text,
    """,

    """CREATE TABLE enumeration (
    stoichiometry text,
    spacegroup int,
    number int,
    num_type text
    );""",

    """CREATE TABLE batch_status (
    batch_no INTEGER PRIMARY KEY AUTOINCREMENT,
    structure_ids text,
    completed_ids text
    );""",

    """CREATE TABLE status (
    id integer,
    chemical_formulas text,
    batch_size int,
    enumerated int,
    fingerprinted int,
    initialized int,
    last_batch_no int,
    n_completed int,
    n_errored int,
    most_stable_id int
    );"""
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
            filename = basepath + '/prototypes.db'
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
            'SELECT COUNT(*) FROM sqlite_master WHERE name="status"')

        if cur.fetchone()[0] == 0:  # no prototype table
            for init_command in init_commands:
                con.execute(init_command)  # Create tables
            con.commit()

        self.initialized = True

    def get_status(self):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()

        cur.execute('SELECT * from status;')

        status = cur.fetchall()[0]
        status_dict = {
            'chemical_formulas': status[0],
            'batch_size': status[1],
            'enumerated': status[2],
            'fingerprinted': status[3],
            'initialized': status[4],
            'last_batch_no': status[5],
            'n_completed': status[6],
            'n_errored': status[7],
            'most_stable_id': status[8]}
        return status_dict

    def write_status(self, **args):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()
        query = "INSERT into status (id) VALUES (0)"
        cur.execute(query)

        for key, value in args.items():
            query = "UPDATE status SET {}=".format(key)

            if isinstance(value, str):
                query += "'{}'".format(value)

            if isinstance(value, list):
                query += "'{}'".format(json.dumps(value))

            else:
                query += "{}".format(value)
            query += ' where id=0'
            cur.execute(query)

        if not self.connection:
            con.commit()
            con.close()
        return

    def write_job_status(self):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()

        cur.execute(
            "SELECT count(id) from number_key_values where key='relaxed' and value=1")
        n_completed = cur.fetchall()[0][0]
        cur.execute(
            "SELECT count(id) from number_key_values where key='completed' and value=-1")
        n_errored = cur.fetchall()[0][0]

        self.write_status(n_completed=n_completed, n_errored=n_errored)

        if not self.connection:
            con.commit()
            con.close()
        return

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

    def write_batch_status(self, batch_no, structure_ids):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()

        structure_ids = json.dumps(structure_ids)

        cur.execute("INSERT into batch_status (batch_no, structure_ids) VALUES ({}, '{}')".
                    format(batch_no, structure_ids))

        if not self.connection:
            con.commit()
            con.close()

        return

    def write_enumerated(self, stoichiometry, spacegroup, number, num_type):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()

        cur.execute(
            """INSERT into enumeration (stoichiometry, spacegroup, number, num_type) 
            VALUES ('{}', {}, {}, '{}')""".
            format(stoichiometry, spacegroup, number, num_type)
        )

        if not self.connection:
            con.commit()
            con.close()
        return

    def update_status_complete(self, batch_no, structure_ids):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()

        structure_ids = json.dumps(structure_ids)

        cur.execute("UPDATE status SET batch_no={}, structure_ids='{}' where id = 0".
                    format(batch_no, structure_ids))

        if not self.connection:
            con.commit()
            con.close()

        return

    def select_atoms(self, **kwargs):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()

        atoms_list = []
        for row in self.ase_db.select(**kwargs):
            atoms_list += row.toatoms()

        return atoms_list

    def get_atom_by_id(self, id):
        con = self.connection or self._connect()
        self._initialize(con)
        row = self.ase_db.get(id=id)
        return row.toatoms()

    def get_atoms_list(self, ids):
        con = self.connection or self._connect()
        self._initialize(con)
        atoms_list = []
        for id in ids:
            atoms_list += [self.ase_db.get(id=int(id)).toatoms()]
        return atoms_list

    def select(self, **kwargs):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()

        query = ''
        for key, value in kwargs.items():
            query += ' {}={}'.format(key, value)
        statement = 'SELECT * from prototype'
        if query:
            statement += 'where {}'.format(query)
        cur.execute(statement)
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
            'SELECT spacegroup,wyckoffs,from prototype where name={};'.format(p_name))
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

    def get_next_batch_no(self):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()
        cur.execute(
            "SELECT max(value) from number_key_values where key='batch'")
        cur_batch = cur.fetchall()
        if cur_batch:
            if cur_batch[0][0]:
                return cur_batch + 1
            else:
                return 0
        else:
            return 0

    def get_structure_ids(self, start_id=1, n_ids=None):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()
        query = "SELECT id from systems where id>{}".format(start_id-1)
        if n_ids:
            query += ' limit {}'.format(n_ids)
        cur.execute(query)
        ids = cur.fetchall()
        ids = [i[0] for i in ids]
        return ids

    def save_fingerprint(self, id, input_data, output_data=None):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()
        cur.execute('SELECT count(id) from fingerprint where id={}'.format(id))
        count = cur.fetchall()[0][0]

        input_data = json.dumps(input_data.tolist())
        columns = ['id', 'input']
        #columns = 'id, input'
        values = [id, input_data]
        #values = "{}, '{}'".format(id, input_data)

        if output_data:
            output_data = json.dumps(output_data)
            columns += ['output']
            values += [output_data]  # ", '{}'".format(output_data)

        values = ['{}'.format(v) if isinstance(v, int) else "'{}'".format(v)
                  for v in values]
        if count == 0:
            columns = ','.join(columns)
            values = ','.join(values)
            cur.execute(
                'INSERT INTO fingerprint ({}) VALUES ({})'.format(columns, values))
        elif count == 1:
            collist = ['{}={}'.format(columns[i], values[i])
                       for i in range(1, len(columns))]
            collist = ','.join(collist)
            cur.execute(
                "UPDATE fingerprint set {} where id={}".format(collist, id))
        con.commit()
        con.close()

    def get_fingerprints(self, ids):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()
        ids = sorted(ids)
        ids = [str(i) for i in ids]
        id_str = ','.join(ids)
        cur.execute(
            'SELECT * from fingerprint where id in ({}) order by id'.format(id_str))
        data = cur.fetchall()
        feature_matrix = []
        target_list = []
        for d in data:
            feature_matrix += [json.loads(d[1])]
            if d[2]:
                target_list += [json.loads(d[2]).get('Ef', None)]
            else:
                target_list += [None]
        return np.array(feature_matrix), np.array(target_list)

    def get_new_fingerprint_ids(self, completed=True):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()

        cur.execute('SELECT id from fingerprint where output is not null')
        data = cur.fetchall()
        ids = [i[0] for i in ids]
        return ids

    def save_predictions(self, ids, Efs, var):
        # Just save formation energy and uncertainty for now
        # link to structure id
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()
        for i, id in enumerate(ids):
            cur.execute('INSERT INTO prediction VALUES ({}, {}, {})'.format(
                id, Efs[i], var[i]))

        con.commit()
        con.close()

    # def read_predictions(self, ids, EFs, var):
