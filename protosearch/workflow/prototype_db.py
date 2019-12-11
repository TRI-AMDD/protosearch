import os
import sys
import subprocess
import json
import numpy as np
import pandas as pd
import ase.db
from ase.db.sqlite import SQLite3Database
from ase.io import read
import sqlite3

from protosearch.utils import get_basepath
from protosearch.utils.standards import VaspStandards

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

    """CREATE TABLE prediction (
    batch_no,
    id integer,
    energy real,
    uncertainty real,
    FOREIGN KEY (id) REFERENCES systems(id));
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
    );""",

    """INSERT into status
    (id, enumerated, fingerprinted, initialized, n_completed, n_errored)
    VALUES
    (0, 0, 0, 0, 0, 0);"""
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
                print(init_command)
                try:
                    con.execute(init_command)  # Create tables
                except:
                    pass
            con.commit()

        self.initialized = True

    def get_status(self):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()

        cur.execute('SELECT * from status;')

        status = cur.fetchall()[0]
        status_dict = {
            'chemical_formulas': status[1],
            'batch_size': status[2],
            'enumerated': status[3],
            'fingerprinted': status[4],
            'initialized': status[5],
            'last_batch_no': status[6],
            'n_completed': status[7],
            'n_errored': status[8],
            'most_stable_id': status[9]}
        return status_dict

    def write_status(self, **args):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()

        for key, value in args.items():
            query = "UPDATE status SET {}=".format(key)

            if isinstance(value, str):
                query += "'{}'".format(value)
            elif isinstance(value, list):
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

    def is_enumerated(self, stoichiometry, spacegroup, number, num_type):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()
        cur.execute(
            """SELECT count(*) from enumeration where stoichiometry='{}' and spacegroup={} and
            number={} and num_type='{}';""".format(stoichiometry, spacegroup, number, num_type))

        count = cur.fetchall()[0]

        return count

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
            if value is None:
                continue
            if key == 'max_atoms':
                query += "natom<={}".format(value)
            elif key == 'spacegroups':
                if len(query) > 0:
                    query += ' and'
                value = [str(v) for v in value]
                sg_str = '(' + ','.join(value) + ')'
                query += " spacegroup in {}".format(sg_str)
            else:
                query += ' {}={}'.format(key, value)
        statement = 'SELECT * from prototype'
        if query:
            statement += ' where {}'.format(query)

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
        if self.ase_db.count('completed>-1',
                             submitted=1,
                             formula=formula,
                             p_name=p_name) > 0:
            print('Allready calculated')
            return True
        else:
            return False

    def is_calculated_id(self, id):
        con = self.connection or self._connect()
        if self.ase_db.count('completed>-1',
                             id=id,
                             p_name=p_name) > 0:
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
                return cur_batch[0][0] + 1
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

    def get_completed_structure_ids(self, max_energy=0):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()
        query = \
            """select distinct id from number_key_values
            where key='relaxed' and value=1
            and id not in (SELECT distinct id from systems where energy > {})
            order by id""".format(max_energy)
        cur.execute(query)
        ids = cur.fetchall()
        ids = [i[0] for i in ids]
        return ids

    def get_uncompleted_structure_ids(self, unsubmitted_only=True):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()
        if unsubmitted_only:
            query =\
                """SELECT distinct id from number_key_values
                where key='submitted' and value=0 and id not in
                (SELECT distinct value from number_key_values
                where key='initial_id') order by id"""
        else:
            query =\
                """SELECT distinct id from number_key_values
                where key='completed' and value=0 and id not in
                (SELECT distinct value from number_key_values
                where key='initial_id') order by id"""

        cur.execute(query)
        ids = cur.fetchall()
        ids = [i[0] for i in ids]
        return ids

    def get_initial_structure_ids(self, completed=False):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()
        if completed:
            query =\
                """select distinct id from number_key_values
                where key='relaxed' and value=0 and id in 
                (select distinct id from number_key_values
                where key='completed' and value=1)
                order by id"""
        else:
            query =\
                """select distinct id from number_key_values
                where key='relaxed' and value=0
                order by id"""
        cur.execute(query)
        ids = cur.fetchall()
        ids = [i[0] for i in ids]
        return ids

    def write_dataframe(self, table, df):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()
        ids = df['id'].values
        id_string = ', '.join([str(i) for i in ids])
        cur.execute(
            """SELECT count(*) FROM sqlite_master WHERE
            type='table' AND name='{}';""".format(table))
        table_count = cur.fetchall()[0][0]
        if table_count:
            cur.execute(
                'DELETE from {} where id in ({})'.format(table, id_string))
        df.to_sql(table, con=con, index=False,
                  index_label=None, if_exists='append')

    def load_dataframe(self, table, ids=None):
        con = self.connection or self._connect()
        self._initialize(con)

        query = 'SELECT * from {}'.format(table)
        if ids is not None:
            ids = sorted(ids)
            id_str = ','.join([str(i) for i in ids])
            query += ' where id in ({})'.format(id_str)
        query += ' order by id'
        df = pd.read_sql_query(query, con)
        return df

    def write_fingerprint(self, id, input_data, output_data=None):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()

        cur.execute('DELETE from fingerprint where id={}'.format(id))

        input_data = json.dumps(input_data.tolist())
        columns = ['id', 'input']
        values = [id, input_data]

        if output_data:
            output_data = json.dumps(output_data)
            columns += ['output']
            values += [output_data]

        values = ['{}'.format(v) if isinstance(v, int) else "'{}'".format(v)
                  for v in values]

        columns = ','.join(columns)
        values = ','.join(values)
        cur.execute(
            'INSERT INTO fingerprint ({}) VALUES ({})'.format(columns, values))
        con.commit()
        con.close()

    def get_new_fingerprint_ids(self, completed=True):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()
        cur.execute(
            """SELECT count(*) FROM sqlite_master WHERE
            type='table' AND name='target';""")
        table_count = cur.fetchall()[0][0]

        if completed:
            query = 'SELECT id from systems where energy is not null'
            if table_count:
                query += ' and id not in (SELECT distinct id from target)'
        else:
            query = \
                """SELECT id from systems"""
            if table_count:
                query += ' where id not in (SELECT distinct id from fingerprint)'
        cur.execute(query)
        ids = cur.fetchall()
        ids = [i[0] for i in ids]
        return ids

    def write_predictions(self, batch_no, ids, Efs, var):
        # Just save formation energy and uncertainty for now
        # link to structure id
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()

        cur.execute('DELETE FROM prediction WHERE batch_no={}'.format(batch_no))
        for i, idi in enumerate(ids):
            cur.execute('INSERT INTO prediction VALUES ({}, {}, {}, {})'.format(
                batch_no, idi, float(Efs[i]), float(var[i])))

        con.commit()
        con.close()

    def get_predictions(self, batch_no=None):
        con = self.connection or self._connect()
        self._initialize(con)
        cur = con.cursor()
        if not batch_no:
            cur.execute('SELECT distinct batch_no FROM prediction')
            batch_nos = cur.fetchall()
            batch_nos = [b[0] for b in batch_nos]
        else:
            batch_nos = [batch_no]

        predictions = []
        for b in batch_nos:
            cur.execute(
                'SELECT * FROM prediction where batch_no={};'.format(b))
            data = cur.fetchall()
            ids = [d[1] for d in data]
            energies = np.array([float(d[2]) for d in data])
            var = np.array([float(d[3]) for d in data])
            predictions += [{'batch_no': b,
                             'ids': ids,
                             'energies': energies,
                             'vars': var}]
        return predictions

    def get_pandas_tables(self,
                          write_csv_tables=False,
                          tables_list=None):
        """Convert SQL tables to Pandas dataframes.

        Parameters:
        ----------
        write_csv_tables: Bool
            Write data tables to .csv files to current working dir
        tables_list: list or None
            List of table names to retrieve if you don't want all of them
            tables_list = [
                'systems',
                'sqlite_sequence',
                'species',
                'keys',
                'text_key_values',
                'number_key_values',
                'information',
                'prototype',
                'fingerprint',
                'target'
                'prediction',
                'enumeration',
                'batch_status',
                'status']
        """
        db = sqlite3.connect(self.filename)
        cursor = db.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        tables_dict = {}
        for table_name in tables:
            table_name = table_name[0]

            if tables_list is not None:
                if table_name in tables_list:
                    table_i = get_table(table_name, db, write_csv_tables)
                    tables_dict[table_name] = table_i
            else:
                table_i = get_table(table_name, db, write_csv_tables)
                tables_dict[table_name] = table_i

        cursor.close()
        db.close()

        # tables = to_csv(db_file=self.filename)
        return(tables_dict)


def get_table(table_name, db, write_csv_tables):
    table = pd.read_sql_query("SELECT * from %s" % table_name, db)
    # tables_dict[table_name] = table
    if write_csv_tables:
        table.to_csv(table_name + '.csv', index_label='index')
    return(table)
