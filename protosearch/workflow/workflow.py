import os
import json
import subprocess
import ase
from ase.io import read
import bulk_enumerator as be
from build_bulk.build_bulk import BuildBulk
from protosearch.utils.standards import Standards
from .prototype_db import PrototypeSQL
from .classification import get_classification


class Workflow(PrototypeSQL):

    def __init__(self):
        super().__init__()
        TRI_PATH = os.environ['TRI_PATH']
        username = os.environ['TRI_USERNAME']
        self.basepath = TRI_PATH + '/model/vasp/1/u/{}'.format(username)

        subprocess.call('trisync', cwd=self.basepath)

    def collect(self):
        self.check_submissions()
        self.rerun_failed_calculations()

    def submit(self, prototype, ncpus=None, calc_parameters=None):

        BB = BuildBulk(prototype['spacegroup'],
                       prototype['wyckoffs'],
                       prototype['species'],
                       ncpus=ncpus,
                       calc_parameters=calc_parameters)

        BB.submit_calculation()

        key_value_pairs = {'p_name': BB.prototype_name,
                           'path': BB.excpath,
                           'spacegroup': BB.spacegroup,
                           'wyckoffs': json.dumps(BB.wyckoffs),
                           'species': json.dumps(BB.species)}

        self.write_submission(key_value_pairs)

    def submit_enumerated(self, map_species, selection={}):
        """
        map_species = {'A0': 'Fe', 'A1': 'O'} 

        selection = {'spacegroup': 166, natom=3}
        """

        prototypes = self.select(**selection)

        for prototype in prototype:
            self.submit(prototype)

    def write_submission(self, key_value_pairs):
        con = self.connection or self._connect()
        self._initialize(con)

        atoms = read(key_value_pairs['path'] + '/initial.POSCAR')

        key_value_pairs.update({'relaxed': 0,
                                'completed': 0,
                                'submitted': 1})

        self.ase_db.write(atoms, key_value_pairs)

    def check_submissions(self):
        con = self.connection or self._connect()
        self._initialize(con)

        completed = 0
        errored = 0

        for d in self.ase_db.select(completed=0):
            print('hep')
            path = d.path + '/simulation'
            calcid = d.id
            for root, dirs, files in os.walk(path):
                if 'INCAR' in files and 'finalized' in files:
                    # Finished running

                    if self.ase_db.count(runpath=root) > 0:
                        print('Already in db')
                        continue

                    param_dict = {}
                    with open(root + '/param', 'r') as f:
                        param = f.read().lstrip('/').rstrip('\n')
                        print(param)
                        param_values = param.split('/')
                        for i, param_key in enumerate(Standards.sorted_calc_parameters):
                            param_dict[param_key] = param_values[i]

                    if 'completed' in files:
                        # Calculation completed - now save everything
                        self.ase_db.update(id=calcid,
                                           completed=1)
                        completed += 1

                    elif 'err' in files:
                        # Job crashed
                        key_value_pairs = {'error': 1,
                                           'runpath': root}

                        atoms = ase.io.read(root + '/initial.POSCAR')
                        prototype, cell_parameters = get_classification(atoms)

                        print(cell_parameters)
                        key_value_pairs.update(prototype)
                        key_value_pairs.update(cell_parameters)
                        key_value_pairs.update(param_dict)

                        key_value_pairs = clean_key_value_pairs(
                            key_value_pairs)
                        print(key_value_pairs)
                        with open(root + '/err', 'r') as errorf:
                            message = errorf.read().replace("'", '')
                            #data = json.dumps({'error': message})
                            # print(data)

                        self.ase_db.update(id=calcid,
                                           **key_value_pairs,
                                           data={'error': message})
                        errored += 1
                        continue

                    atoms = ase.io.read(root + '/OUTCAR')

                    prototype, cell_parameters = get_classification(atoms)

                    key_value_pairs = {'relaxed': 1,
                                       'completed': 1,
                                       'submitted': 1,
                                       'path': path,
                                       'runpath': root}

                    key_value_pairs.update(prototype)
                    key_value_pairs.update(cell_parameters)
                    key_value_pairs.update(param_dict)

                    key_value_pairs = clean_key_value_pairs(key_value_pairs)

                    atoms = set_calculator_info(atoms, param_dict)

                    self.ase_db.write(atoms, key_value_pairs)

        print('Found {} completed and {} failed calculations'.
              format(completed, errored))
        return

    def rerun_failed_calculations(self):
        con = self.connection or self._connect()
        self._initialize(con)

        for d in self.ase_db.select(error=1):
            if 'Vasp exited' in d.data['error']:
                ncpus = 8
            else:
                ncpus = 1

            p_name = d.p_name
            path = d.path + '/simulation'

            calc_parameters = {}
            for param in Standards.sorted_calc_parameters:
                if d.get(param, None):
                    calc_parameters.update({param: d.get(param)})

            self.submit({'spacegroup': d.spacegroup,
                         'wyckoffs': json.loads(d.wyckoffs),
                         'species': json.loads(d.species)},
                        ncpus=ncpus,
                        calc_parameters=calc_parameters)

            # Treat as completed, now it's resubmitted
            self.ase_db.update(id=d.id, completed=1)


def clean_key_value_pairs(key_value_pairs):
    print('hep')
    for key, value in key_value_pairs.items():
        if isinstance(value, list):
            key_value_pairs[key] = json.dumps(value)
        try:
            value = int(value)
            key_value_pairs[key] = value
        except:
            pass
    return key_value_pairs


def set_calculator_info(atoms, parameters):

    atoms.calc.name = 'vasp.5.4.4.18Apr17-6-g9f103f2a35'
    atoms.calc.parameters = parameters

    return atoms
