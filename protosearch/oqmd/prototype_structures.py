__maintainer__ = "Meng Zhao"

import pprint
import csv
import json
import os,sys
import ase
from ase.db import connect
from ase.visualize import view

from protosearch.build_bulk.build_bulk import BuildBulk


database = connect('structures.db')
file_path = 'AB2_unique.csv'

csv_file = open(file_path)
data_complete = csv.reader(csv_file, delimiter=',')

# generate POSCAR scripts for bulk structures 
for i, row in enumerate(data_complete):
    if i < 1:
        continue

    print('---------- Structure {} ---------------'.format(i))

    cell_parameters = json.loads(row[2].replace("'", '"'))

    # Allow lattice constants to be optimized
    for del_param in ['a', 'b', 'c']:
        if del_param in cell_parameters:
            del cell_parameters[del_param]

    
    spacegroup = int(row[4])
    wyckoffs = eval(row[6])
    species = eval(row[5])
    atomA = species[0]

    # Change to TiO2 structure
    for i in range(len(species)):
        if species[i] == atomA:
            species[i] = 'Ti'
        else:
            species[i] = 'O'
    try: 
        bulk_generator = BuildBulk(spacegroup = int(row[4]),
                                   wyckoffs = eval(row[6]),
                                   species = species,
                                   cell_parameters=cell_parameters)
    
        atoms = bulk_generator.atoms
        
        if atoms:
            view(atoms)
            database.write(atoms)
    except BaseException:        
        print('Error: something is wrong with this structure')
        continue
