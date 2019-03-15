from build_bulk.build_bulk import BuildBulk


BB = BuildBulk(spacegroup=221,
               wyckoffs=['a', 'd'],
               species=['Fe', 'O'],
               cell_parameters={'a': 3.7},
               calc_parameters={'encut': 300})

BB.submit_calculation()
