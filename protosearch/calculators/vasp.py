from protosearch.utils.standards import VaspStandards
from protosearch.utils.valence import VaspValence

class VaspModel:
    def __init__(self,
                 calc_parameters,
                 symbols,
                 ncpus):

        self.calc_parameters = calc_parameters
        self.natoms = len(symbols)
        self.symbols = set(symbols)
        self.ncpus = ncpus
        self.setups = {}
        for symbol in symbols:
            if symbol in VaspStandards.paw_potentials:
                self.setups.update({symbol: Standards.paw_potentials[symbol]})

        self.calc_value_list = []
        for param in VaspStandards.sorted_calc_parameters:
            if param in self.calc_parameters:
                self.calc_value_list += [self.calc_parameters[param]]
            else:
                self.calc_value_list += [VaspStandards.calc_parameters[param]]

        nbands_index = VaspStandards.sorted_calc_parameters.index('nbands')
        nbands = self.calc_value_list[nbands_index]
        if nbands < 0:
            self.calc_value_list[nbands_index] = self.get_nbands(
                n_empty=abs(nbands))
        else:
            self.calc_value_list[nbands_index] = self.get_nbands()            
                
    def get_model(self):
        """
        Construct model string, which uses the ASE interface
        to Vasp.
        Since a parameterized model will be submitted with trisub, 
        the parameters are given the values #1, #2, #3... etc       
        """
        modelstr = '#!/usr/bin/env python\n\n'
        modelstr += 'from ase.io import read\n' + \
            'from ase.units import Rydberg as Ry\n' + \
            'from ase.calculators.vasp import Vasp\n\n'

        modelstr += "atoms = read('initial.POSCAR')\n\n"

        for i, calc_key in enumerate(VaspStandards.sorted_calc_parameters):
            factor = None
            if calc_key in VaspStandards.calc_decimal_parameters:
                factor = VaspStandards.calc_decimal_parameters[calc_key]
            if factor:
                modelstr += '{} = #{} * {}\n'.format(calc_key, i+1, factor)

            elif isinstance(self.calc_value_list[i], str):
                modelstr += "{} = '#{}'\n".format(calc_key, i+1)
            else:
                modelstr += '{} = #{}\n'.format(calc_key, i+1)

        modelstr += 'setups = {\n'
        for symbol, setup in self.setups.items():
            modelstr += "    '{}': '{}'\n".format(symbol, setup)
        modelstr += '}\n'

        modelstr += '\ncalc = Vasp(\n'
        modelstr += '    setups=setups,\n'
        for i, calc_key in enumerate(VaspStandards.sorted_calc_parameters):
            modelstr += '    {}={},\n'.format(calc_key, calc_key)

        modelstr += ')\n\ncalc.calculate(atoms)\n'
        return modelstr, self.calc_value_list
        

    def get_nbands(self, n_empty=5):
        """ get the number of bands from structure, based on the number of
        valence electrons listed in utils/valence.py"""

        N_val = 0

        for symbol in self.symbols:
            if symbol in self.setups:
                setup = self.setups[symbol]
            else:
                setup = 's'
            N_val += VaspValence.__dict__.get(symbol)[setup]

        nbands = int(N_val / 2 + self.natoms / 2 + n_empty)
        nbands += nbands % self.ncpus

        return nbands
        
