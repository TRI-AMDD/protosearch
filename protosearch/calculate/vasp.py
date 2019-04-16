import io
import numpy as np
from ase.io.vasp import write_vasp

from protosearch.utils.standards import VaspStandards, CommonCalc
from protosearch.utils.valence import VaspValence


class VaspModel:
    def __init__(self,
                 calc_parameters,
                 symbols,
                 ncpus):

        self.calc_parameters = VaspStandards.calc_parameters
        if calc_parameters:
            self.calc_parameters.update(calc_parameters)
        self.natoms = len(symbols)
        self.symbols = set(symbols)
        self.ncpus = ncpus
        self.setups = {}
        U_luj = CommonCalc.U_luj
        ldau_luj = {}
        for symbol in symbols:
            if symbol in VaspStandards.paw_potentials:
                self.setups.update({symbol:
                                    VaspStandards.paw_potentials[symbol]})

        if np.any([t in symbols for t in CommonCalc.U_trickers]) and \
           np.any([t in symbols for t in CommonCalc.U_metals]):
            for symbol in symbols:
                ldau_luj.update({symbol: U_luj[symbol]})

        if ldau_luj:
            self.calc_parameters.update({'ldau_luj': ldau_luj})
            self.all_parameters = VaspStandards.sorted_calc_parameters + \
                VaspStandards.u_parameters
        else:
            self.all_parameters = VaspStandards.sorted_calc_parameters

        self.calc_values = []
        for param in self.all_parameters:
            self.calc_values += [self.calc_parameters[param]]

        nbands_index = self.all_parameters.index('nbands')
        nbands = self.calc_values[nbands_index]
        if nbands < 0:
            self.calc_values[nbands_index] = self.get_nbands(
                n_empty=abs(nbands))
        else:
            self.calc_values[nbands_index] = self.get_nbands()

    def get_parameters(self):
        return self.all_parameters, self.calc_values

    def get_parameter_dict(self):
        return self.calc_parameters

    def get_parametrized_model(self):
        """
        Construct model string, which uses the ASE interface
        to Vasp.
        Since a parameterized model will be submitted with trisub, 
        the parameters are given the values #1, #2, #3... etc       
        """
        modelstr = get_model_header()

        if self.setups:
            modelstr += 'setups = {'
            for symbol, setup in self.setups.items():
                modelstr += "'{}': '{}',".format(symbol, setup)
            modelstr = modelstr[:-1]
            modelstr += '}\n\n'
        else:
            modelstr += 'setups = {}\n\n'

        for i, param in enumerate(self.all_parameters):
            factor = VaspStandards.calc_decimal_parameters.get(param, None)
            value = self.calc_values[i]
            if factor:
                modelstr += '{} = #{} * {}\n'.format(param, i+1, factor)
            elif isinstance(value, str):
                modelstr += "{} = '#{}'\n".format(param, i+1)
            elif isinstance(value, dict):
                modelstr += '{}='.format(param)
                modelstr += '{'
                nkeys = len(value)
                i = 1
                for k, v in value.items():
                    modelstr += "'{}': {}".format(k, v)
                    if i < nkeys:
                        modelstr += ',\n' + ' ' * (len(param) + 2)
                    i += 1
                modelstr += '}\n'

            else:
                modelstr += '{} = #{}\n'.format(param, i+1)

        # if self.calc_parameters.get('ldau_luj', None):
        #    for param in VaspStandards.u_parameters:
        #        value = self.calc_parameters[param]
        #        modelstr += '{} = {}\n'.format(param, value)
        #    all_parameters = VaspStandards.sorted_calc_parameters + \
        #        VaspStandards.u_parameters
        # else:
        #    all_parameters = VaspStandards.sorted_calc_parameters

        modelstr += '\ncalc = Vasp(\n'
        modelstr += '    setups=setups,\n'
        for i, param in enumerate(self.all_parameters):
            modelstr += '    {}={},\n'.format(param, param)

        modelstr += ')\n\ncalc.calculate(atoms)\n'
        return modelstr

    def get_model(self):
        """
        Construct model string, which uses the ASE interface
        to Vasp.
        """

        all_parameters = self.collect_parameters()

        modelstr = get_model_header()
        modelstr += 'calc = Vasp(\n'

        if self.setups:
            modelstr += '    setups={'
            for symbol, setup in self.setups.items():
                modelstr += "'{}': '{}'".format(symbol, setup)
            modelstr += '},\n'

        for param in all_parameters:
            value = self.calc_parameters[param]
            factor = VaspStandards.calc_decimal_parameters.get(param, None)
            if factor:
                modelstr += '    {}={},\n'.format(param,
                                                  round(factor * value, 8))
            elif isinstance(value, str):
                modelstr += "    {}='{}',\n".format(param, value)
            elif isinstance(value, dict):
                modelstr += '    {}='.format(param)
                modelstr += '{'
                nkeys = len(value)
                i = 1
                for k, v in value.items():
                    modelstr += "'{}': {}".format(k, v)
                    if i < nkeys:
                        modelstr += ',\n' + ' ' * (len(param) + 6)
                    i += 1
                modelstr += '},\n'
            else:
                modelstr += '    {}={},\n'.format(param, value)

        modelstr += '    )\n\ncalc.calculate(atoms)\n'
        return modelstr

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


def get_poscar_from_atoms(atoms):
    poscar = io.StringIO()
    write_vasp(filename=poscar, atoms=atoms, vasp5=True,
               long_format=False, direct=True)

    return poscar.getvalue()


def get_model_header():
    modelstr = ''
    modelstr = '#!/usr/bin/env python\n\n'
    modelstr += 'from ase.io import read\n' + \
        'from ase.units import Rydberg as Ry\n' + \
        'from ase.calculators.vasp import Vasp\n\n'

    modelstr += "atoms = read('initial.POSCAR')\n\n"

    return modelstr
