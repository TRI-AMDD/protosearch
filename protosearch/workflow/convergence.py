import pylab as p
import numpy as np
from protosearch.utils.standards import VaspStandards

from .workflow import Workflow, PrototypeSQL


class Convergence(Workflow):
    """TO DO: Automated convergence check """

    def __init__(self,
                 calculator='vasp',
                 basepath_ext=None):

        super().__init__(calculator=calculator,
                         basepath_ext=basepath_ext)
        self._connect()

        self.fixed_convergence_params = ['encut', 'kspacing', 'ismear']

        self.fix_params = {}
        for p in self.fixed_convergence_params:
            self.fix_params.update({p: VaspStandards.calc_parameters[p]})
        # index for highest converged point in numerically sorted array
        self.convergence_index = {'encut': -1, 'kspacing': 0, 'ismear': 0}

    def get_parametrized_energies(self, check_parameters=['encut', 'kspacing']):

        total_energies = {}
        for param in check_parameters:
            param_st = VaspStandards.calc_parameters[param]
            prototypes = []
            for d in self.ase_db.select('{}!={}'.format(param, param_st)):
                if not d.p_name + '|' + d.formula in prototypes:
                    prototypes.append(d.p_name + '|' + d.formula)

            total_energies[param] = {}
            for proto in prototypes:
                total_energies[param][proto] = {}
                name, formula = proto.split('|')
                fix_params = self.fix_params.copy()
                del fix_params[param]
                for d in self.ase_db.select(p_name=name,
                                            relaxed=1,
                                            formula=formula,
                                            **fix_params):

                    value = d.get(param, None)
                    total_energies[param][proto][value] = d.energy

        return total_energies

    def plot_convergence(self, check_parameters=['encut', 'kspacing']):

        total_energies = self.get_parametrized_energies()

        for param, proto_dict in total_energies.items():
            p.figure()
            p.title(param)
            for proto, points in proto_dict.items():
                x = np.array(list(points.keys()))
                y = np.array(list(points.values()))

                if len(x) <= 1:
                    continue

                index = np.argsort(x)
                x = x[index]
                y = y[index]

                y -= y[self.convergence_index[param]]
                p.plot(x, np.zeros_like(x), 'r--')

                p.plot(x, y, '-*', label=proto)
                p.ylim([-0.05, 0.05])
                p.legend()
        p.show()

    def submit_convergence_check(self,
                                 prototype,
                                 check_parameters=['encut',
                                                   'kspacing',
                                                   'sigma']):

        parameter_mapping = {'encut': [300, 400, 500, 525, 550,
                                       600, 700, 800],
                             'kspacing': [10, 15, 20, 25, 30, 40],
                             'sigma': [0.05, 0.1, 0.15, 0.2]}

        for param in check_parameters:
            for value in parameter_mapping[param]:
                self.submit(prototype, calc_parameters={param: value})
