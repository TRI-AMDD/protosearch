class CellStandards():

    sorted_cell_parameters = ['a', 'b/a', 'c/a', 'alpha', 'beta', 'gamma']


class VaspStandards():
    # Parameters to that will be tracked in parameterized model
    sorted_calc_parameters = ['xc', 'encut', 'nbands', 'ispin', 'kspacing',
                              'kgamma', 'ismear', 'sigma', 'ibrion', 'isif',
                              'nsw', 'nelm', 'ediff', 'ediffg']

    fixed_parameters = ['prec', 'algo', 'lwave']

    u_parameters = ['ldau', 'lmaxmix', 'ldautype', 'ldau_luj']

    """Parameters are heavily inspired by MP standard settings at
    https://github.com/materialsproject/pymatgen/blob/master/pymatgen/
    io/vasp/MPRelaxSet.yaml
    """

    # is kspacing = 0.25 compatible with materials project reciprocal_density = 64?
    # (1 / 64) ** (1/3) = 0.25  ?
    calc_parameters = {'xc': 'pbe',
                       'encut': 520,  # energy cutoff for plane waves
                       'nbands': -5,  # number of bands / empty bands
                       'ispin': 2,  # number of spins
                       'kspacing': 25,  # kspacing in units of 0.01
                       'kgamma': True,  # include gamma point
                       'ismear': -5,  # smearing function
                       'sigma': 5,  # k-point smearing in units of 0.01
                       'ibrion': 2,  # ion dynamics
                       'isif': 3,  # degrees of freedom to relax
                       'nsw': 99,  # maximum number of ionic steps
                       'nelm': 100,  # maximum number of electronic steps
                       'ediff': 10,  # sc accuracy in units of 1e-6
                       'ediffg': 20,  # force convergence in units of 1e-3
                       'prec': 'Accurate',  # Precision
                       'algo': 'Fast',  # optimization algorithm
                       'lwave': False,  # save wavefunctions or not
                       'ldau': True,  # USE U
                       'lmaxmix': 4,
                       'ldautype': 2,
                       'ldau_luj': {},
                       }

    # parameters are submitted as an integer,
    # that will be multiplied by the standard below
    calc_decimal_parameters = {'kspacing': 0.01,
                               'sigma': 0.01,
                               'ediff': 1e-6,
                               'ediff': -1e-3,
                               }

    paw_potentials = {'Li': '_sv',
                      'Na': '_pv',
                      'K': '_sv',
                      'Ca': '_sv',
                      'Sc': '_sv',
                      'Ti': '_sv',
                      'V': '_sv',
                      'Cr': '_pv',
                      'Mn': '_pv',
                      'Ga': '_d',
                      'Ge': '_d',
                      'Rb': '_sv',
                      'Sr': '_sv',
                      'Y': '_sv',
                      'Zr': '_sv',
                      'Nb': '_sv',
                      'Mo': '_sv',
                      'Tc': '_pv',
                      'Ru': '_pv',
                      'Rh': '_pv',
                      'In': '_d',
                      'Sn': '_d',
                      'Cs': '_sv',
                      'Ba': '_sv',
                      'Pr': '_3',
                      'Nd': '_3',
                      'Pm': '_3',
                      'Sm': '_3',
                      'Eu': '_2',
                      'Gd': '_3',
                      'Tb': '_3',
                      'Dy': '_3',
                      'Ho': '_3',
                      'Er': '_3',
                      'Tm': '_3',
                      'Yb': '_2',
                      'Lu': '_3',
                      'Hf': '_pv',
                      'Ta': '_pv',
                      'W': '_pv',
                      'Tl': '_d',
                      'Pb': '_d',
                      'Bi': '_d',
                      'Po': '_d',
                      'At': '_d',
                      'Fr': '_sv',
                      'Ra': '_sv'}


class EspressoStandards():
    # Espresso parameters to that will be tracked in parameterized model
    sorted_calc_parameters = ['xc', 'encut', 'nbands', 'ispin', 'kspacing',
                              'kgamma', 'ismear', 'sigma', 'ibrion', 'isif',
                              'nsw', 'nelm', 'ediff', 'prec', 'algo', 'lwave',
                              'ldau', 'ldautype']


class CommonCalc():
    """+U values"""
    U_trickers = ['O', 'F']  # Oxides and Flourides will have +U
    U_luj = {'Au': {'L': -1, 'U': 0.0, 'J': 0.0},
             'C':  {'L': -1, 'U': 0.0, 'J': 0.0},
             'Cu': {'L': -1, 'U': 0.0, 'J': 0.0},
             'H':  {'L': -1, 'U': 0.0, 'J': 0.0},
             'Ir': {'L': -1, 'U': 0.0, 'J': 0.0},
             'O':  {'L': -1, 'U': 0.0, 'J': 0.0},
             'F':  {'L': -1, 'U': 0.0, 'J': 0.0},
             'Co': {'L': 2, 'U': 3.32, 'J': 0.0},
             'Cr': {'L': 2, 'U': 3.7, 'J': 0.0},  # Meng U: 3.5
             'Fe': {'L': 2, 'U': 5.3, 'J': 0.0},  # 'U': 4.3
             'Mn': {'L': 2, 'U': 3.9, 'J': 0.0},  # 'U': 3.75
             'Mo': {'L': 2, 'U': 4.38, 'J': 0.0},
             'Nb': {'L': 2, 'U': 4.00, 'J': 0.0},
             'Ni': {'L': 2, 'U': 6.2, 'J': 0.0},  # 'U': 6.45
             'Sn': {'L': 2, 'U': 3.5, 'J': 0.0},
             'Ta': {'L': 2, 'U': 4.00, 'J': 0.0},
             'Ti': {'L': 2, 'U': 3.00, 'J': 0.0},
             'V':  {'L': 2, 'U': 3.25, 'J': 0.0},
             'W':  {'L': 2, 'U': 6.2, 'J': 0.0},  # 'U': 2.0
             'Zr': {'L': 2, 'U': 4.00, 'J': 0.0},
             'Ce': {'L': 3, 'U': 4.50, 'J': 0.0}}

    U_metals = list(U_luj.keys())
    for U in U_trickers:
        U_metals.remove(U)

    initial_magmoms = {'Ce': 5,
                       'Co': 5,
                       'Cr': 5,
                       'Fe': 5,
                       'Mn': 5,
                       'Mo': 5,
                       'Ni': 5,
                       'V': 5,
                       'W': 5}

    magnetic_trickers = list(initial_magmoms.keys())


class CrystalStandards():

    """Reference structures for formation energies, taken from 
    Materials Project.
    Most metals falls in the standard crystal structures:
    hcp: 194
    fcc: 225
    bcc: 229
    """

    standard_lattice_mp = {
        'Li': {'p_name': 'A_1_a_229',
               'spacegroup': 229,
               'wyckoffs': ['a'],
               'species': ['Li']},
        'Be': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Be']
               },
        'Na': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Na']
               },
        'Mg': {'p_name': 'A_3_ac_166',
               'spacegroup': 166,
               'wyckoffs': ['a', 'c'],
               'species': ['Mg', 'Mg'],
               'parameters': {'a': 3.2112466672290045,
                              'c/a': 7.139511029771984,
                              'zc1': 0.2222079999999999}},
        'Al': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Al']},
        'K':  {'p_name': 'A_20_cd_213',
               'spacegroup': 213,
               'wyckoffs': ['c', 'd'],
               'species': ['K', 'K'],
               'parameters': {'a': 11.435131,
                              'xc0': 0.062133,
                              'yd1': 0.202742}},
        'Rb': {'p_name': 'A_29_acg2_217',
               'spacegroup': 217,
               'wyckoffs': ['a', 'c', 'g', 'g'],
               'species': ['Rb', 'Rb', 'Rb', 'Rb'],
               'parameters': {'a': 17.338553,
                              'xc1': 0.817731,
                              'xg2': 0.639646,
                              'zg2': 0.042192,
                              'xg3': 0.09156399999999998,
                              'zg3': 0.2818120000000004}},
        'Ca': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Ca']},
        'Sr': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Sr']},
        'Sc': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Sc']},
        'V': {'p_name': 'A_2_c_194',
              'spacegroup': 194,
              'wyckoffs':  ['c'],
              'species': ['V']},
        'La': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['La']},
        'Ti': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Ti']},
        'Zr': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Zr']},
        'Hf': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Hf']},
        'V': {'p_name': 'A_1_a_229',
              'spacegroup': 229,
              'wyckoffs': ['a'],
              'species': ['V']},
        'Nb': {'p_name': 'A_1_a_166',
               'spacegroup': 166,
               'wyckoffs': ['a'],
               'species': ['Nb']},
        'Ta': {'p_name': 'A_1_a_229',
               'spacegroup': 229,
               'wyckoffs': ['a'],
               'species': ['Ta']},
        'Cr': {'p_name': 'A_1_a_229',
               'spacegroup': 229,
               'wyckoffs': ['a'],
               'species': ['Cr']},
        'Mo': {'p_name': 'A_1_a_229',
               'spacegroup': 229,
               'wyckoffs': ['a'],
               'species': ['Mo']},
        'W': {'p_name': 'A_1_a_229',
              'spacegroup': 229,
              'wyckoffs': ['a'],
              'species': ['W']},
        'Mn': {'p_name': 'A_29_acg2_217',  # magnetic
               'spacegroup': 217,
               'wyckoffs': ['a', 'c', 'g', 'g'],
               'species': ['Mn', 'Mn', 'Mn', 'Mn'],
               'parameters': {'a': 8.618498,
                              'xc1': 0.818787,
                              'xg2': 0.643796,
                              'zg2': 0.035468,
                              'xg3': 0.9109409999999998,
                              'zg3': 0.282544}},
        'Fe': {'p_name': 'A_1_a_229',
               'spacegroup': 229,
               'wyckoffs': ['a'],
               'species': ['Fe']},  # magnetic
        'Tc': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Tc']},
        'Re': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Re']},
        'Ru': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Ru']},
        'Os': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Os']},
        'Co': {'p_name': 'A_2_c_194',  # magnetic
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Co']},
        'Rh': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Rh']},
        'Ir': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Ir']},
        'Ni': {'p_name': 'A_1_a_225',  # magnetic
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Ni']},
        'Pd': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Pd']},
        'Pt': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Pt']},
        'Cu': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Cu']},
        'Ag': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Ag']},
        'Au': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Au']},
        'Zn': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Zn']},
        'Cd': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Cd']},
        'Hg': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Hg']},
        'Ga': {'p_name': 'A_4_f_64',
               'spacegroup': 64,
               'wyckoffs': ['f'],
               'species': ['Ga']},
        'In': {'p_name': 'A_3_ac_166',
               'spacegroup': 166,
               'wyckoffs': ['a', 'c'],
               'species': ['In', 'In'],
               'parameters': {'a': 3.3328619310176943,
                              'c/a': 7.623639840442314,
                              'zc1': 0.22166400000000003}},
        'Tl': {'p_name': 'A_2_c_194',
               'spacegroup': 194,
               'wyckoffs':  ['c'],
               'species': ['Tl']
               },
        'Si': {'p_name': 'A_2_a_227',
               'spacegroup': 227,
               'wyckoffs': ['a'],
               'species': ['Si']},
        'Ge': {'p_name': 'A_2_a_227',
               'spacegroup': 227,
               'wyckoffs': ['a'],
               'species': ['Ge']},
        'Sn': {'p_name': 'A_2_a_227',
               'spacegroup': 227,
               'wyckoffs': ['a'],
               'species': ['Sn']},
        'Pb': {'p_name': 'A_1_a_225',
               'spacegroup': 225,
               'wyckoffs': ['a'],
               'species': ['Pb']},
        'As': {'p_name': 'A_2_c_166',
               'spacegroup': 166,
               'wyckoffs': ['c'],
               'species': ['As']},
        'Sb': {'p_name': 'A_2_c_166',
               'spacegroup': 166,
               'wyckoffs': ['c'],
               'species': ['Sb']},
        'Bi': {'p_name': 'A_2_c_166',
               'spacegroup': 166,
               'wyckoffs': ['c'],
               'species': ['Bi']},
        'Se': {'p_name': 'A_64_e16_14', 'spacegroup': 14,
               'wyckoffs': ['e', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'e',
                            'e', 'e', 'e', 'e', 'e', 'e'],
               'species': ['Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se',
                           'Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se', 'Se'],
               'parameters': {'a': 15.739181,
                              'b/a': 0.9726696706772735,
                              'c/a': 0.6451328742335684,
                              'beta': 94.0428681461934,
                              'xe0': 0.986572,
                              'ye0': 0.593959,
                              'ze0': 0.23545999999999978,
                              'xe1': 0.984712,
                              'ye1': 0.214049,
                              'ze1': 0.12086899999999978,
                              'xe2': 0.913585,
                              'ye2': 0.675372,
                              'ze2': 0.828421,
                              'xe3': 0.906751,
                              'ye3': 0.009713,
                              'ze3': 0.847639,
                              'xe4': 0.814616,
                              'ye4': 0.142437,
                              'ze4': 0.4519159999999999,
                              'xe5': 0.806848,
                              'ye5': 0.690537,
                              'ze5': 0.978908,
                              'xe6': 0.769548,
                              'ye6': 0.047426,
                              'ze6': 0.27722899999999984,
                              'xe7': 0.76865,
                              'ye7': 0.215177,
                              'ze7': 0.888981,
                              'xe8': 0.761848,
                              'ye8': 0.508143,
                              'ze8': 0.2745009999999999,
                              'xe9': 0.695855,
                              'ye9': 0.589422,
                              'ze9': 0.43704999999999994,
                              'xe10': 0.6914370000000001,
                              'ye10': 0.736288,
                              'ze10': 0.36462799999999984,
                              'xe11': 0.646212,
                              'ye11': 0.524822,
                              'ze11': 0.842267,
                              'xe12': 0.644476,
                              'ye12': 0.185345,
                              'ze12': -0.000248000000000026,
                              'xe13': 0.52639,
                              'ye13': 0.213948,
                              'ze13': 0.8495199999999999,
                              'xe14': 0.525862,
                              'ye14': 0.645501,
                              'ze14': 0.10287099999999982,
                              'xe15': 0.524356,
                              'ye15': 0.04622,
                              'ze15': 0.24163899999999983}},
        'Te': {'p_name': 'A_3_a_152',
               'spacegroup': 152,
               'wyckoffs': ['a'],
               'species': ['Te'],
               'parameters': {'a': 4.5123742098481765,
                              'c/a': 1.3207900592536466,
                              'xa0': 0.26895}}
    }
