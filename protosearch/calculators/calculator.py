
def get_calculator(name):
    if name == 'vasp':
        from .vasp import VaspModel as Calculator

    return Calculator

    
