from ase.calculators.general import Calculator
from ase.calculators.calculator import PropertyNotImplementedError


class DummyCalc(Calculator):
    name = 'DummyCalc'
    implemented_properties = ['energy']

    def __init__(self,
                 energy_zero=None,
                 track_output=False,
                 ):
        self.energy_zero = energy_zero
        self.atoms = None

    def todict(self):
        out_dict = {"key0": 0, "key1": 1}
        return(out_dict)

    def get_potential_energy(self, force_consistent=False):
        return self.energy_zero

    def check_state(self, atoms, tol=1e-15):
        """Check for system changes since last calculation."""
        from ase.calculators.calculator import all_changes, equal
        if self.atoms is None:
            system_changes = all_changes[:]
        else:
            system_changes = []
            if not equal(self.atoms.positions, atoms.positions, tol):
                system_changes.append('positions')
            if not equal(self.atoms.numbers, atoms.numbers):
                system_changes.append('numbers')
            if not equal(self.atoms.cell, atoms.cell, tol):
                system_changes.append('cell')
            if not equal(self.atoms.pbc, atoms.pbc):
                system_changes.append('pbc')
            if not equal(self.atoms.get_initial_magnetic_moments(), atoms.get_initial_magnetic_moments(), tol):
                system_changes.append('initial_magmoms')
            if not equal(self.atoms.get_initial_charges(), atoms.get_initial_charges(), tol):
                system_changes.append('initial_charges')

        return(system_changes)

    def get_property(self, name, atoms=None, allow_calculation=True):
        """Returns the value of a property"""
        if name not in DummyCalc.implemented_properties:
            raise PropertyNotImplementedError
        if atoms is None:
            atoms = self.atoms

        saved_property = {'energy': 'energy_zero'}
        if hasattr(self, saved_property[name]):
            result = getattr(self, saved_property[name])
        else:
            result = None

        return(result)
