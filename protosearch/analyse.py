import numpy as np
import pylab as p
import ase

from protosearch.active_learning import ActiveLearningLoop

class MetaAnalysis(ActiveLearningLoop):

    def __init__(self,
                 chemical_formulas,
                 max_atoms):
        self.energies = None
        
        super().__init__(chemical_formulas=chemical_formulas,
                         max_atoms=max_atoms)


    def plot_fingerprint_variation(self):
        test_ids = self.DB.get_completed_structure_ids(completed=0)
        test_features, test_target = self.DB.get_fingerprints(ids=test_ids)
        for f in test_features:
            p.plot(f)
        p.show()

    def get_new_prediction(self):
        self.train_ids = self.DB.get_completed_structure_ids(completed=1)
        self.test_ids = self.DB.get_completed_structure_ids(completed=0)
        self.train_ml()

    def plot_acquisition(self):
        if self.energies is None:
            self.get_new_prediction()

        all_ids = self.test_ids + self.train_ids
        all_energies = np.array(list(self.energies) + list(self.train_target))
        all_uncertainties = np.array(list(self.uncertainties) +
                                      list(np.zeros_like(self.train_target)))

        idx = np.argsort(all_energies)
        energies = all_energies[idx]
        vars = all_uncertainties[idx]
        ids = [all_ids[i] for i in idx]
        p.plot(range(len(idx)), energies, 'x', label='energies')
        p.plot(range(len(idx)), energies -
               vars / 2, 'k--', label='confidence')
        p.plot(range(len(idx)), energies +
               vars / 2, 'k--')
        marker = ['o', '*', '<', 'D', 's', '^']
        for kappa in [0, 1, 2, 3, 4, 5]:
            self.acquire_batch(kappa=kappa, batch_size=10)
            new_ids_idx = [ids.index(i) for i in self.batch_ids]
            p.plot(new_ids_idx, energies[new_ids_idx], marker[kappa], label='{}'.format(kappa))
        p.legend()
        p.show()


    def plot_predictions(self):
        predictions = self.DB.get_predictions()
        calculated = []
        if self.energies is None: 
            self.get_new_prediction()
        all_ids = self.test_ids + self.train_ids
        all_energies = np.array(list(self.energies) + list(self.train_target))
        all_uncertainties = np.array(list(self.uncertainties) +
                                      list(np.zeros_like(self.train_target)))
        prediction = {'energies': all_energies,
                      'vars': all_uncertainties,
                      'ids': all_ids,
                      'batch_no': 'next'}
        predictions += [prediction]
        
        for prediction in predictions:
            p.figure()
            idx = np.argsort(prediction['energies'])
            energies = prediction['energies'][idx]
            vars = prediction['vars'][idx]
            ids = [prediction['ids'][i] for i in idx]
            p.plot(range(len(idx)), energies, 'x', label='energies')
            p.plot(range(len(idx)), energies - vars / 2, 'k--', label='confidence')
            p.plot(range(len(idx)), energies + vars / 2, 'k--')
            
            idx1 = np.where(vars==0)[0]
            ids1 = [ids[i] for i in idx1]
            new_ids = [i for i in ids1 if not i in calculated]
            old_ids_idx = [ids.index(i) for i in calculated if i in ids]
            calculated += new_ids
            
            new_ids_idx = [ids.index(i) for i in new_ids]

            if prediction['batch_no'] == 'next':
                self.acquire_batch(kappa=1.5, batch_size=10)
                new_ids_idx = [ids.index(i) for i in self.batch_ids]
                p.plot(new_ids_idx, energies[new_ids_idx], 'yo', label='previous batches')
                p.plot(new_ids_idx, energies[new_ids_idx], 'o', label='next batch')
                
            else:
                p.plot(old_ids_idx, energies[old_ids_idx], 'yo', label='previous batches')
                p.plot(new_ids_idx, energies[new_ids_idx], 'ro', label='latest batch')
                

            p.xlabel('Structure id')
            p.title('Predictions for batch {}'.format(prediction['batch_no']))
            
            p.ylabel('Energy(eV)')
            p.legend()

            p.savefig('prediction_batch_{}.png'.format(prediction['batch_no']))

        p.show()

    def test_prototype_change(self):
        print('id', 'initial prototype', 'final prototype')
        count_all = 0
        count_changed = 0
        for row in self.DB.ase_db.select(relaxed=1):
            count_all += 1
            p_name = row.p_name
            row_i = self.DB.ase_db.get(id=row.initial_id)
            p_name_i = row_i.p_name
            if not p_name==p_name_i:
                count_changed += 1
                print(row_i.id, '-->', row.id, '{} --> {}'.format(p_name, p_name_i))
        print('{} out of {} structures changed symmetry during optimization'
              .format(count_changed, count_all))

if __name__ == "__main__":
    MA = MetaAnalysis(chemical_formulas=['Cu2O'], max_atoms=10)
    MA.plot_predictions()
    MA.plot_acquisition()
    MA.test_prototype_change()

        
