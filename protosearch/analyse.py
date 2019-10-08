import numpy as np
import pylab as p
import ase


from protosearch.active_learning import ActiveLearningLoop
from protosearch.ml_modelling.fingerprint import clean_features


class MetaAnalysis(ActiveLearningLoop):

    def __init__(self,
                 *args, **kwargs,
                 ):
        self.energies = None

        super().__init__(*args, **kwargs)

    def plot_fingerprint_variation(self):
        test_ids = self.DB.get_initial_structure_ids()
        all_fingerprints = self.DB.load_dataframe('fingerprint', ids=test_ids)
        all_fingerprints = {'train': all_fingerprints.values, 'test': None}
        all_fingerprints, bad_ids = clean_features(all_fingerprints,
                                                   scale=True)
        all_fingerprints = all_fingerprints['train']
        max_values = np.max(all_fingerprints, axis=0)
        indices = np.argsort(max_values)
        p.plot(all_fingerprints[:, indices].T)

        p.xlabel('Feature id')
        p.ylabel('Standardized feature value')
        p.title('Fingerprints for {} structures'.format(len(all_fingerprints)))
        p.show()

    def plot_prediction(self, energies, uncertainties, calculated_indices=[]):
        p.figure()

        x = np.arange(len(energies))
        p.plot(x, energies, 'x', label='energies')
        p.plot(x, energies - uncertainties, 'k--', label='std deviation')
        p.plot(x, energies + uncertainties, 'k--')

        new_calculated_indices = \
            [i for i in np.where(uncertainties == 0)[0]
             if not i in calculated_indices]

        p.plot(x[calculated_indices], energies[calculated_indices],
               'yo', label='calculated')
        if len(new_calculated_indices) > 0:
            p.plot(x[new_calculated_indices],
                   energies[new_calculated_indices], 'ro',
                   label='new calculated')

        p.xlabel('Structure #')
        p.ylabel('Energy(eV)')
        p.legend()

        return p

    def collect_results(self):
        energies = np.append(self.energies, self.targets)
        uncertainties = np.append(self.uncertainties,
                                  np.zeros_like(self.targets))

        indices = np.argsort(energies)
        energies = energies[indices]
        uncertainties = uncertainties[indices]

        calculated_ids = np.where(uncertainties == 0)[0]

        return energies, uncertainties, calculated_ids

    def get_new_prediction(self):
        self.train_ids = self.DB.get_completed_structure_ids()
        self.test_ids = \
            self.DB.get_uncompleted_structure_ids(unsubmitted_only=False)
        self.get_ml_prediction()

        energies, uncertainties, calculated_ids = self.collect_results()

        p = self.plot_prediction(energies,
                                 uncertainties,
                                 calculated_ids)

        p.title('Energy predictions')

        p.show()

    def plot_acquisition(self, kappa=0.5, method='LCB'):
        #self.train_ids = self.train_ids[:100]
        # if self.energies is None:

        self.train_ids = self.DB.get_completed_structure_ids()[:30]
        # get_completed_structure_ids(completed=0)
        self.test_ids = self.DB.get_structure_ids()

        self.test_ids = [i for i in self.test_ids if i not in self.train_ids]
        self.get_ml_prediction()

        all_ids = self.test_ids + self.train_ids
        all_energies = np.array(list(self.energies) + list(self.targets))
        all_uncertainties = np.array(list(self.uncertainties) +
                                     list(np.zeros_like(self.targets)))

        idx = np.argsort(all_energies)
        energies = all_energies[idx]
        uncer = all_uncertainties[idx]
        ids = [all_ids[i] for i in idx]
        p.plot(range(len(idx)), energies, 'x', label='energies')
        p.plot(range(len(idx)), energies -
               uncer / 2, 'k--', label='confidence')
        p.plot(range(len(idx)), energies +
               uncer / 2, 'k--')
        marker = ['o', '*', '<', 'D', 's', '^']

        self.acquire_batch(method=method, kappa=kappa, batch_size=10)
        acquisition_values = np.array(
            list(self.acquisition_values) + [None] * len(self.targets))
        p.plot(range(len(idx)), acquisition_values[idx], 'r--')
        new_ids_idx = [ids.index(i) for i in self.batch_ids]
        p.plot(new_ids_idx, energies[new_ids_idx],
               marker[0], label='{}'.format(method + str(kappa)))
        p.legend()
        p.show()

    def plot_predictions(self):
        #predictions = self.DB.get_predictions()
        predictions = []
        calculated = []
        if self.energies is None:
            self.get_new_prediction()

        energies, uncertainties, calculated_ids = self.collect_results()
        prediction = {'energies': energies,
                      'vars': uncertainties,
                      'batch_no': predictions[-1]['batch_no'] + 1}
        predictions += [prediction]

        for prediction in predictions[-5:]:
            p.figure()
            idx = np.argsort(prediction['energies'])
            energies = prediction['energies'][idx]
            unc = prediction['vars'][idx]
            self.plot_prediction(energies, inc)

            new_ids_idx = [ids.index(i) for i in new_ids]

            p.plot(old_ids_idx, energies[old_ids_idx],
                   'yo', label='previous batches')
            p.plot(new_ids_idx,
                   energies[new_ids_idx], 'ro', label='latest batch')

            p.xlabel('Structure id')
            p.title('Predictions for batch {}'.format(prediction['batch_no']))

            p.ylabel('Energy(eV)')
            p.legend()

            p.savefig('prediction_batch_{}.png'.format(prediction['batch_no']))

        p.show()

    def plot_test_predictions(self, method='random'):
        predictions = []
        calculated = []
        i = 0
        ids_old = []
        for prediction in self.test_run(acquire=method, kappa=0.5):
            i += 1
            p.figure()
            idx = np.argsort(prediction['energies'])
            energies = prediction['energies'][idx]
            unc = prediction['vars'][idx]
            ids = [int(prediction['ids'][i]) for i in idx]

            p.plot(range(len(idx)), energies, 'x', label='predictions')
            p.plot(range(len(idx)), energies -
                   unc / 2, '-', color='lightslategray', label='uncertainty')
            p.plot(range(len(idx)), energies + unc /
                   2, '-', color='lightslategray')

            idx1 = np.where(unc == 0)[0]
            ids1 = [ids[i] for i in idx1]

            new_ids = [i for i in ids1 if i not in calculated]
            old_ids_idx = [ids.index(i) for i in calculated if i in ids]
            calculated += new_ids

            new_ids_idx = [ids.index(i) for i in new_ids]

            p.plot(idx1, energies[idx1],
                   'ro', label='calculated')

            for new_id in new_ids:
                j = ids.index(new_id)
                old_id = self.DB.ase_db.get(id=new_id).initial_id
                if len(ids_old) > 0:
                    index_old = ids_old.index(old_id)
                    p.plot([j], [energies_old[index_old]], 'rx')
                    p.plot([j, j], [energies_old[index_old],
                                    energies[j]], 'r-')

            p.text(400, -2.4, '{} calculations'.format(len(self.train_ids)),
                   fontsize=14)
            p.xlabel('Structure id', fontsize=18)
            p.title('Predictions for batch {}'.format(prediction['batch_no']),
                    fontsize=20)

            p.ylabel('Energy(eV)', fontsize=18)
            p.ylim(-5, 5)
            p.legend()

            p.savefig('prediction_{}_batch_{}.pdf'.format(
                method, prediction['batch_no']))

            p.title('')

            good_energies = {'Anatase': -2.420,
                             'Rutile': -2.412,
                             'Rutile1': -2.411,
                             'Rutile2': -2.410,                                                                 'Rutile3': -2.388,
                             'Rutile4': -2.375,
                             'porous tetragon1': -2.3218,
                             'porous tetragon2': -2.3218,
                             'porous tetragon3': -2.31605,
                             'layered1': -2.281,
                             'layered1': -2.280,

                             }

            for key, value in good_energies.items():
                found_energies = np.isclose(value, energies[idx1], atol=0.001)
                if np.any(found_energies):
                    idx2 = np.where(found_energies)[0][0]
                    print(idx2)
                    p.text(idx1[idx2]+0.1, energies[idx1][idx2]-0.01, key)

            p.savefig('zoom_prediction_{}_batch_{}.pdf'.
                      format(method,
                             prediction['batch_no']))
            unc_old = unc.copy()
            energies_old = energies.copy()
            ids_old = ids.copy()

        p.show()

    def test_model(self, method='UCB', kappa=None, plot=True):

        lowest_energy = list(self.DB.ase_db.select(relaxed=1, sort='Ef'))[0].Ef
        calculated = []
        i = 0

        from matplotlib import cm
        colors = cm.get_cmap('viridis')(np.linspace(0, 1, 20))
        for prediction in self.test_run(acquire=method, kappa=kappa):
            i += 1
            idx = np.argsort(prediction['energies'])
            energies = prediction['energies'][idx]
            unc = prediction['vars'][idx]
            ids = prediction['ids'][idx]

            idx1 = np.where(unc == 0)[0]
            print(ids[idx1])
            n_calc = len(idx1)

            if plot:
                p.figure()
                p.plot(range(len(idx)), energies, '-', color=colors[i],
                       label='batch no {}, {} calculations'.format(str(i), n_calc))
                p.plot(idx1, energies[idx1], 's', color=colors[i],
                       fillstyle='none')

            if plot:
                p.xlabel('Structure id')
                p.ylabel('Energy(eV)')
                titlestr = '{} aquisition'.format(method)
                if kappa:
                    titlestr += ' kappa={}'.format(kappa)
                p.title(titlestr)
                p.legend()
            if i > 8:
                break
        p.show()

        return i

    def test_prototype_change(self):
        count_all = 0
        count_changed = 0
        for row in self.DB.ase_db.select(relaxed=1):
            count_all += 1
            p_name = row.p_name
            row_i = self.DB.ase_db.get(id=row.initial_id)
            p_name_i = row_i.p_name
            if not p_name == p_name_i:
                count_changed += 1
                print(row_i.id, '-->', row.id,
                      '{} --> {}'.format(p_name, p_name_i))
        print('{} out of {} structures changed symmetry during optimization'
              .format(count_changed, count_all))

    def plot_performance(self, verbose=False):
        train_ids = self.DB.get_completed_structure_ids(completed=1)
        p_names = []
        energies = []
        for i in train_ids:
            entry = self.DB.ase_db.get(i)
            p_names += [entry.p_name]
            energies += [entry.energy]

        p_names, index = np.unique(p_names, return_index=True)
        train_ids = np.array(train_ids)[index]
        energies = np.array(energies)[index]

        remove_idx = []
        for i, e in enumerate(energies):
            if np.any(np.isclose(e, energies[i+1:], atol=0.01)):
                remove_idx += [i]

        train_ids = np.delete(train_ids, remove_idx)

        errors = []
        se = []
        errors_nonrelaxed = []
        se_nr = []
        energies = []
        mine = 100
        maxe = -100

        p.figure(figsize=(8, 8))
        print('Using {} training points'.format(len(train_ids) - 1))
        print('-------------------------------')
        for j in range(len(train_ids)):
            self.test_ids = [train_ids[j]]
            self.train_ids = np.delete(train_ids, j)
            initial_id = self.DB.ase_db.get(
                id=int(self.test_ids[0])).initial_id
            self.test_ids += [initial_id]
            self.test_ids = sorted(self.test_ids)
            print(self.test_ids)
            row = self.DB.ase_db.get(id=int(self.test_ids[-1]))
            row_initial = self.DB.ase_db.get(id=initial_id)
            energy = row.Ef

            self.train_ml()

            errors += [abs(self.energies[1] - energy)]
            se += [(self.energies[1] - energy)**2]
            errors_nonrelaxed += [abs(self.energies[0] - energy)]
            se_nr += [(self.energies[0] - energy)**2]

            if np.any(self.energies < mine):
                mine = np.min(self.energies)
            if np.any(self.energies > maxe):
                maxe = np.max(self.energies)

            p.plot([energy, energy], self.energies, linestyle='--',
                   color='gray', zorder=0)
            p.plot([energy], [self.energies[0]], 'ro')
            p.plot([energy], [self.energies[-1]], 'bo')

            if verbose:
                print('--------------------------')
                print('ids: ', self.test_ids)
                print('Ef: ', energy)
                print('Prototype: ', row.p_name)
                print('fmax: ', row.fmax)
                print('error relaxed: ', self.energies[1] - energy)
                print('error nonrelaxed: ', self.energies[0] - energy)

        print('-------------------------------')
        print('MAE relaxed Voronoi: ', np.mean(errors))
        print('MAE non-relaxed Voronoi: ', np.mean(errors_nonrelaxed))
        print('MSE relaxed Voronoi: ', np.mean(se))
        print('MSE non-relaxed Voronoi: ', np.mean(se_nr))

        p.plot([100], [100], 'ro',
               label='pre-relaxation voronoi. MAE: {0:.3f}'.\
               format(np.mean(errors_nonrelaxed)))
        p.plot([100], [100], 'bo',
               label='post-relaxed voronoi. MAE: {0:.3f}'.\
               format(np.mean(errors)))

        p.plot([mine-10, maxe+10], [mine-10, maxe+10], 'k-')
        p.xlim(mine-0.1, maxe+0.1)
        p.ylim(mine-0.1, maxe+0.1)

        p.title('ML performance for n_train={}'.format(len(train_ids) - 1),
                fontsize=24)
        p.xlabel('DFT Energy/atom (eV)', fontsize=18)
        p.ylabel('ML Energy/atom (eV)', fontsize=18)

        p.legend()

        p.savefig('performance.pdf')
        p.show()


if __name__ == "__main__":
    MA = MetaAnalysis(chemical_formulas=['Cu2O'], max_atoms=10, batch_size=10)
    MA.plot_performance()
    MA.plot_predictions()
    MA.plot_test_model()
    MA.plot_acquisition()
    MA.test_prototype_change()
