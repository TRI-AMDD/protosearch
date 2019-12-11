import numpy as np
import pylab as plt
import string

from pprint import pprint

from protosearch import build_bulk

path = build_bulk.__path__[0]

wyckoff_data = path + '/Wyckoff.dat'
wyckoff_pairs = path + '/Symmetric_Wyckoff_Pairs.dat'


class BulkEnumerator:

    def __init__(
            stoichiometry='',
            spacegroup_start=1,
            spacegroup_end=230):

        self.stoichiometry = stoichiometry

        self.spacegroups = range(spacegroup_start, spacegroup_end + 1)


def get_wyckoff_combinations_for_spacegroup(spacegroup, stoichiometry,
                                            n_max_atoms=6):

    letters, multiplicity = get_wyckoff_letters_and_multiplicity(spacegroup)

    pair_names, M, free_letters = get_wyckoff_pair_symmetry_matrix(spacegroup)
    n_pairs = len(pair_names)
    #print(pair_names[:17])
    #print(M[:17][:, :17])
    #sys.exit()
    stoich_vector = stoichiometry.split('_')
    n_types = len(stoich_vector)
    n_atoms = [int(n) for n in stoich_vector]
    n_tot = sum(n_atoms)

    n_rep = n_max_atoms // n_tot
    n_atoms_rep = [n * n_rep for n in n_atoms]

    start_index = 0

    combinations = [['']]

    letter_combinations = [[]]

    pair_indices = []

    final_combinations = []

    n = 0

    natoms = []

    natoms_combinations = []

    while len(pair_combinations) > 0:
        prev_combinations = pair_combinations.copy()
        combinations = []
        prev_natoms_combinations = natoms_combinations.copy()
        natoms_combinations = []
        #prev_letter_combinations = prev_letter_combinations.copy()
        for j, prev_comb in enumerate(prev_combinations):
            next_letter = prev_comb[-1].split('_')[-1]
            if next_letter == '':
                next_pair_indices = range(len(pair_names))
                start_index = 0
            else:
                next_pair_indices = [i for i, p in enumerate(
                    pair_names) if p[0] == next_letter]
                start_index = next_pair_indices[0] - 1

            excl_indices = []
            for i in next_pair_indices:
                pair = pair_names[i]

                check_range = np.array(
                    [k for k in range(start_index, i) if not k in excl_indices])

                if len(check_range) == 0:
                    check_range = []
                    continue
                #print(pair, check_range)
                #print(M[i, check_range])
                # np.all(M[check_range, first_match] == 0):
                if np.all(M[i][check_range] == 0):
                    # print(pair)

                    if prev_comb == ['']:
                        new_comb = [pair]
                    else:
                        new_comb = prev_comb + [pair]

                    last_letter = new_comb[-1][-1]
                    comb_letters = [c[0] for c in new_comb] + [last_letter]

                    if not last_letter in free_letters and last_letter in comb_letters[:-1]:
                        #print(comb_letters, last_letter)
                        #start_index = i + 1
                        excl_indices += [i]
                        #prev_indices += [i]
                        # print(new_comb)
                        # Only add position once
                        continue

                    comb_natoms = [multiplicity[np.where(
                        letters == l)[0]] for l in comb_letters]

                    #cumsum_
                    if sum(comb_natoms) > sum(n_atoms_rep):
                        continue
                    

                    if sum(comb_natoms) % n_tot == 0:

                        rep = sum(comb_natoms) / n_tot
                        n_atoms_temp = [n * rep for n in n_atoms]

                        last_i = 0
                        match = True
                        atoms_wyckoff_cut = []
                        for na in n_atoms_temp:
                            cumsum_atoms = np.cumsum(comb_natoms[last_i:])
                            idx = np.where(cumsum_atoms==na)[0]
                            if len(idx) == 0:
                                match = False
                            else:
                                atoms_wyckoff_cut += [idx[0] + 1]
                                last_i = idx[0] + 1
                        #print(atoms_wyckoff_cut)
                        #if not idx[0] + 2 == len(comb_natoms) or not match:
                        #    continue

                        if not match:
                            continue
                        
                        print(new_comb)
                        unique = True
                        pair_test = new_comb.copy()
                        if len(new_comb) > 1:
                            pair_test += [new_comb[0]
                                          [0] + '_' + new_comb[1][-1]]

                        indices_1 = [pair_names.index(p) for p in pair_test]
                        for f_comb in [f for f in final_combinations if len(f) == len(new_comb)]:
                            print(new_comb, f_comb)
                            pair_test_2 = f_comb.copy()
                            if len(f_comb) > 1:
                                pair_test_2 += [f_comb[0]
                                                [0] + '_' + f_comb[1][-1]]

                            indices_0 = [pair_names.index(
                                p) for p in pair_test_2]

                            cut0 = 0
                            uniques = []
                            for cut in atoms_wyckoff_cut:
                                idx0 = indices_0[cut0: cut]
                                idx1 = indices_1[cut0: cut]
                                cut0 = cut.copy()
                                M_test = M[idx0, :][:, idx1]
                                if np.all(M_test.any(axis=0)) and np.all(M_test.any(axis=1)):
                                    print(idx0, idx1, M_test)
                                    print('no')
                                    uniques += [False]
                                else:
                                    uniques += [True]
                            unique += np.any(uniques)
                        #print(unique)
                        if unique:
                            final_combinations += [new_comb]
                            #print(new_comb, 'All ok')
                    combinations += [new_comb]
                    print(combinations)
            # print(final_combinations)
            # sys.exit()
        n += 1

    final_names = []
    for final_comb in final_combinations:
        last_letter = final_comb[-1][-1]
        comb_letters = [c[0] for c in final_comb] + [last_letter]
        comb_multi = [multiplicity[np.where(letters == l)[0]]
                      for l in comb_letters]

        n_rep = sum(comb_multi) / n_tot
        n_atoms_rep = [n * n_rep for n in n_atoms]

        name = ''
        atom_i = 0

        atoms_count = 0
        for i, l in enumerate(comb_letters):
            if atoms_count == n_atoms_rep[atom_i]:
                atom_i += 1
                atoms_count = 0
                name += '_'
            if atoms_count < n_atoms_rep[atom_i]:
                name += l
                atoms_count += comb_multi[i]
            elif atoms_count > n_atoms_rep[atom_i]:
                name = None

        if name:
            name = '_'.join([''.join(sorted(list(n)))
                             for n in name.split('_')])
            final_names += [name]

    final_names = sorted(list(set(final_names)))
    print(len(final_names))
    pprint(final_names)


def get_wyckoff_letters_and_multiplicity(spacegroup):
    letters = np.array([], dtype=str)
    multiplicity = np.array([])
    with open(wyckoff_data, 'r') as f:
        i_sg = np.inf
        i_w = np.inf
        for i, line in enumerate(f):
            if '1 {} '.format(spacegroup) in line \
               and not i_sg < np.inf:
                i_sg = i
            if i > i_sg:
                if len(line) > 15:
                    continue
                if len(line) == 1:
                    break
                multi, w, sym, sym_multi = line.split(' ')
                letters = np.insert(letters, 0, w)
                multiplicity = np.insert(multiplicity, 0, multi)

    return letters, multiplicity


def get_wyckoff_pair_symmetry_matrix(spacegroup):
    letters, multiplicity = get_wyckoff_letters_and_multiplicity(spacegroup)

    n_points = len(letters)
    pair_names = []
    for i in range(n_points):
        for j in range(n_points):
            pair_names.append('{}_{}'.format(letters[i], letters[j]))

    M = np.zeros([n_points**2, n_points**2])

    with open(wyckoff_pairs, 'r') as f:
        sg = 1
        for i, line in enumerate(f):
            if len(line) == 1:
                if sg < spacegroup:
                    sg += 1
                    continue
                else:
                    break
            if sg < spacegroup:
                continue
            w_1 = line[:3]
            if not w_1 in pair_names:
                continue
            k = pair_names.index(w_1)
            pairs0 = line.split('\t')[2: -1]
            for w_2 in pairs0:
                j = pair_names.index(w_2)
                M[k, j] = 1

    free_letters = []
    for l in letters:
        i = pair_names.index(l + '_' + l)
        if M[i, i] == 1:
            free_letters += [l]

    np.fill_diagonal(M, 1)

    return pair_names, M, free_letters


if __name__ == "__main__":
    #pair_names, M = get_wyckoff_pair_symmetry_matrix(3)
    get_wyckoff_combinations_for_spacegroup(10, '1_2', n_max_atoms=3)
