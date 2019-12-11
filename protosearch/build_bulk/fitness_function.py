import numpy as np
import scipy
from shapely.geometry import Polygon
from ase.data import covalent_radii as cradii
from catkit.gen.utils.coordinates import expand_cell

from protosearch.utils.data import metal_numbers, prefered_O_state,\
    favored_O_connections, electronegs


def get_area_neighbors(atoms, cell_cutoff=15, cutoff=None,
                       return_std_con=False):
    # Builds on top of CatKit connectivity matrix
    """Return the connectivity matrix from the Voronoi
    method weighted by the area of the Voronoi vertices.
    Multi-bonding occurs through periodic boundary conditions.

    Parameters
    ----------
    atoms : atoms object
        Atoms object with the periodic boundary conditions and
        unit cell information to use.
    cell_cutoff : float
        Cutoff for supercell repetition used for Voronoi
    cutoff : float
        Maximum atomic bond distance to consider for connectivity

    Returns
    -------
    connectivity : ndarray (n, n) of floats
       Weighted number of edges formed between atoms in a system.
    connectivity_int : ndarray (n, n) of ints
       Number of edges formed between atoms in a system.
    """
    index, coords, offsets = expand_cell(atoms,
                                         cutoff=cell_cutoff)
    L = int(len(offsets) / 2)

    origional_indices = np.arange(L * len(atoms), (L + 1) * len(atoms))

    areas = []
    area_indices = []
    connectivity = np.zeros((len(atoms), len(atoms)))
    connectivity_int = np.zeros((len(atoms), len(atoms)))
    try:
        voronoi = scipy.spatial.Voronoi(coords, qhull_options='QbB Qc Qs')
        points = voronoi.ridge_points
    except:
        print('Voronoi failed')
        if not return_std_con:
            return connectivity
        else:
            return connectivity, connectivity_int

    Voro_cell = voronoi.points.ptp(axis=0)
    for i, n in enumerate(origional_indices):
        ridge_indices = np.where(points == n)[0]
        p = points[ridge_indices]
        dist = np.linalg.norm(np.diff(coords[p], axis=1), axis=-1)[:, 0]
        edges = np.sort(index[p])

        if not edges.size:
            warnings.warn(
                ("scipy.spatial.Voronoi returned an atom which has "
                 "no neighbors. This may result in incorrect connectivity."))
            continue

        unique_edge = np.unique(edges, axis=0)

        for j, edge in enumerate(unique_edge):
            indices = np.where(np.all(edge == edges, axis=1))[0]
            if cutoff:
                indices = indices[np.where(dist[indices] < cutoff)[0]]
            d = dist[indices]
            count = len(d)
            u, v = edge
            area_indices += [sorted([u, v])]
            edge_areas = []
            for ii, ind in enumerate(ridge_indices[indices]):
                vertices = np.array([list(voronoi.vertices[v]) for v in
                                     voronoi.ridge_vertices[ind]])
                if len(vertices) < 3:
                    continue

                vertices *= Voro_cell

                area = get_weighted_area(vertices, d[ii])
                if area is None:
                    continue

                connectivity[u][v] += area
                connectivity[v][u] += area

            connectivity_int[u][v] += count
            connectivity_int[v][u] += count

    connectivity /= 2

    if not return_std_con:
        return connectivity
    else:
        connectivity_int /= 2
        return connectivity, connectivity_int.astype(int)


def get_voronoi_neighbors(atoms, cutoff=5.0, return_distances=False):
    """Return the connectivity matrix from the Voronoi
    method. Multi-bonding occurs through periodic boundary conditions.

    Parameters
    ----------
    atoms : atoms object
        Atoms object with the periodic boundary conditions and
        unit cell information to use.
    cutoff : float
        Radius of maximum atomic bond distance to consider.

    Returns
    -------
    connectivity : ndarray (n, n)
        Number of edges formed between atoms in a system.
    """
    index, coords, offsets = expand_cell(atoms, cutoff=cutoff)

    xm, ym, zm = np.max(coords, axis=0) - np.min(coords, axis=0)

    L = int(len(offsets) / 2)
    origional_indices = np.arange(L * len(atoms), (L + 1) * len(atoms))

    voronoi = scipy.spatial.Voronoi(coords, qhull_options='QbB Qc Qs')
    points = voronoi.ridge_points

    connectivity = np.zeros((len(atoms), len(atoms)))
    distances = []
    distance_indices = []
    for i, n in enumerate(origional_indices):
        ridge_indices = np.where(points == n)[0]
        p = points[ridge_indices]
        dist = np.linalg.norm(np.diff(coords[p], axis=1), axis=-1)[:, 0]
        edges = np.sort(index[p])

        if not edges.size:
            warnings.warn(
                ("scipy.spatial.Voronoi returned an atom which has "
                 "no neighbors. This may result in incorrect connectivity."))
            continue

        unique_edge = np.unique(edges, axis=0)

        for j, edge in enumerate(unique_edge):
            indices = np.where(np.all(edge == edges, axis=1))[0]
            d = dist[indices][np.where(dist[indices] < cutoff)[0]]
            count = len(d)
            if count == 0:
                continue

            u, v = edge

            distance_indices += [sorted([u, v])]
            distances += [sorted(d)]

            connectivity[u][v] += count
            connectivity[v][u] += count

    connectivity /= 2
    if not return_distances:
        return connectivity.astype(int)

    if len(distances) > 0:
        distance_indices, unique_idx_idx = \
            np.unique(distance_indices, axis=0, return_index=True)
        distance_indices = distance_indices.tolist()

        distances = [distances[i] for i in unique_idx_idx]

    pair_distances = {'indices': distance_indices,
                      'distances': distances}

    return connectivity.astype(int), pair_distances


def get_weighted_area(vertices, d):
    center = vertices[0, :]
    vertices -= center

    # Plane of polygon in 3d defined by unit vectors ux and uy
    ux = vertices[1, :].copy()
    ux /= np.linalg.norm(ux)
    vy = vertices[2, :].copy()

    # ensure uy is orthogonal to ux
    vyproj = np.dot(vy, ux) * ux
    uy = vy - vyproj
    uy /= np.linalg.norm(uy)

    # Project all vectors onto plane to get 2D coordinates
    Xs = np.dot(vertices, ux)
    Ys = np.dot(vertices, uy)

    poly_coords = [[Xs[i], Ys[i]] for i in range(len(Xs))]
    polygon = Polygon(poly_coords)

    try:
        area = polygon.area
    except:
        return None

    perimeter = polygon.length

    # Scale area with bond distance and perimeter
    weight = d * perimeter / 4
    area /= weight

    return area


def get_covalent_density(atoms):

    covalent_radii = np.array([cradii[n] for n in atoms.numbers])
    covalent_volume = np.sum(4/3 * np.pi * covalent_radii ** 3)
    cell_volume = atoms.get_volume()
    density = covalent_volume / cell_volume

    return density


def get_apf_fitness(atoms, apf_0=0.1):
    """ Volume dependent fitness function, expressed in terms of the 
    atomic packing factor V_atoms/V_cell assuming a parabolic dependency
    F ~ 1/V**2

    Parameters:
    apf_0:  APF value where fitness=0
    """

    apf = get_covalent_density(atoms)
    a0 = apf_0**2/(1 - apf_0**2)

    fitness = 1 + a0 * (1 - 1 / apf**2)

    return fitness


def get_fitness(atoms, use_density=True):
    """
    Structure fitness for Wyckoff coordinate optimization

    Only implemented for  oxides and metals.
    Fitness is based on the prefered connectivity of atoms,
    as discussed in D. Waroquiers et al. Chem. Mater. 2017, 29, 8346−8360

    Parameters:
    atoms: ASE atoms object
    use_density: bool
        whether to include structure density in the fitness function 
        (default is False)
    """

    symbols = list(atoms.get_chemical_symbols())
    numbers = atoms.get_atomic_numbers()

    if np.all([z in metal_numbers for z in numbers]):
        # Pure metallic
        fitness = get_apf_fitness(atoms, apf_0=0.5)
        return fitness

    elif 'O' in symbols:
        # Oxide
        if use_density:
            fitness = get_apf_fitness(atoms, apf_0=0.1)
        else:
            fitness = 1
        # if fitness < -2: # don't do voronoi analysis on low volume structures
        #    return np.max([fitness, -20])

        # get connectivity matrix
        con_matrix, con_int = get_area_neighbors(atoms,
                                                 return_std_con=True)

        con_matrix_norm = con_matrix / con_int
        #print(np.round(con_matrix, 2))
        metal_idx = [i for i, a in enumerate(
            atoms) if a.number in metal_numbers]
        metal_symbols = [symbols[i] for i in metal_idx]
        O_idx = [i for i, a in enumerate(atoms) if a.symbol == 'O']
        n_O = len(O_idx)
        n_M = len(metal_idx)

        oxi_states = get_oxidation_states(metal_symbols, n_O)

        # M-O connection should be ~ 6
        M_O_contribution = 0
        for i, m in enumerate(metal_symbols):
            oxi_state = str(oxi_states[m])
            f_O_c = favored_O_connections[m].get(oxi_state, [6])
            n_connections = sum(con_matrix[i, O_idx])
            closest_f_O_c = f_O_c[np.argmin(np.abs(f_O_c - n_connections))]
            M_O_contribution +=\
                np.abs((n_connections - closest_f_O_c) / closest_f_O_c)
        fitness -= M_O_contribution / n_M

        # M-M connection should be zero ? Omit for now
        #fitness -= np.sum(con_matrix[metal_idx, ][:, metal_idx]) / n_M

        # O-M connection should be ~ 3
        # O-O connectivity shoud not exceed 2 (we don't want 02)
        for i in O_idx:
            n_connections = sum(con_matrix[i, metal_idx])
            f_O = np.array(f_O_c) / n_O * n_M
            closest_f_O = f_O[np.argmin(np.abs(f_O - n_connections))]
            fitness -= np.abs(n_connections - closest_f_O) \
                / n_O / closest_f_O

            for j in np.where(con_matrix_norm[i, O_idx] > 1)[0]:
                fitness -= (con_matrix_norm[i, O_idx][j] - 1) / n_O

    return np.max([fitness, -20])


def get_oxidation_states(metal_symbols, n_O):
    """
    A_nB_mO_k with oxidation states O_A and O_B setting O_O = 2
    Solve for integers O_A and O_B
    n_A * O_A + n_B * O_B = n_O * 2
    """

    n_M = len(metal_symbols)
    avg_oxi_state = n_O / n_M * 2

    oxi_states_dict = {}
    metal_symbols, counts = np.unique(metal_symbols, return_counts=True)

    electroneg = [electronegs[m] for m in metal_symbols]
    indices = np.argsort(electroneg)[::-1]
    metal_symbols = metal_symbols[indices]

    #n_M_list = [list(metal_symbols).count(m) for m in metal_symbols]
    # print(n_M_list)
    pref_O_states = [prefered_O_state[m] for m in metal_symbols]

    if len(metal_symbols) == 2:
        # Make a guess for favorable oxidation states
        oxy_matches = []
        n_A, n_B = counts
        for o_A in range(1, n_O * 2 // n_A):
            o_B = (2 * n_O - n_A * o_A) / n_B
            if o_B % 1 == 0:
                oxy_matches += [[o_A, int(o_B)]]

        oxy_matches = np.array(oxy_matches)

        pref_O_M = np.repeat(np.expand_dims(pref_O_states, 0),
                             len(oxy_matches[:, 0]), 0)

        Oxy_fitness = np.sum(np.abs(oxy_matches - pref_O_M), axis=1)

        best_fit = np.argmin(Oxy_fitness)

        oxy_state_list = oxy_matches[best_fit]

    elif len(metal_symbols) == 1:  # only one type of metal
        oxy_state_list = [avg_oxi_state]

    oxi_states = {}
    for i, m in enumerate(metal_symbols):
        oxi_states.update({m: int(oxy_state_list[i])})

    return oxi_states


def get_connections(atoms, decimals=1):

    connectivity = get_voronoi_neighbors(atoms, cutoff=3)
    atoms_connections = {}

    for i in range(len(atoms)):
        for j in range(len(atoms)):
            symbols = '-'.join(sorted([atoms[i].symbol, atoms[j].symbol]))
            if not symbols in atoms_connections:
                atoms_connections[symbols] = [0 for n in range(len(atoms))]

            if not atoms[i].symbol == atoms[j].symbol:
                k = np.min([i, j])
            else:
                k = i

            if not connectivity[i][j]:
                continue
            atoms_connections[symbols][k] += \
                connectivity[i][j] / 2

    final_connections = {}
    for key, value in atoms_connections.items():
        value = [v for v in value if v > 0]
        if len(value) > 0:
            m = np.mean(value)
            std = np.std(value)
        else:
            m = 0
            std = 0
        final_connections[key + '_mean'] = np.round(m, decimals)
        final_connections[key + '_std'] = np.round(std, decimals)
    return final_connections
