# Graph creation functions
import ase
from ase.neighborlist import primitive_neighbor_list

import numpy as np

import pandas as pd

from sne_ml_databases.property_tables import TABLES as atomic_data
from sne_ml_databases.property_tables import TABLES_NUMERIC as atomic_data_numeric

from .graph_property_encoding import get_plain_atom_features, _weight_and_expand

from collections import Iterable
from scipy.spatial import Voronoi

"""
def _get_atom_features_onehot(struc, per_atom_features=["e", "r"]):
    #atom_df = pd.DataFrame(np.zeros((len(struc), len(per_atom_features))),
    #                       columns=per_atom_features, index=struc.get_chemical_symbols())
    atom_data = np.zeros((len(struc), len(per_atom_features)))
    for idx, spec in enumerate(struc.get_chemical_symbols()):
        atom_data[idx] = np.array([atomic_data[p][spec] for p in per_atom_features])
    return atom_data
"""

def node_graph(struc, get_features="pos", cell=True):
    if callable(get_features):
        atom_features = get_features(struc)
    elif get_features == "pos":
        atom_features = struc.positions
    elif get_features == "symbols":
        atom_features = struc.get_chemical_symbols()
    else:
        raise NotImplementedError()
    if cell and struc.cell is not None:
        return atom_features, np.array([[],[],]), np.array([]), {"cell" : np.array(struc.cell).reshape(1, 3, 3)}
    else:
        return atom_features, np.array([[],[],]), np.array([]), None
    


def radialconnection_graph(struc : ase.Atoms,
                           r_cut : float = 5,
                           max_nearest : int = 5, # maximum number of nearest neighbors taken into account
                           get_features : callable = get_plain_atom_features,
                           override_pbc : bool = True,
                           connect_self : bool = True, # connect same idx across periodic pbcs!
                           # TODO: add actual self-loop?!
                           weight_distance : str = "basic", # or False or 1/r2 or ...
                           expand_distance : (str, bool) = False, # or "ohe" or "gauss"
                           expand_bins : (int) = 50,
                           expand_width : float = 0.2,
                           aggregate_distance : bool = False # or "min" or "mean" or "sum"
                           ):
    if isinstance(get_features, Iterable):
        atom_features = np.hstack([ffunc(struc) for ffunc in get_features])
    else:
        atom_features = get_features(struc)
    atom_adj_features = []

    sys_pbc = struc.pbc
    if override_pbc and struc.cell is not None:
        sys_pbc = [True,]*3
    
    start, end, dist = primitive_neighbor_list("ijd", sys_pbc, struc.cell,
                                               struc.positions, r_cut)

    if connect_self is False:
        scs = start != end
        start = start[scs]
        end = end[scs]
        dist = dist[scs]
        
    start_nn = []
    end_nn = []
    dist_nn = []

    # just select a maximum of neighbors, disregard self-connections if connect_self == False
    if max_nearest > 0:
        for start_idx in np.unique(start):
            #print("==="*10)
            end_idxs = end[start == start_idx]
            end_dists = dist[start == start_idx]
            #print(end_idxs, end_dists)
            end_idxs_s, end_dists_s = list(zip(*sorted(zip(end_idxs, end_dists), key=lambda t: t[1])))
            #print(end_idxs_s, end_dists_s)
            valid_ends = end_idxs_s[:max_nearest]
            valid_end_dists = end_dists_s[:max_nearest]
            start_nn.append(np.array([start_idx,]*len(valid_ends)))
            end_nn.append(valid_ends)
            dist_nn.append(valid_end_dists)
            
        start_nn = np.concatenate(start_nn)
        end_nn = np.concatenate(end_nn)
        dist_nn = np.concatenate(dist_nn)
    elif max_nearest == 0:
        start_nn = np.array([])
        end_nn = np.array([])
        dist_nn = np.array([])
    else:
        start_nn = start
        end_nn = end
        dist_nn = dist

    distances = _weight_and_expand(
        dist_nn, 0, r_cut,
        weight_method=weight_distance,
        expand_method=expand_distance,
        expand_bins = expand_bins,
        ohe_include_extremes = True)

    final_starts = start_nn
    final_ends = end_nn
    # do the aggregation using pandas, because...
    if aggregate_distance:
        connections = pd.Series(zip(final_starts, final_ends), name="c")
        distance_series = pd.Series(list(distances), name="d")
        merged = pd.concat([connections, distance_series], axis=1)
        if expand_distance is not False and aggregate_distance != "sum":
            raise NotImplementedError
        aggregate = merged.groupby("c").agg(aggregate_distance)
        agg_series = aggregate["d"]
        final_starts, final_ends = zip(*(agg_series.index))
        final_starts = np.array(final_starts)
        final_ends = np.array(final_ends)
        distances = np.vstack(agg_series)

    atom_adj_tensor = np.vstack((final_starts, final_ends))
    atom_adj_features = np.array(distances)

    return atom_features, atom_adj_tensor, atom_adj_features, None


def voronoi_graph(struc : ase.Atoms,
                  get_features : callable = get_plain_atom_features,
                  #override_pbc : bool = True,
                  #connect_self : bool = True, # maybe actually add self-loops!
                  weight_distance : str = "basic", # or False or 1/r2 or ...
                  expand_distance : (str, bool) = False, # or "ohe" or "gauss"
                  min_distance : float = 1,
                  max_distance : float = 10,
                  expand_dist_bins = 50,
                  expand_dist_width = 0.2,
                  use_area = True, # or False...
                  weight_area : str = "basic", # or False or 1/r2
                  expand_area : (str, bool) = False, # or ohe or gauss
                  min_area = 0.1,
                  max_area = 20,
                  expand_area_bins = 10,
                  expand_area_width = 0.2):
    """
    make a graph by getting connections from voronoi-tesselation
    """
    atom_features = get_features(struc)

    positions = struc.positions
    cell = struc.cell
    rep_vector = np.zeros((3*3*3*len(positions), 3))

    shifted = []
    rep_idx = 0
    for x_shift in (-1, 0, 1):
        for y_shift in (-1, 0, 1):
            for z_shift in (-1, 0, 1):
                shifted.append(
                    positions+x_shift*cell[0]+y_shift*cell[1]+z_shift*cell[2])
                rep_vector[rep_idx*len(positions):rep_idx*len(positions)+len(positions)] = \
                    np.array([x_shift, y_shift, z_shift])
                rep_idx += 1
    
    all_positions = np.concatenate(shifted)
    rep_idx_map = np.array(list(range(len(positions)))*(3*3*3))
    rep_original = np.all(rep_vector == np.array([0, 0, 0]), axis=1)

    original_idxs = np.argwhere(rep_original).flatten()

    voronoi = Voronoi(all_positions)

    conn_idxs = []
    for idx in original_idxs:
        conn_idxs.append(np.argwhere(np.any(voronoi.ridge_points == idx, axis=1)).flatten())
    conn_idxs = np.concatenate(conn_idxs)

    conns = voronoi.ridge_points[conn_idxs]
    conn_surfaces = [voronoi.ridge_vertices[idx] for idx in conn_idxs]
    conns_start = all_positions[conns[:, 0]]
    conns_end = all_positions[conns[:, 1]]
    conn_dists = np.linalg.norm(conns_end - conns_start, axis=1)

    # https://www.mathwords.com/a/area_convex_polygon.htm
    # https://stackoverflow.com/questions/451426/how-do-i-calculate-the-area-of-a-2d-polygon
    # https://web.archive.org/web/20100405070507/http://valis.cs.uiuc.edu/~sariel/research/CG/compgeom/msg00831.html
    # motivation: project everything on a vector, then calculate trapezoids
    # TODO: someone check this is right

    def poly_area(verts):
        # polygon not closed! (otherwise range(N-1))
        N = len(verts)
        area = 0.0
        for i in range(N):
            j = (i+1)% N
            #print("x, y", i,j)
            area += verts[i][0] * verts[j][1]
            area -= verts[i][1] * verts[j][0]
        return abs(0.5*area)

    conn_surfaces_xy = []
    conn_surface_areas = []

    for csurf_verts in conn_surfaces:
        csurf_points = voronoi.vertices[csurf_verts]
        base_ax = csurf_points[1]-csurf_points[0]
        ax2 = csurf_points[2]-csurf_points[0]
        normal = np.cross(base_ax, ax2)
        ax2_orth = np.cross(normal, base_ax)
        orthnorm_sys = np.vstack([v/np.linalg.norm(v, axis=0) for v in (base_ax, ax2_orth, normal)])
        csurf_xys = [np.zeros(3)] # first point at origin
        for point in csurf_points[1:]:
            vector_orig = point - csurf_points[0]
            vector_new = np.zeros(3)
            for c_idx in range(3):
                # project vector orig on all sys-vectors (unit-length already)
                vector_new[c_idx] = np.dot(vector_orig, orthnorm_sys[c_idx])
            csurf_xys.append(vector_new)
        #csurf_xys.append(np.zeros(3))
        csurf_xys = np.vstack(csurf_xys)
        conn_surfaces_xy.append(csurf_xys)
        if np.any(csurf_xys[:,2] > 1e-8):
            #print("Something's wrong with the Voronoi-tesselation")
            # TODO: fixthis by searching for better vectors!
            print("VORO_WARNING:", struc,) #"\n", csurf_xys)
            conn_surface_areas.append(poly_area(csurf_xys[:,:2]))
            #raise Exception
        else:
            conn_surface_areas.append(poly_area(csurf_xys[:,:2]))

    # in theory: with voronoi-ridges, we can `easily` get "node"-features too! (e.g. volume of the cell!)
    edge_dists = np.concatenate([conn_dists, conn_dists])
    edge_areas = np.concatenate([conn_surface_areas, conn_surface_areas])

    edges_old = np.vstack(
        [conns, conns[:, [1,0]]]).T
    remapper = np.vectorize(lambda s: rep_idx_map[s])
    edges = remapper(edges_old)

    distances = _weight_and_expand(edge_dists, min_distance, max_distance,
                                   weight_method = weight_distance,
                                   expand_method = expand_distance,
                                   expand_bins = expand_dist_bins, expand_width = expand_dist_width,
                                   ohe_include_extremes = True)
    edge_attrs = distances
    if use_area:
        areas = _weight_and_expand(edge_areas, min_area, max_area,
                                   weight_method = weight_area,
                                   expand_method = expand_area,
                                   expand_bins = expand_area_bins,
                                   expand_width = expand_area_width,
                                   ohe_include_extremes = True)
    
        edge_attrs = np.hstack([edge_attrs, areas])

    return atom_features, edges, edge_attrs, None
