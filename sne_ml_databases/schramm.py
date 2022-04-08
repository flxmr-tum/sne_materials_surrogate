import os
import re
import json

import numpy as np
import numpy.random
import random

import ase.io

from joblib import Parallel, delayed

from .util import dicts2df

UNIQUE_ID = "id"
MAIN_STRUCS = "asestruc"
SIMPLE_STRUCS = "basestruc"
GEN_PARAMS = "parameters"
PROPDIRS = ["gap"]
EXTRA_INFOS = ["specs", "metal", "halide", "beta", "delta", "scaling_fax", "scaling_feq"]


def _parse_paramsfile(f):
    ps = {}
    for l in open(f):
        k, v = l.strip().split(": ")
        ps[k] = v
    return ps

def read_id(dbdir, idno):
    d = {UNIQUE_ID : idno}
    d[MAIN_STRUCS] = ase.io.read(f"{dbdir}/{MAIN_STRUCS}/{idno}.cif")
    simple_path = f"{dbdir}/{SIMPLE_STRUCS}/{idno}.cif"
    params_file = f"{dbdir}/{GEN_PARAMS}/{idno}.json"
    if os.path.exists(simple_path):
        d[SIMPLE_STRUCS] = ase.io.read(simple_path)
        d["specs"] = "".join(sorted(set(d[SIMPLE_STRUCS].get_chemical_symbols())))
    else:
        d[SIMPLE_STRUCS] = None
        d["specs"] = ""
    # TODO: copy cells from simplified strucs, if applicable
    # if not any(d[MAIN_STRUCS].cell) and d[SIMPLE_STRUCS]:
    #     print(f"idno: {idno}, setting cell from simple struc")
    #     d[MAIN_STRUCS].cell = d[SIMPLE_STRUCS].cell
    #     d[MAIN_STRUCS].pbc = d[SIMPLE_STRUCS].pbc
    for p in PROPDIRS:
        p_path = f"{dbdir}/{p}/{p}_{idno}.dat"
        if os.path.exists(p_path):
            d[p] = float(open(p_path, 'r').read())
        else:
            d[p] = None
    if os.path.exists(params_file):
        avg_dpbi = DEF_DUMMY_OPTS["avg_dpbi"]
        ps = _parse_paramsfile(params_file)
        d["metal"] = ps["ions"].split()[0]
        d["halide"] = ps["ions"].split()[2]
        d["beta"] = float(ps["beta"])
        d["delta"] = float(ps["delt"])
        d["scaling_fax"] = float(ps["d_ax"])/avg_dpbi[d["metal"]][d["halide"]]
        d["scaling_feq"] = float(ps["d_eq"])/avg_dpbi[d["metal"]][d["halide"]]
    else:
        d["beta"] = None
        d["delta"] = None
        d["scaling_fax"] = None
        d["scaling_feq"] = None
        d["metal"] = None
        d["halide"] = None
    return d
        

def load(dbdir, parallel=4):
    base_idnos = [
        f[:-4] for f in os.listdir(f"{dbdir}/{MAIN_STRUCS}") if f.endswith(".cif")]
    db_dicts = Parallel(n_jobs=parallel)(delayed(read_id)(dbdir, idno) for idno in base_idnos)
    return db_dicts

import data_tools as dt
import itertools
from math import cos, sin, tan, atan, sqrt, pi

def _product_dict(**kwargs):
    #https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

# _get_coord and _create_struc_point are totally f****d up...
# Taken from Lukas Schramms code, probably originating from the group of C. Kathan

def _get_coord(dpbi1, dpbi2, csura, beta, delta):
    # dpbi1 = 3.18 # base du calcul on fixe la longueur de liaison equatorial
    # dpbi2 = 3.22 # base du calcul on fixe la longueur de liaison axial
    # csura = 3.5 #relation hight to length --> UC is 3.5 as high as long

    # in-plane
    # beta_degree = 10.0
    beta_degree = beta
    beta_angle = beta_degree * pi / 180
    u = tan(beta_angle) / 4.0

    # out-of-plane
    # delta_degree = 12.0
    delta_degree = delta
    delta_angle = delta_degree * pi / 180.0
    epsilon = tan(delta_angle) / csura / 4.0
    gamma_angle = atan(2 * sqrt(2) * epsilon * csura)
    gamma_degree = gamma_angle * 180.0 / pi
    angle_pb_i_pb = 180.0 - 2.0 * gamma_degree

    # in-plane lattice parameter
    # print u
    a_u_delta = dpbi1 / sqrt(
        (1.0 / 4.0 - u) ** 2 + (1.0 / 4.0 + u) ** 2 + csura ** 2 * epsilon ** 2
    )
    aBohr = a_u_delta / 0.529177249
    # aBohr = a_u_delta/0.529177

    # out-of_plane lattice parameter
    c_u_delta = csura * a_u_delta
    cBohr = c_u_delta / 0.529177249
    # cBohr = c_u_delta/0.529177

    # cell volume
    cell_volume = a_u_delta * a_u_delta * c_u_delta

    i7x = cos(beta_angle) * sin(delta_angle) * dpbi2 / a_u_delta
    i7y = sin(beta_angle) * sin(delta_angle) * dpbi2 / a_u_delta
    i7z = cos(delta_angle) * dpbi2 / c_u_delta
    i8x = -(cos(beta_angle)) * sin(delta_angle) * dpbi2 / a_u_delta
    i8y = -(sin(beta_angle)) * sin(delta_angle) * dpbi2 / a_u_delta
    i8z = -cos(delta_angle) * dpbi2 / c_u_delta
    i9x = 0.5 - (cos(-beta_angle)) * sin(delta_angle) * dpbi2 / a_u_delta
    i9y = 0.5 - (sin(-beta_angle)) * sin(delta_angle) * dpbi2 / a_u_delta
    i9z = cos(delta_angle) * dpbi2 / c_u_delta
    i10x = 0.5 + (cos(-beta_angle)) * sin(delta_angle) * dpbi2 / a_u_delta
    i10y = 0.5 + (sin(-beta_angle)) * sin(delta_angle) * dpbi2 / a_u_delta
    i10z = -cos(delta_angle) * dpbi2 / c_u_delta

    xred = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [1.0 / 4.0 - u, 1.0 / 4.0 + u, -epsilon],
            [1.0 / 4.0 + u, 3.0 / 4.0 + u, -epsilon],
            [3.0 / 4.0 + u, 3.0 / 4.0 - u, epsilon],
            [3.0 / 4.0 - u, 1.0 / 4.0 - u, epsilon],
            [i7x, i7y, i7z],
            [i8x, i8y, i8z],
            [i9x, i9y, i9z],
            [i10x, i10y, i10z],
            [0.5, 0.0, dpbi1 / c_u_delta],
            [0.5, 0.0, 1.0 - dpbi1 / c_u_delta],
            [0.0, 0.5, dpbi1 / c_u_delta],
            [0.0, 0.5, 1.0 - dpbi1 / c_u_delta],
        ]
    )

    #  print "\n-------------- Beta-Delta-----------------"
    #  print "Beta = ", beta_degree," degrees; delta = ",delta_degree," degrees"
    #
    #  print "\n-------------- Lattice paramters (Angstrom)  ------------------"
    #  print "a = %.15f    b = %.15f    c = %.15f    alpha = beta = gamma = 90.0"%(a_u_delta, a_u_delta, c_u_delta)
    #
    #  print "\n-------------- Lattice paramters (Bohr)  ------------------"
    #  print "a = %.15f    b = %.15f    c = %.15f    alpha = beta = gamma = 90.0"%(aBohr, aBohr, cBohr)
    #
    #  print "\n-------------- Atomic coordinates -----------------"
    symbols = [
        "Pb",
        "Pb",
        "I ",
        "I ",
        "I ",
        "I ",
        "I ",
        "I ",
        "I ",
        "I ",
        "Cs",
        "Cs",
        "Cs",
        "Cs",
    ]
    data = list()
    for i in range(0, len(xred)):
        data.append([symbols[i], xred[i][0], xred[i][1], xred[i][2]])
    #  for i in xrange(0,len(xred)):
    #    print("atom %2d %3s :   %3.15f    %3.15f    %3.15f"%(i+1, symbols[i], xred[i][0], xred[i][1], xred[i][2]))

    return a_u_delta, a_u_delta, c_u_delta, data

def _create_struc_point(dpbi1=None, dpbi2=None, beta=None, delta=None,
                        csura=None, ions=None, backgr_charge=None):
    elementlist = ["Sn", "Pb", "Cl", "Br", "I", "Cs"]
    elementcount = [0, 0, 0, 0, 0, 0]
    elementnumber = [50, 82, 17, 35, 53, 55]
    elementid = [0, 0, 0, 0, 0, 0]  # id used in siesta

    for el in ions:
        for i in range(6):
            if el == elementlist[i]:
                elementcount[i] = elementcount[i] + 1
    if backgr_charge == True:
        elementcount[5] = 0

    speciescount = 0
    for el in elementcount:
        if el != 0:
            speciescount += 1

    atoms_count = 0
    if backgr_charge:
        atoms_count = 10
    else:
        atoms_count = 14

    acell, bcell, ccell, data = _get_coord(
        dpbi1, dpbi2, csura, beta, delta
    )

    # Build ase.atoms with this structure
    name = ""
    if backgr_charge:
        ionsbkg = ions[:-4]
    else:
        ionsbkg = ions

    for el in ionsbkg:
        name = name + el
    uc = [[acell, 0.0, 0.0], [0.0, bcell, 0.0], [0.0, 0.0, ccell]]
    asestruc = ase.Atoms(name, cell=uc)
    site_corr = 0
    if backgr_charge:
        site_corr = 4
    
    for i in range(0, len(data) - site_corr):
        asestruc.positions[i] = (
            data[i][1] * acell,
            data[i][2] * bcell,
            data[i][3] * ccell,
        )

    return asestruc


DUMMY_PROPS = ["beta", "delta", "scaling_fax",
               "scaling_feq", "metal", "halide", "combined_species"]
GEN_PROPS_INCAT = ["beta", "delta", "scaling_fax", "scaling_feq",
                   "metal", "halide"]
GEN_PROPS = ["beta", "delta", "scaling_fax", "scaling_feq"]

def create_dummy_data_grid(parallel=4,
                           beta=list(range(0,21,1)),
                           delta=list(range(0,16,1)),
                           scaling_ax=None,
                           scaling_eq=None,
                           avg_dpbi=None, metals=None, halides=None,
                           backgr_charge=None
                           ) -> dt.AtomisticMLContainer:
    if metals is None:
        metals = ["Pb", "Sn"]
    if halides is None:
        halides = ["Cl", "Br", "I"]
    if scaling_ax is None:
        scaling_ax = [0.98, 0.985, 0.99, 0.995, 1.0, 1.005, 1.01, 1.015, 1.02]
    if scaling_eq is None:
        scaling_eq = [0.98, 0.985, 0.99, 0.995, 1.0, 1.005, 1.01, 1.015, 1.02]
    if avg_dpbi is None:
        avg_dpbi = {"Pb" : {"Cl": 2.87, "Br": 2.99, "I": 3.20},
                    "Sn" : {"Cl": 2.83, "Br": 2.94, "I": 3.13}}
    dummy_configs = _product_dict(
        **{"beta" : beta,
           "delta": delta,
           # adapted fromt the stupid old code
           "scaling_feq": scaling_eq,
           "scaling_fax": scaling_ax,
           "metal" : metals.copy(),
           "halide" : halides.copy()})
    tot_idx = 0
    datalist = []
    # TODO: parallelize...
    for idx, config in enumerate(dummy_configs):
        ddict = {UNIQUE_ID : tot_idx, **config}
        ddict["scaled_feq"] = ddict["scaling_feq"] * avg_dpbi[ddict["metal"]][ddict["halide"]]
        ddict["scaled_fax"] = ddict["scaling_fax"] * avg_dpbi[ddict["metal"]][ddict["halide"]]
        ddict["combined_species"] = ""
        ddict["struc"] = _create_struc_point(
            dpbi1=ddict["scaled_feq"], dpbi2=ddict["scaled_fax"], beta=ddict["beta"], delta=ddict["delta"],
            csura=3.5, ions=[ddict["metal"],]*2+[ddict["halide"],]*8+["Cs",]*4, backgr_charge=backgr_charge)
        datalist.append(ddict)
        tot_idx += 1
    property_df = dicts2df(datalist,
                           uid=UNIQUE_ID,
                           columns=DUMMY_PROPS)
    structures = [s["struc"] for s in datalist]
    return dt.AtomisticMLContainer(property_df, structures)


DEF_DUMMY_OPTS = {
    "beta" : (0,20),
    "delta" : (0,15),
    "scaling_fax" : (0.98, 1.02),
    "scaling_feq" : (0.98, 1.02),
    'metal' : ["Pb", "Sn"],
    'halide' : ["Cl", "Br", "I"],
    'avg_dpbi' : {"Pb" : {"Cl": 2.87, "Br": 2.99, "I": 3.20},
                  "Sn" : {"Cl": 2.83, "Br": 2.94, "I": 3.13}}
}

def dummy_data_sampling(
        idx,
        beta=None, delta=None,
        scaling_fax=None, scaling_feq=None,
        avg_dpbi=None, metal=None, halide=None,
        backgr_charge=None):
    config = {
        UNIQUE_ID : idx,
        "beta" : random.uniform(*beta) if isinstance(beta, tuple) else beta,
        "delta" : random.uniform(*delta) if isinstance(delta, tuple) else delta,
        "scaling_feq" : random.uniform(*scaling_feq) if isinstance(scaling_feq, tuple) else scaling_feq,
        "scaling_fax" : random.uniform(*scaling_fax) if isinstance(scaling_fax, tuple) else scaling_fax,
        "metal" : random.choice(metal) if not isinstance(metal, str) else metal,
        "halide" : random.choice(halide) if not isinstance(metal, str) else metal,
    }
    config["scaled_feq"] = config["scaling_feq"] * avg_dpbi[config["metal"]][config["halide"]]
    config["scaled_fax"] = config["scaling_fax"] * avg_dpbi[config["metal"]][config["halide"]]
    config["combined_species"] = ""
    config["struc"] = _create_struc_point(
        dpbi1=config["scaled_feq"], dpbi2=config["scaled_fax"], beta=config["beta"], delta=config["delta"],
        csura=3.5, ions=[config["metal"],]*2+[config["halide"],]*8+["Cs",]*4, backgr_charge=backgr_charge)
    return config
    

def create_dummy_data_sampled(
        n_samples,
        parallel=4,
        beta=None,
        delta=None,
        scaling_fax=None,
        scaling_feq=None,
        metal=None, halide=None,
        avg_dpbi=None,
        backgr_charge=None
) -> dt.AtomisticMLContainer:
    datalist = Parallel(n_jobs=parallel)(delayed(dummy_data_sampling)(idx, beta=beta, delta=delta,
                                                                      scaling_fax=scaling_fax, scaling_feq=scaling_feq,
                                                                      avg_dpbi=avg_dpbi, metal=metal, halide=halide,
                                                                      backgr_charge=backgr_charge)
                for idx in range(n_samples))
    property_df = dicts2df(datalist,
                           uid=UNIQUE_ID,
                           columns=DUMMY_PROPS)
    structures = [s["struc"] for s in datalist]
    return dt.AtomisticMLContainer(property_df, structures)
