from joblib import Memory, Parallel, delayed
import os

import numpy as np
import pandas as pd
import json

import ase
import ase.io
from ase.data import atomic_numbers
from ase.formula import Formula

UNIQUE_LABEL = "id"
CORE_PROPERTIES = [
    "ind_gap", "opt_gap",
    "energy", "formation_energy_pa",
    "iterations",
    ]
REALSTRUC = ["struc"]

# in eV/atom, from the paper SI
mus = {
    "Sn" : -1746.248,
    "Ge" : -1932.695,
    "Rb" : -665.270,
    "Cs" : -557.126,
    "K" : -773.077,
    "Na" : -1159.054,
    "I" : -2694.854,
    "Br" : -366.784,
    "Cl" : -445.682
    }

def transform_raw_dict(identifier, parsed_dict, relaxed):
    info_dict = {}

    info_dict["ind_gap"] = parsed_dict["gaps"]["indirect_gap"]
    info_dict["opt_gap"] = parsed_dict["gaps"]["direct_gap"]
    info_dict["energy"] = parsed_dict["energy"]

    if relaxed:
        slist = parsed_dict["rstruc"]
    else:
        slist = parsed_dict["ustruc"]

    info_dict["iterations"] = parsed_dict["iterations"]

    info_dict["struc"] = ase.Atoms(
        symbols=slist[2],
        scaled_positions=np.array(slist[1]),
        cell=np.array(slist[0]))

    fcount = Formula(info_dict["struc"].get_chemical_formula()).count()
    musum = sum([abs(mus.get(k, ""))*count for k, count in fcount.items()])
    
    formation_energy = abs(info_dict["energy"]) - abs(musum)
    info_dict["formation_energy_pa"] = formation_energy/len(info_dict["struc"])
    info_dict["id"] = identifier
    
    return info_dict

def load(db_path, parallel=4, relaxed=True, cull_convergence=200):
    db_dict = {}
    with open(db_path, 'r') as store:
        db_dict = json.load(store)
    info_dicts = Parallel(n_jobs=parallel)(delayed(transform_raw_dict)(t[0],t[1],relaxed) for t in db_dict.items())
    info_dicts = list(filter(lambda x: x["iterations"] < cull_convergence, info_dicts))
    return info_dicts

# save to json
def get_order_dict(data: dict, prefix="") -> dict:
    ids = sorted(data.keys())
    enum = [f"{prefix}{f}" for f in range(len(ids))]
    order_dict = dict(zip(ids, enum))
    return order_dict

def apply_order_dict(data : dict, order_dict : dict, cull_convergence=200) -> dict:
    new_data = {}
    for k in order_dict.keys():
        if data[k]["iterations"] < cull_convergence:
            new_data[order_dict[k]] = data[k]
    return new_data

# for reference: convert to prettified representations
# datafiles = {
#     "half_cells" : '2018-Q1-half-cells.json',
#     "lead_set" : '2018-Q3-leadSet.json',
#     "db_1" : '2018-Q2-randomset.json',
#     "benchmarks" : '2018-Q2-benchmarks.json',
#     "db_2" : '2018-Q3-augmentedDataSet.json',
#     "rocksalts" : '2018-Q1-rocksalts.json',
# }
# datadir = "/home/fmm/phd/perovskite_db/jared/"
# outdir = f"{datadir}/v1.0.0"
# os.makedirs(outdir, exist_ok=1)
# for k, fn in datafiles.items():
#     basepath = f"{datadir}/{fn}"[:-5]
#     data = json.load(open(basepath+".json", 'r'))
#     od = get_order_dict(data, prefix=f"{k}-")
#     json.dump(od, open(f"{outdir}/{k}.lut.json", 'w'))
#     new_data = apply_order_dict(data, od)
#     json.dump(new_data, open(f"{outdir}/{k}.json", 'w'))
    
