# https://figshare.com/articles/dataset/Materials_Project_Data/7227749
from joblib import Memory, Parallel, delayed
from itertools import chain
import os
import json

import numpy as np
import pandas as pd

import ase
import ase.io
from ase.data import atomic_numbers

UNIQUE_LABEL = "mpid"
CORE_PROPERTIES = [
    'formula', 'e_hull', 'gap pbe', 'mu_b', 'elastic anisotropy',
    'bulk modulus', 'shear modulus', "e_form"
]
REALSTRUC = ['struc']
# 'initial structure' as an option?

def subselect_mp_ids(data, idfile=None):
    mp_ids = open(idfile, "r").read().split()
    indexmap = dict((k, i) for k, i in zip(data.core_features.index, range(len(data.core_features))))
    subselect_list = []
    for m in mp_ids:
        if m not in indexmap.keys():
            print(f"missing id {m}")
            continue
        subselect_list.append(indexmap[m])
    return data[subselect_list]

def parse_structure(struclist, m):
    info = {}
    info["mpid"] = struclist[m["mpid"]]
    for p in CORE_PROPERTIES:
        info[p] = struclist[m[p]]
    sd = struclist[m["structure"]]
    cell = np.array(sd["lattice"]["matrix"])
    sites = sd["sites"]
    tot_sites = len(sites)
    positions = np.zeros((tot_sites, 3), np.float)
    names = np.zeros((tot_sites), dtype='<U2')
    for idx in range(tot_sites):
        positions[idx] = sites[idx]['xyz']
        names[idx] = sites[idx]['label']
    info[REALSTRUC[0]] = ase.Atoms(symbols=names, positions=positions, cell=cell)
    return info
    

def load(db_path, parallel=4, minimal_gap=0):
    datadict = json.load(open(db_path))
    mapper = dict((n, i) for i, n in enumerate(datadict["columns"]))
    info_dicts = Parallel(n_jobs=parallel)(delayed(parse_structure)(datalist, mapper) for datalist in datadict["data"])
    info_dicts = [r for r in info_dicts if r["gap pbe"] >= minimal_gap]
    #info_dicts.sort(key=lambda x: x["mpid"])
    del datadict
    return info_dicts
