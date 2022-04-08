from joblib import Memory, Parallel, delayed
from itertools import chain

import numpy as np

import ase
import ase.db
from ase.data import atomic_numbers

import mendeleev as ml
from .property_tables import TABLES, average_countdict

UNIQUE_LABEL = "id"
SITE_PROPERTIES = ["r", "ea", "en", "cov_r", "val_s", "val_p"]
CORE_PROPERTIES = [
    'project', 'composition',
    'gllbsc_ind_gap', 'gllbsc_dir_gap',
    ### to calculate
    'energy',
    'energy_pa',
    'minimal_gap',
    # NO INTEGRATED_FEATURES!
]
STRUC = [
    'struc'
]

def _loadentry(entry):
    info = entry.key_value_pairs
    info["id"] = entry.unique_id
    info["minimal_gap"] = min(info["gllbsc_ind_gap"], info["gllbsc_dir_gap"])
    info["energy"] = entry.energy
    info["struc"] = entry.toatoms()
    info["energy_pa"] =  info["energy"]/len(info["struc"])
    return info

def load(db_path, parallel=4, minimal_gap=0):
    rows = list(ase.db.connect(db_path).select())
    #results = Parallel(n_jobs=parallel)(delayed(_loadentry)(row)\
    #                                    for row in rows)
    results = [_loadentry(row) for row in rows]
    results = [r for r in results if (r is not None)]
    results = [r for r in results if r["minimal_gap"] >= minimal_gap]
    results.sort(key=lambda x: x["id"])
    return results
