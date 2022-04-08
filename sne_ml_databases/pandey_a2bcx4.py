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
    'name', 'prototype',
    'energy',
    'gllbsc_ind_gap', 'gllbsc_dir_gap',
    "energy_pa", #eV
    "energy_pa_mev", #meV
    ### to calculate
    'minimal_gap',
    # NO INTEGRATED_FEATURES!
]
STRUC = [
    'struc'
]

def _loadentry(entry):
    info = entry.key_value_pairs
    info["id"] = entry.unique_id
    info["gllbsc_ind_gap"] = info["GLLB_ind"]
    info["gllbsc_dir_gap"] = info["GLLB_dir"]
    info["minimal_gap"] = min(info["gllbsc_ind_gap"], info["gllbsc_dir_gap"])
    info["energy"] = info["mbeef_en"]
    info["energy_pa_mev"] = info["E_relative_per_atom"]
    info["energy_pa"] = info["E_relative_per_atom"]/1000 # meV→ eV
    info["struc"] = entry.toatoms()
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
