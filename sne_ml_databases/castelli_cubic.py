from joblib import Memory, Parallel, delayed
from itertools import chain

import numpy as np
import numpy.random

import ase
import ase.db
from ase.data import atomic_numbers

import mendeleev as ml
from . property_tables import TABLES, average_countdict


UNIQUE_LABEL = "id"
SITE_PROPERTIES = ["r", "ea", "en", "cov_r", "val_s", "val_p"]
SITE_PROPERTIES_EXPANDED = list(chain(*[[f'{x}_a', f'{x}_b', f'{x}_xmix']
                                        for x in SITE_PROPERTIES]))
CORE_PROPERTIES = [
    'project', 'combination',
    'B_ion',  'A_ion', 'anion', 
    'VB_ind', 'CB_ind', 'gllbsc_ind_gap',
    'VB_dir', 'CB_dir', 'gllbsc_dir_gap',
    'heat_of_formation_all', 'standard_energy',
    'heat_of_formation_all',
    'heat_of_formation_all_pa', # eV!
    ### to calculate
    'minimal_gap',
    ### above from the database, other stuff from mendeleev
]
CORE_PROPERTIES += SITE_PROPERTIES_EXPANDED
STRUC = [
    'struc'
]

def _loadentry(entry, reference=False, cull_struc=None, replace_pos=None):
    # discard reference calcs which are in the database for reference
    if entry.get("reference"):
            return None

    info = entry.key_value_pairs
    #print("Loading ", entry)

    info["minimal_gap"] = min(info["gllbsc_ind_gap"], info["gllbsc_dir_gap"])
    
    info["id"] = entry.unique_id
    info["struc"] = entry.toatoms()
    info["heat_of_formation_all_pa"] = info["heat_of_formation_all"]/len(info["struc"])
    
    site_a = info["A_ion"]
    site_b = info["B_ion"]
    site_x = ase.formula.Formula(info["anion"])

    site_x = site_x.count()
    site_x = dict((x, c) for x, c in site_x.items())

    data_tables = dict(
        (k, TABLES[k]) for k in SITE_PROPERTIES)
    
    for site, species in [("a", site_a),
                          ("b", site_b),
                          ("xmix", site_x)]:
        for k in data_tables.keys():
            if isinstance(species, str):
                info[f'{k}_{site}'] = data_tables[k][species]
            elif isinstance(species, dict):
                info[f'{k}_{site}'] = average_countdict(species, data_tables[k])
            else:
                print(k, site, species)
                print("something's wrong with the database")
                raise NotImplementedError()

    if cull_struc:
        sel_array = None
        for s in cull_struc:
            sel = (np.array(info["struc"].get_chemical_symbols()) == s)
            if sel_array is not None:
                sel_array += sel
            else:
                sel_array = sel
        if replace_pos:
            ns = info["struc"].numbers
            ns[np.logical_not(sel_array)] = 1
            info["struc"].numbers = ns
        else:
            info["struc"] = info["struc"][sel_array]
            
    return info

def load(db_path, parallel=4, nonzero_gap=False, cull_struc=None, replace_pos=None):
    rows = list(ase.db.connect(db_path).select())
    results = Parallel(n_jobs=parallel)(delayed(_loadentry)(row,
                                                            cull_struc=cull_struc,
                                                            replace_pos=replace_pos)\
                                        for row in rows)
    #results = [_loadentry(row) for row in rows]
    results = [r for r in results if (r is not None)]
    if nonzero_gap:
        results = [r for r in results if r["minimal_gap"] > 0]
    results.sort(key=lambda x: x["id"])
    return results
