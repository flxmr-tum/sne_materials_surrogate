import data_tools as dt

from joblib import Memory, Parallel, delayed
memory = Memory("workflows_cache", verbose=0)
#from functools import lru_cache
from itertools import chain

from typing import List

import numpy as np
import pandas as pd

import ase
import ase.db


from . import kimdb, castelli_cubic, castelli_lowsym, pandey_a2bcx4, marchenko_2d, sutton, qm9, stanley, mp_2018, schramm, md17
from .util import dicts2df

__LOAD_DB = {}

def _load_castelli_cubic(db_path, parallel=4, nonzero_gap=False, cull_struc=None, replace_pos=None):
    #LOADING IS FASTEST WITHOUT CACHING!
    #results = memory.cache(castelli_cubic.load)(db_path, parallel)
    results = castelli_cubic.load(db_path, parallel,
                                  nonzero_gap=nonzero_gap,
                                  cull_struc=cull_struc, replace_pos=replace_pos)
    property_df = dicts2df(results,
                           uid=castelli_cubic.UNIQUE_LABEL,
                           columns=castelli_cubic.CORE_PROPERTIES)
    structures = [s[castelli_cubic.STRUC[0]] for s in results]
    data = dt.AtomisticMLContainer(property_df, structures)
    return data

def _load_castelli_lowsym(db_path, parallel=4, minimal_gap=0):
    results = castelli_lowsym.load(db_path, parallel, minimal_gap=minimal_gap)
    property_df = dicts2df(results,
                           uid=castelli_lowsym.UNIQUE_LABEL,
                           columns=castelli_lowsym.CORE_PROPERTIES)
    structures = [s[castelli_lowsym.STRUC[0]] for s in results]
    data = dt.AtomisticMLContainer(property_df, structures)
    return data

def _load_pandey_a2bcx4(db_path, parallel=4, minimal_gap=0):
    results = pandey_a2bcx4.load(db_path, parallel, minimal_gap=minimal_gap)
    property_df = dicts2df(results,
                           uid=pandey_a2bcx4.UNIQUE_LABEL,
                           columns=pandey_a2bcx4.CORE_PROPERTIES)
    structures = [s[pandey_a2bcx4.STRUC[0]] for s in results]
    data = dt.AtomisticMLContainer(property_df, structures)
    return data

def _load_kimdb(db_folder, parallel=4, simplified=True):
    #db_dicts = memory.cache(kimdb.load)(db_folder, parallel)
    db_dicts = kimdb.load(db_folder, parallel)
    property_df = dicts2df(db_dicts,
                           uid=kimdb.UNIQUE_LABEL,
                           columns=kimdb.CORE_PROPERTIES)
    structures = list([tuple(s[k] for k in kimdb.PEROVSKITESTRUC[simplified]) for s in db_dicts])
    data = dt.AtomisticMLContainer(property_df, structures)
    return data

def _load_sutton(db_folder, parallel=4):
    print("Loading in parallel", parallel)
    #db_dicts = memory.cache(sutton.load)(db_folder, parallel)
    db_dicts = sutton.load(db_folder, parallel)
    property_df = dicts2df(db_dicts,
                           uid=sutton.UNIQUE_LABEL,
                           columns=sutton.CORE_PROPERTIES)
    structures = [s[sutton.REALSTRUC[0]] for s in db_dicts]
    data = dt.AtomisticMLContainer(property_df, structures)
    return data

def _load_marchenko(db_folder, parallel=4, filtergaps=True):
    md, s = marchenko_2d.load(db_folder, parallel, filtergaps=True)
    data = dt.AtomisticMLContainer(md, s.iloc[:,0].to_list())
    return data

def _load_stanley(db_path, parallel=4, relaxed=True):
    db_dicts = stanley.load(db_path, parallel=parallel, relaxed=relaxed)
    property_df = dicts2df(db_dicts,
                           uid=stanley.UNIQUE_LABEL,
                           columns=stanley.CORE_PROPERTIES)
    structures = [s[stanley.REALSTRUC[0]] for s in db_dicts]
    data = dt.AtomisticMLContainer(property_df, structures)
    return data

def _load_qm9(dbfile, parallel=4, drop_ids=None):
    db_dicts = memory.cache(qm9.load)(dbfile, parallel=parallel)
    if drop_ids:
        bad_ids = list(pd.read_csv(drop_ids, header=None).iloc[:,0])
        db_dicts = list(filter(lambda x: x[qm9.UNIQUE_LABEL] not in bad_ids, db_dicts))
    property_df = dicts2df(db_dicts,
                           uid=qm9.UNIQUE_LABEL,
                           columns=qm9.CORE_PROPERTIES)
    structures = [s[qm9.REALSTRUC[0]] for s in db_dicts]
    data = dt.AtomisticMLContainer(property_df, structures)
    return data

def _load_mp_2018(dbfile, parallel=4, minimal_gap=0, subselect_list=None):
    db_dicts = mp_2018.load(dbfile, parallel=parallel, minimal_gap=minimal_gap)
    property_df = dicts2df(db_dicts,
                           uid=mp_2018.UNIQUE_LABEL,
                           columns=mp_2018.CORE_PROPERTIES)
    structures = [s[mp_2018.REALSTRUC[0]] for s in db_dicts]
    data = dt.AtomisticMLContainer(property_df, structures)
    if subselect_list:
        data = mp_2018.subselect_mp_ids(data, idfile=subselect_list)
    return data

def _load_dimer_v1(dbfile, parallel=4):
    pass

def _load_schramm(dbdir, parallel=4, simplified=False):
    db_dicts = schramm.load(dbdir)
    db_dicts = [d for d in db_dicts if (d["gap"] and d[schramm.SIMPLE_STRUCS])]
    property_df = dicts2df(db_dicts,
                           uid=schramm.UNIQUE_ID,
                           columns=schramm.PROPDIRS+schramm.EXTRA_INFOS)
    if simplified:
        structures = [s[schramm.SIMPLE_STRUCS] for s in db_dicts]
    else:
        structures = [s[schramm.MAIN_STRUCS] for s in db_dicts]
    data = dt.AtomisticMLContainer(property_df, structures)
    return data

def _load_md17(dbfile, parallel=4):
    db_dicts = md17.load(dbfile)
    strucs = [s["struc"] for s in db_dicts]
    forces = [s["forces"] for s in db_dicts]
    prop_df = dicts2df(db_dicts,
                       uid="id",
                       columns=md17.CORE_PROPERTIES)
    data = dt.AtomisticMLContainer(prop_df, strucs,
                                   data={
                                       "forces" : forces
                                   },
                                   data_meta={
                                       "forces" : dt.AtomisticDataInfo(
                                           False, 0, (), False, False)
                                   })
    return data

__LOAD_DB["kimdb"] = _load_kimdb
# https://cmr.fysik.dtu.dk/cubic_perovskites/cubic_perovskites.html#cubic-perovskites
__LOAD_DB["castelli_cubic"] = _load_castelli_cubic
# low_symmetry_perovskites: https://cmr.fysik.dtu.dk/low_symmetry_perovskites/low_symmetry_perovskites.html#low-symmetry-perovskites
__LOAD_DB["castelli_lowsym"] = _load_castelli_lowsym
# https://cmr.fysik.dtu.dk/a2bcx4/a2bcx4.html#a2bcx4
__LOAD_DB["pandey_A2BCX4"] = _load_pandey_a2bcx4
# https://arxiv.org/pdf/2006.14302.pdf
__LOAD_DB["marchenko_2D"] = _load_marchenko
# nomad kaggle challenge
__LOAD_DB["sutton"] = _load_sutton
# QM9 - no crystal, but still
__LOAD_DB["qm9"] = _load_qm9
# Jareds work
__LOAD_DB["stanley"] = _load_stanley
# Materials_Project 2018-dump
__LOAD_DB["mp2018"] = _load_mp_2018
# Lukas' 2D pervoskites
__LOAD_DB["schramm"] = _load_schramm
# the tetracene-DB from Obis course
__LOAD_DB["sne_dimer"] = _load_dimer_v1
# MD17-like abinitio trajectories
__LOAD_DB["md17"] = _load_md17

def load_normalized(db, dataset_type, targetmap={}, parallel=4, **opts) -> dt.AtomisticMLContainer:
    # load the dataset
    dt.timecp("Loading {}".format(dataset_type))
    data = __LOAD_DB[dataset_type](db, parallel=parallel, **opts)
    dt.timecp("Loaded {}".format(dataset_type))
    data.add_targetmap(targetmap)
    return data
