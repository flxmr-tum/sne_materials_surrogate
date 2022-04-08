from joblib import Memory, Parallel, delayed
from itertools import chain
import os

import numpy as np
import pandas as pd

import ase
import ase.io
from ase.data import atomic_numbers

import mendeleev as ml
from . property_tables import TABLES, average_countdict

UNIQUE_LABEL = "id"
CORE_PROPERTIES = [
    "bandgap_th", "bandgap_opt"
]
REALSTRUC = ["struc",]

def read_cif_to_dict(cpath):
    file_id = os.path.split(cpath)[1][:-4]
    try:
        file_id_num = int(file_id)
        struc = ase.io.read(cpath)
        return (file_id_num, struc)
    except:
        print(f"Error parsing file {cpath}")

def load(db_path, parallel=4, filtergaps=True):
    metadata = pd.read_csv(f"{db_path}/desc.csv")
    metadata.set_index(UNIQUE_LABEL, inplace=True)

    # convert bandgaps to numerical columns
    ciffolder = f"{db_path}/cifs_converted"
    cifs = [f"{ciffolder}/{fn}" for fn in os.listdir(ciffolder)]
    cifs.sort()
    structures = [read_cif_to_dict(fn) for fn in cifs if fn]
    #structures = Parallel(n_jobs=parallel)(delayed(read_cif_to_dict)(fn) for fn in cifs)
    structures = filter(lambda x: True if x else False, structures)
    struc_ids, strucs = list(zip(*structures))
    struc_df = pd.DataFrame({ REALSTRUC[0] : strucs}, index=struc_ids)

    s_idx = set(struc_df.index)
    # filter out NANs if column contains them
    if filtergaps:
        m_idx = set(metadata["bandgap_th"].dropna().index)
    else:
        m_idx = set(metadata.index)
    # all idx, where a structure is available are valid
    valid_idx = m_idx.intersection(s_idx)
    
    return metadata.loc[valid_idx], struc_df.loc[valid_idx]

