import os
import re

import numpy as np
import numpy.random

import ase.io

#from .special_features import get_ionradius, get_electron_affinity, get_pauling_en

from joblib import Parallel, delayed

# the dataset from the nomad-kaggle-
UNIQUE_LABEL = "n"
CORE_PROPERTIES = [
    "gap", "formation_energy_pa"
]
REALSTRUC = ["struc",]


def xyz_to_dict(path, nofeatures=False):
    item = {}
    formation_energy_pa = None
    gap = None
    species = []
    positions = []
    cell = []
    # primitive... every file is struc-<number>.xyz
    n = int(os.path.split(path)[1][6:-4])
    with open(path) as sf:
        total_atoms = int(sf.readline())
        formation_energy_pa, gap, _ = [float(x) for x in sf.readline().split()]

        for atom in range(total_atoms):
            atominfo = sf.readline().split()
            species.append(atominfo[0])
            pos = [float(p) for p in atominfo[1:4]]

            positions.append(pos)

        for v in range(3):
            cellvec = [float(c) for c in sf.readline().split()]
            cell.append(cellvec)
    struc = ase.Atoms(symbols=species, positions=positions, cell=cell)
    item["n"] = n
    item["struc"] = struc
    item["gap"] = gap
    item["formation_energy_pa"] = formation_energy_pa

    return item

def load(db_folder, parallel=4, nofeatures=False):
    filenames = ["{}/{}".format(db_folder, f)
                for f in os.listdir(db_folder) if f.endswith(".xyz")]

    structures = Parallel(n_jobs=parallel)(delayed(xyz_to_dict)(fn, nofeatures) for fn in filenames)
    structures.sort(key=lambda x: x["n"])
    return structures
