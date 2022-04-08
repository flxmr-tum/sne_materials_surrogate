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
META_PROPERTIES = [
    "energy"
]
CORE_PROPERTIES = [
    "energy",
]
REALSTRUC = ["struc"]

class XYZTrajIterator:
    def __init__(self, file2split):
        self._file = file2split
        self._openfile = False
        self._pos = 0
    
    def __iter__(self):
        return self

    def __getnextpart(self):
        head = self._openfile.readline()
        if head:
            try:
                count = int(head)
                self._pos += 1
            except:
                raise Exception
        else:
            return None
        meta = self._openfile.readline()
        atoms = [self._openfile.readline().strip() for _ in range(count)]
        return (self._pos, meta, atoms)
        

    def __next__(self):
        if self._openfile:
            pass
        else:
            self._openfile = open(self._file, 'r')

        res = self.__getnextpart()

        if res is not None:
            return res
        else:
            raise StopIteration

def parse_xyz_to_dict(idx, meta, struc):
    info = dict((k, float(v)) for k, v in zip(META_PROPERTIES, meta.split()))
    info["id"] = idx
    specs = []
    positions = []
    forces = []
    for part in struc:
        sp = part.split()
        specs.append(sp[0])
        num_sp = [float(e) for e in sp[1:]]
        positions.append(num_sp[0:3])
        forces.append(num_sp[3:])
    info["struc"] = ase.Atoms(symbols=specs, positions=positions)
    info["forces"] = np.array(forces)
    return info
    


def load(db_path, parallel=4):
    file_iterator = XYZTrajIterator(db_path)
    #info_dicts = [parse_xyz_to_dict(idx, meta, struc) for idx, meta, struc in file_iterator]
    info_dicts = Parallel(n_jobs=parallel)(delayed(parse_xyz_to_dict)(idx, meta, struc) for idx, meta, struc in file_iterator)
    info_dicts.sort(key=lambda x: x["id"])
    return info_dicts
