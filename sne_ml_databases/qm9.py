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

"""
reference data
=========================================================================================================
  Ele-    ZPVE         U (0 K)      U (298.15 K)    H (298.15 K)    G (298.15 K)     CV
  ment   Hartree       Hartree        Hartree         Hartree         Hartree        Cal/(Mol Kelvin)
=========================================================================================================
   H     0.000000     -0.500273      -0.498857       -0.497912       -0.510927       2.981
   C     0.000000    -37.846772     -37.845355      -37.844411      -37.861317       2.981
   N     0.000000    -54.583861     -54.582445      -54.581501      -54.598897       2.981
   O     0.000000    -75.064579     -75.063163      -75.062219      -75.079532       2.981
   F     0.000000    -99.718730     -99.717314      -99.716370      -99.733544       2.981
=========================================================================================================
"""

U0_refs = {
    "H" : -0.500273,
    "C" : -37.846772,
    "N" : -54.583861,
    "O" : -75.064579,
    "F" : -99.718730
}

UNIQUE_LABEL = "id"
CORE_PROPERTIES = [
    "rotA", "rotB", "rotC", "mu", "alpha",
    "homo", "lumo", "gap",
    "r2", "zpve", "U0",
    "U", "H", "G", "CV",
    "U0_at_eV",
]
REALSTRUC = ["struc"]

class BlanklineIterator:
    def __init__(self, file2split):
        self._file = file2split
        self._openfile = False
        self._pos = 0
    
    def __iter__(self):
        return self

    def __getnextpart(self):
        lines = []
        filegood = False
        for li in self._openfile:
            filegood = True
            if li == '\n':
                break
            else:
                lines.append(li)

        lines = "".join(lines)

        if filegood:
            return lines
        else:
            return None

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

def parse_xyz_to_dict(xyz : str):
    lines = xyz.split("\n")
    
    total_atoms = int(lines[0])
    infoline = lines[1]
    last_atom_line = 2+total_atoms
    species = []
    positions = []
    charges = []
    for l in range(2, last_atom_line):
        atomline = lines[l].split()
        species.append(atomline[0])
        positions.append([float(i) for i in atomline[1:4]])
        charges.append(float(atomline[4]))
    frequencyline = lines[last_atom_line+1]
    smiles_start, smile_relaxed = lines[last_atom_line+1].split()
    inchi_start, inchi_relaxed = lines[last_atom_line+2].split()

    info = {}
    info["struc"] = ase.Atoms(symbols=species, positions=positions, charges=charges)
    info["smiles_start"] = smiles_start
    info["smile_relaxed"] = smile_relaxed
    info["inchi_start"] = inchi_start
    info["inchi_relaxed"] = inchi_relaxed
    infoline_split = infoline.split()
    info["id"] = int(infoline_split[1])
    info = {**info, **dict(zip(CORE_PROPERTIES, [float(i) for i in infoline_split[2:]]))}
    # TODO: handle frequencyline?
    HARTREE2EV = 27.211386245988 # CODATA
    info["U0_at_eV"] = info["U0"] - \
        sum([U0_refs[k]*v for k, v in info["struc"].symbols.formula.count().items()])
    info["U0_at_eV"] *= HARTREE2EV
    
    return info
    


def load(db_path, parallel=4):
    file_iterator = BlanklineIterator(db_path)
    info_dicts = Parallel(n_jobs=parallel)(delayed(parse_xyz_to_dict)(xyz) for xyz in file_iterator)
    info_dicts.sort(key=lambda x: x["id"])
    return info_dicts
