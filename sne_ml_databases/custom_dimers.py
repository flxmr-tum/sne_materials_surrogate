from joblib import Memory, Parallel, delayed
from itertools import chain
import os
import re

import numpy as np
import pandas as pd

import ase
import ase.io
from ase.data import atomic_numbers

import mendeleev as ml
from . property_tables import TABLES, average_countdict

UNIQUE_LABEL = "id"
CORE_PROPERTIES = [
    "ti"
]
REALSTRUC = ["struc"]

class TripleBlankLineIterator(object):
    def __init__(self, file2split, entry_start=):
        self._file = file2split
        self._openfile = False
        self._pos = 0
    
    def __iter__(self):
        return self

    def __getnextpart(self):
        lines = []
        filegood = False
        empty_lines = 0
        empty_line_before = False
        for li in self._openfile:
            filegood = True
            if li == '\n':
                if not empty_line_before:
                    empty_lines = 1
                else:
                    empty_lines += 1
            if emtpy_lines == 3:
                break
            else:
                lines.append(li)
                empty_line_before = False

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


def parse_snippet_to_dict(snippet : str):
    parts = re.split(r'^>+$', snippet)
    properties = parts[0]

    substrucs = []
    for s in parts[1:]:
        substrucs.append(s)

    
    
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
    
    return info


def load(db_path, parallel=4):
    file_iterator = DimerIterator(db_path)
    info_dicts = Parallel(n_jobs=parallel)(delayed(parse_snippet_to_dict)(snippet) for snippet in file_iterator)
    info_dicts.sort(key=lambda x: x["id"])
    return info_dicts
    
