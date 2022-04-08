from joblib import Parallel, delayed
from collections import namedtuple, OrderedDict
from collections.abc import Sized
from dataclasses import dataclass
from typing import List, Dict, Tuple
import functools
import joblib
import os
from datetime import datetime

#
import sklearn.model_selection as skms

# for type hinting
import pandas as pd
import numpy as np
import ase

# store train/test/validation-splits in a named tuple
def timecp(msg=""):
    print("{}: {}".format(datetime.now().strftime("%H:%M:%S.%f"), msg))

TTV_split = namedtuple("TTV_split", ["train", "val", "test"])

def sselect_dict(d, indices):
    out_d = {}
    for k in d.keys():
        if isinstance(d[k], list):
            t = d[k]
            out_d[k] = [t[i] for i in indices]
        elif d[k] is None:
            out_d[k] = None
        else:
            out_d[k] = d[k][indices]
    return out_d

@dataclass
class GraphBunch:
    nodes: list
    edges: list
    edge_attrs: list
    meta: List[Dict] = None

@dataclass
class GraphBunchShape:
    node_shape: Tuple
    edge_shape: Tuple

@dataclass
class AtomisticGraph:
    atom_features : list
    bonds : list
    bond_features : list
    bonds_sparse : bool = True
    meta : dict = None

@dataclass
class AtomisticDataInfo:
    is_graph: bool
    max_atoms: bool
    orig_shape: tuple
    pad_needed: bool
    pad_possible: bool

class DataMismatchError(Exception):
    def __init__(self, msg=""):
        super().__init__()
        print(f"DataMismatchError: {msg}")

class DataNotAvailableError(Exception):
    def __init__(self, msg=""):
        super().__init__()
        print(f"DataNotAvailableError: {msg}")

class AtomisticMLContainer(object):
    """ 
    In memory store for:
    - basic information per compound/system
    - a list of all compounds as ase.Atoms
    - a replacement dict for each compound, if it's primitive coarse-graining
    - other representations of the compounds
    - per system data/graphs
    - per system-unit data/graphs
    """
    
    def __init__(self,
                 core_features,
                 strucs, # either list of ase.Atoms or
                 strucs_meta=None, # metadata for any structures
                 data : Dict[str, list] = None, #
                 data_meta : Dict[str, AtomisticDataInfo] = None, #
                 targetmap=None
                 ):
        if strucs_meta is None:
            strucs_meta = {}
        if data is None:
            data = {}
        if data_meta is None:
            data_meta = {}
        if targetmap is None:
            targetmap = {}
        if not isinstance(core_features, pd.DataFrame):
            raise DataMismatchError("Dataframe necessary for core features!")
        self._core_features : pd.DataFrame = core_features
        self._entries = len(self._core_features)

        self._strucs = {}
        self._strucs_meta = {}
        self._struc_lens = {}
        if isinstance(strucs, list):
            if len(strucs) != self._entries:
                raise DataMismatchError("number of strucs != # of features")
            if isinstance(strucs[0], ase.Atoms):
                self._strucs[0] = strucs
            elif isinstance(strucs[0], tuple) and isinstance(strucs[0][0], ase.Atoms):
                self._strucs[0] = [s[0] for s in strucs]
                self._strucs_meta[0] = [s[1] if len(s) > 1 else None for s in strucs]
            else:
                print("Should not happen")
                print(strucs[0])

        elif isinstance(strucs, dict):
            for k in strucs.keys():
                if len(strucs[k]) != self._entries:
                    print(k, len(strucs[k]), self._entries)
                    raise DataMismatchError(f"number of strucs (subpart {k}) != # of features")
                if strucs_meta.get(k):
                    if len(strucs_meta[k]) != self._entries:
                        raise DataMismatchError(f"number of struc_meta (subpart {k}) != # of features")
                self._strucs[k] = strucs[k]
                self._strucs_meta[k] = strucs_meta.get(k, None)
        else:
            print("ERROR: Structures needed")
            raise NotImplementedError

        for k in self._strucs.keys():
            self._struc_lens[k] = np.array([len(s) for s in self._strucs[k]])

        if set(data.keys()) != set(data_meta.keys()):
            raise DataMismatchError("Mismatching data metadata, this is not working!")

        for k in data.keys():
            if len(data[k]) != self._entries:
                raise DataMismatchError(f"number of datapoints (subpart {k}): {len(data[k])} "
                                        f"!= # of features: {self._entries}")
 
        self._data = data
        self._data_meta = data_meta
        self._targetmap = targetmap

        self._basic_fp_name = "fps"
        self._basic_graph_name = "graphs"
        self._basic_encoding = "encoding"

    def split3(self, test_size_in, validation_size_in, shuffle_seed=1868, shuffle_state=True) -> Tuple:
        if True:
            total_samples = self._entries
            indices = list(range(total_samples))
            if validation_size_in:
                val_size = max(int(round(validation_size_in*total_samples)),1)
            else:
                val_size = 0
            if test_size_in:
                test_size = max(int(round(test_size_in*total_samples)),1)
            else:
                test_size = 0
            if test_size != 0:
                _, test_idxs = skms.train_test_split(
                    indices,
                    test_size=test_size, shuffle=shuffle_state,
                    random_state=shuffle_seed)
            else:
                _ = indices
                test_idxs = []
            if val_size != 0:
                train_idxs, val_idxs = skms.train_test_split(
                    _, test_size=val_size, shuffle=shuffle_state,
                    random_state=shuffle_seed)
            else:
                train_idxs = _
                val_idxs = []
            print(f"SPLIT: dataset of size {total_samples} into test@{len(test_idxs)} and validation@{len(val_idxs)}")
            splits = ["train", "test", "val"]
            s3 = {}
            #print(train_idxs, test_idxs, val_idxs)
            for part, indices in zip(
                    splits, [train_idxs, test_idxs, val_idxs]):
                if indices:
                    s3[part] = self.__getpart_by_indices(indices)
                else:
                    s3[part] = None
            return TTV_split(**s3)
        else:
            return None

    def __getitem__(self, idxs):
        if isinstance(idxs, tuple):
            idxs = list(idxs)
        elif isinstance(idxs, int):
            idxs = [idxs,]
        elif isinstance(idxs, slice):
            idxs = list(range(idxs.start if idxs.start else 0,
                              idxs.stop if idxs.stop else self._entries,
                              idxs.step if idxs.step else 1))
        elif isinstance(idxs, Sized) and len(idxs) > 0:
            if isinstance(idxs[0], bool):
                #print("Executing boolean path")
                if len(idxs) != self._entries:
                    raise IndexError
                idxs = [idx for idx in range(self._entries) if idxs[idx]]
            elif isinstance(idxs[0], int):
                idxs = idxs
            else:
                raise IndexError
        else:
            raise IndexError
        #print(idxs[:10])
        return self.__getpart_by_indices(idxs)

    def __getpart_by_indices(self, indices):
        if indices is not None:
            return AtomisticMLContainer(
                core_features=self._core_features.iloc[indices],
                strucs=sselect_dict(self._strucs, indices),
                strucs_meta=sselect_dict(self._strucs_meta, indices) if self._strucs_meta else {},
                data=sselect_dict(self._data, indices),
                data_meta=self._data_meta,
                targetmap=self._targetmap)
        else:
            return None

    def apply_ase_fp(self, fp_func, target=None, to=0, use_meta=False, ncpus=4):
        if target is None:
            target = self._basic_fp_name
        timecp("applying ase_fp")
        strucs = self._strucs[to]
        strucs_meta = self._strucs_meta.get(to, None)
        struc_lens = self._struc_lens[to]
        if use_meta and strucs_meta:
            fps = Parallel(n_jobs=ncpus)(
                delayed(lambda s_, meta_:
                        fp_func(s_, meta_))(s, meta)
                for s, meta
                in zip(strucs, strucs_meta))
        else:
            fps = Parallel(n_jobs=ncpus)(
                delayed(lambda s_:
                        fp_func(s_))(s)
                for s
                in strucs)
        timecp("finished fingerprinting with ase_fp")
        max_atoms = False
        paddable = True
        fp_shape = (1,)
        extra_pad_needed = False

        if isinstance(fps[0], Sized):
            fps_lens = np.array([len(fp) for fp in fps])
            if np.all(fps_lens == struc_lens):
                max_atoms = max(fps_lens)
            
            if isinstance(fps[0], np.ndarray) or (max_atoms and isinstance(fps[0][0], np.ndarray)):
                # TODO: will error when provided with multiple dimensionality in the fingerprints
                fps_shapes = np.array([fp.shape for fp in fps])
                max_shape = np.max(fps_shapes, axis=0)
                min_shape = np.max(fps_shapes, axis=0)
                if max_atoms:
                    max_shape = max_shape[1:]
                    min_shape = min_shape[1:]
                if np.all(max_shape == min_shape):
                    fp_shape = max_shape
                else:
                    fp_shape = max_shape
                    extra_pad_needed = True
            elif max_atoms and not isinstance(fps[0][0], np.ndarray):
                paddable = False

            if not max_atoms and not extra_pad_needed:
                fps_final = np.array(fps)
                fps_meta = AtomisticDataInfo(is_graph=False, max_atoms=False, orig_shape=fp_shape,
                                             pad_needed=False, pad_possible=paddable)
            else:
                fps_final = fps
                fps_meta = AtomisticDataInfo(is_graph=False, max_atoms=max_atoms, orig_shape=fp_shape,
                                             pad_needed=extra_pad_needed, pad_possible=paddable)
        else:
            fps_final = fps
            fps_meta = AtomisticDataInfo(is_graph=False, max_atoms=False, orig_shape=fp_shape,
                                         pad_needed=False, pad_possible=False)
        timecp("analysed format for ase_fp")
        self._data[target] = fps_final
        self._data_meta[target] = fps_meta

    def apply_ase_graph(self, graph_func, target=None, to=0, ncpus=4):
        if target is None:
            target = self._basic_graph_name
        strucs = self._strucs[to]
        #strucs_meta = self._strucs_meta.get(to, None)
        struc_lens = self._struc_lens[to]
        timecp("applying graph_func")
        graphs = Parallel(n_jobs=ncpus)(
            delayed(lambda s_,:
                    graph_func(s_))(s)
            for s
            in strucs)
        # TODO: check that bonds are indeed sparse!
        try:
            graphs = [AtomisticGraph(af, adj, adj_feat,  bonds_sparse=True, meta=meta) for af, adj, adj_feat, meta in graphs]
        except ValueError:
            raise DataMismatchError("Can't unpack graph representations. graph_func should return a triplet"
                                    "(atom_features, adjacency_matrix, adjacency_features, metathings)...")
        timecp("finished graph_func")
        self._data[target] = graphs
        self._data_meta[target] = AtomisticDataInfo(is_graph=True, max_atoms=max(struc_lens),
                                                    orig_shape=None, pad_needed=False, pad_possible=False)


    def add_encoding(self, encoded, target=None):
        if target is None:
            target = self._basic_encoding
        if not isinstance(encoded, np.ndarray):
            raise NotImplementedError
        if not len(encoded) == self._entries:
            raise DataMismatchError("Encoding has to have the same length as the data")
        if len(encoded.shape) == 1:
            self._data[target] = encoded.reshape((self._entries, -1))
        else:
            self._data[target] = encoded
        self._data_meta[target] = AtomisticDataInfo(is_graph=False, max_atoms=False, orig_shape=encoded.shape[1:],
                                                    pad_needed=False, pad_possible=False)


    def get(self, name):
        # get the respective data, as stored
        out = None
        if isinstance(name, list):
            out = self._core_features[name]
        elif name in self._data_meta.keys():
            meta = self._data_meta[name]
            if meta.is_graph:
                out = GraphBunch(
                    *zip(
                        *[(g.atom_features, g.bonds, g.bond_features, g.meta) for g in self._data[name]]
                    ))
            else:
                out = self._data[name]
        elif name in self.core_features_names:
            out = self._core_features[[name]]
        elif name in self._targetmap:
            # TODO: is this recursion bad?
            out = self.get(self._targetmap[name])
        else:
            raise DataNotAvailableError(f"no attribute {name}")
        if isinstance(out, pd.Series):
            out = np.array(out).reshape(len(out), 1)
        elif isinstance(out, pd.DataFrame):
            out = np.array(out)
        return out

    def get_padded(self, name):
        meta = self._data_meta[name]
        padded = None
        timecp(f"Padding started ({name})")
        if meta.is_graph:
            raise DataNotAvailableError("Padding not possible for graph data "
                                        "(do it in an extra step with a little more 'graph'-like representations)")
        elif not meta.pad_possible:
            raise DataNotAvailableError("Padding not possible")
        elif (not meta.max_atoms) and (not meta.pad_needed):
            padded = self.get(name)
        elif meta.max_atoms:
            if meta.pad_needed:
                raise NotImplementedError
            else:
                data = self._data[name]
                pad_atoms = meta.max_atoms
                non_pad_dims = len(meta.orig_shape)

                def pad_array(ar):
                    return np.pad(ar, [(0, pad_atoms - len(ar)),] + [(0,0),]*non_pad_dims, 'constant')

                padded = np.array([pad_array(ar) for ar in data])
        elif meta.pad_needed:
            raise NotImplementedError
        else:
            print("Why did this happen?")
            raise NotImplementedError
        timecp("Padding successful")
        return padded


    def get_per_atom(self, name):
        # get the data flattened, add a index mapping to compound ids
        meta = self._data_meta[name]
        concat = None
        idxmap = None
        timecp(f"per-atom concatentation started (name)")
        if meta.is_graph:
            raise DataNotAvailableError("concatentation not possible for graph data "
                                        "(do it in an extra step with a little more 'graph'-like representations)")
        elif not meta.pad_possible:
            raise DataNotAvailableError("concatentation not possible for data, if padding not possible")
        elif meta.max_atoms:
            if meta.pad_needed:
                raise NotImplementedError
            else:
                lens = np.array([len(d) for d in self._data[name]])
                idxs = np.array(self._core_features.index)
                concat = np.vstack(self._data[name])
                idxmap = np.zeros((len(concat)), dtype=idxs.dtype)
                count = 0
                newcount = 0
                for l, idx in zip(lens, idxs):
                    newcount += l
                    idxmap[count:newcount] = idx
                    count = newcount
        else:
            raise DataNotAvailableError(f"per-atom-concatentation not possible ({name})")
        timecp("concatenation finished")
        return concat, idxmap

    def get_info(self, name) -> AtomisticDataInfo:
        # return whether the property is per-node or per-compound-unit
        return self._data_meta[name]

    def get_structures(self, sid=0):
        return self._strucs[sid]

    def get_structures_meta(self, sid=0):
        return self._strucs_meta[sid]

    def get_structures_info(self, sid=0):
        max_atoms = max([len(s) for s in self._strucs[sid]])
        chemical_symbols = [s.get_chemical_symbols() for s in self._strucs[sid]]
        all_elements = set()
        for cs in chemical_symbols:
            all_elements.update(cs)
        has_meta = True if sid in self._strucs_meta.keys() else False
        return {"max_size" : max_atoms, "elements" : all_elements, "has_meta" : has_meta}

    def add_targetmap(self, targetmapping):
        self._targetmap = {**self._targetmap, **targetmapping}

    def get_target(self, target):
        return self._core_features[self._targetmap.get(target, [])]

    @property
    def core_features(self):
        return self._core_features

    @property
    def structures(self):
        ks = list(self._strucs.keys())
        if 0 in ks:
            return self._strucs.get(0)
        else:
            return self._strucs.get(ks[0])

    @property
    def fingerprints(self):
        return self.get(self._basic_fp_name)

    @property
    def graphs(self):
        return self.get(self._basic_graph_name)

    @property
    def encoded(self):
        return self.get(self._basic_encoding)

    @property
    def core_features_names(self):
        return list(self._core_features.columns)
