from joblib import Parallel, delayed
from collections import namedtuple, OrderedDict
from dataclasses import dataclass
from typing import List, Dict, Tuple
import functools
import joblib
import os

# for type hinting
import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection as skms
import sklearn.pipeline

from .AtomisticMLContainer import *

class WrappingImpossible(Exception):
    def __init__(self, msg=""):
        print(f"WrappingImpossible: {msg}")


class WrapScaler(object):
    def __init__(self, scalerbase):
        self._scaler = scalerbase()
        self._shape = None
        self._fitted = False

    @property
    def shape(self):
        return self._shape

    def _fw_shape(self, data):
        dl = len(data)
        # should create a view, so is efficient!
        return np.reshape(data, (dl, -1))

    def _bw_shape(self, data):
        dl = len(data)
        return np.reshape(data, (dl,)+self._shape)

    def fit(self, data, assume_batch=True):
        dshape = data.shape
        #print(data)
        if len(dshape) == 1 and assume_batch:
            # TODO: this is a legacy approach to cover 1-D data for some edge cases
            print("WARNING: Assuming this is 'batched'-data with dimension 1")
            self._shape = (1,)
        else:
            self._shape = dshape[1:]
        fdata = self._fw_shape(data)
        self._scaler.fit(fdata)
        self._fitted = True
        
    def transform(self, data):
        if not self._fitted:
            return self.fit_transform(data)
        else:
            fdata = self._fw_shape(data)
            sdata = self._scaler.transform(fdata)
            return self._bw_shape(sdata)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        if not self._fitted:
            raise Exception
        fdata = self._fw_shape(data)
        usdata = self._scaler.inverse_transform(fdata)
        return self._bw_shape(usdata)

@dataclass
class ScalerSpec:
    name: (list, str)
    scaler: callable
    pad : bool = False
    per_atom : bool = False
    globalfit2localscale : bool = False
    extra_scaler : callable = None
    is_graph: bool = False
    is_list: bool = False
    shape: tuple = None
    scaler_inst : WrapScaler = None
    extra_scaler_inst : WrapScaler = None

class WrappedOutput(object):
    def __init__(self, data : AtomisticMLContainer, properties : List[ScalerSpec]):
        if isinstance(properties, ScalerSpec):
            properties = [properties,]
        self._scaler_specs = properties
        self._interface_shape = None
        self._split_inverse = None
        self._has_graph = False
        self._has_list = False
        
        self._fit(data)

    def _get_data(self, indata, sspec : ScalerSpec):
        data = None
        if sspec.pad:
            data = indata.get_padded(sspec.name)
        elif sspec.per_atom:
            data, _ = indata.get_per_atom(sspec.name)
        else:
            data = indata.get(sspec.name)
        return data

    def _fit(self, data):
        for sidx in range(len(self._scaler_specs)):
            sspec = self._scaler_specs[sidx]
            indata = self._get_data(data, sspec)
            if isinstance(indata, GraphBunch):
                nodes_concat = np.concatenate(indata.nodes)
                edge_features_concat = np.concatenate(indata.edge_attrs)
                self._has_graph = True
                sspec.is_graph = True
                if sspec.scaler:
                    sspec.scaler_inst = WrapScaler(sspec.scaler)
                    sspec.scaler_inst.fit(nodes_concat, assume_batch=False)
                if sspec.extra_scaler:
                    sspec.extra_scaler_inst = WrapScaler(sspec.extra_scaler)
                    sspec.extra_scaler_inst.fit(edge_features_concat, assume_batch=False)
                sspec.shape = GraphBunchShape(node_shape=nodes_concat.shape[1:],
                                              edge_shape=edge_features_concat.shape[1:])
            elif sspec.globalfit2localscale:
                data_concat = np.concatenate(indata)
                if isinstance(indata, list):
                    self._has_list = True
                    sspec.is_list = True
                    # assume that the shape of the single samples can be inhomogeneous
                    # in the first dim, but not in following ones.
                    sspec.shape = (None,) + indata[0].shape[1:]
                else:
                    sspec.shape = indata.shape[1:]

                if sspec.scaler:
                    sspec.scaler_inst = WrapScaler(sspec.scaler)
                    sspec.scaler_inst.fit(data_concat, assume_batch=False)
            else:
                if isinstance(indata, list):
                    raise WrappingImpossible("simple wrapping not possible for list-types, "
                                             "use globalfit2localscale")
                sspec.shape = indata.shape[1:]
                if sspec.scaler:
                    sspec.scaler_inst = WrapScaler(sspec.scaler)
                    sspec.scaler_inst.fit(indata)

        if self._has_graph and len(self._scaler_specs) > 1:
            raise WrappingImpossible("Only one graph can be wrapped per output")
        elif self._has_list and len(self._scaler_specs) > 1:
            raise WrappingImpossible("Only one list-like structure can be wrapped per output")
        elif self._has_graph or self._has_list:
            self._interface_shape = self._scaler_specs[0].shape
        else:
            try:
                shapes = np.array([sspec.shape for sspec in self._scaler_specs], dtype=int)
            except ValueError:
                raise WrappingImpossible("All things should be same dimension for merging")
            relevant_shapes = shapes[:, :-1]
            if any(relevant_shapes) and not (relevant_shapes == relevant_shapes[0]).all():
                raise WrappingImpossible(f"shapes not matching up {relevant_shapes}")
            self._split_inverse = shapes[:, -1]
            self._interface_shape = tuple(relevant_shapes[0])+(sum(self._split_inverse),)


    @property
    def output_shape(self):
        return self._interface_shape

    def transform(self, indata):
        outputs = []
        for sspec in self._scaler_specs:
            data = self._get_data(indata, sspec)
            if sspec.is_graph:
                t_nodes = data.nodes
                t_edges = data.edges
                t_edge_attrs = data.edge_attrs
                t_meta = data.meta
                if sspec.scaler_inst:
                    t_nodes = [sspec.scaler_inst.transform(n) for n in data.nodes]
                if sspec.extra_scaler_inst:
                    t_edge_attrs = [sspec.extra_scaler_inst.transform(e) for e in data.edge_attrs]
                outputs.append(GraphBunch(t_nodes, t_edges, t_edge_attrs, t_meta))
            elif sspec.globalfit2localscale:
                outdata = None
                if sspec.scaler_inst:
                    if isinstance(data, list):
                        outdata = [sspec.scaler_inst.transform(d) for d in data]
                    else:
                        # TODO: reshape should be better for performance
                        outdata = np.array([sspec.scaler_inst.transform(d) for d in data])
                else:
                    outdata = data
                outputs.append(outdata)
            else:
                outdata = data
                if sspec.scaler_inst:
                    outdata = sspec.scaler_inst.transform(data)
                outputs.append(outdata)

        if self._has_graph or self._has_list:
            transformed = outputs[0]
        else:
            transformed = np.concatenate(outputs, axis=-1)
        return transformed

    def inverse_transform(self, indata):
        if len(self._scaler_specs) > 1:
            x = self._split_inverse
            datalist = [indata[..., sum(x[:i]):sum(x[:i+1])] for i in range(len(x))]
        else:
            datalist = [indata,]

        outputs = []

        for data, sspec in zip(datalist, self._scaler_specs):
            if sspec.is_graph:
                t_nodes = data.nodes
                t_edges = data.edges
                t_edge_attrs = data.edge_attrs
                t_meta = data.meta
                if sspec.scaler_inst:
                    t_nodes = [sspec.scaler_inst.inverse_transform(n) for n in data.nodes]
                if sspec.extra_scaler_inst:
                    t_edge_attrs = [sspec.extra_scaler_inst.inverse_transform(e) for e in data.edge_attrs]
                outputs.append(GraphBunch(t_nodes, t_edges, t_edge_attrs, t_meta))
            elif sspec.globalfit2localscale:
                outdata = None
                if sspec.scaler_inst:
                    if isinstance(data, list):
                        outdata = [sspec.scaler_inst.inverse_transform(d) for d in data]
                    else:
                        # TODO: reshape should be better for performance
                        outdata = np.array([sspec.scaler_inst.inverse_transform(d) for d in data])
                else:
                    outdata = data
                outputs.append(outdata)
            else:
                outdata = data
                if sspec.scaler_inst:
                    outdata = sspec.scaler_inst.inverse_transform(data)
                outputs.append(outdata)

        if len(outputs) == 1:
            transformed = outputs[0]
        else:
            transformed = outputs
        return transformed


"""
#sample validation code
p = dbloader.load_normalized(predict_dataset_tuple.path,
                             predict_dataset_tuple.loader,
                             predict_dataset_tuple.targetmapping,
                             **predict_dataset_tuple.load_opts)
i = p.get_structures_info()
fp_func = soap_func_wrapper(i["elements"], i["max_size"])
p.apply_ase_fp(fp_func)
p.apply_ase_graph(graph_func)
p.add_encoding(np.array([[[1,2], [10,4], [1,2], [10,4], [3,4]],
                         [[2,4], [2,4], [1,2], [10,4], [3,4]],
                         [[2,5], [2,5], [1,2], [10,4], [3,4]],
                         [[4,6], [4,6], [1,2], [10,4], [3,4]]]))
from data_tools import WrapScaler, WrappedOutput, ScalerSpec
s = WrappedOutput(p, [ScalerSpec("encoding", sklearn.preprocessing.MaxAbsScaler, globalfit2localscale=True),
                      ScalerSpec("fps", sklearn.preprocessing.MaxAbsScaler)])
x = s.inverse_transform(s.transform(p))
t = WrappedOutput(p, [ScalerSpec("graphs", sklearn.preprocessing.MaxAbsScaler,
                                 extra_scaler=sklearn.preprocessing.MaxAbsScaler),])
g = t.inverse_transform(t.transform(p))
"""

class FeatureWrapper(object):
    """
    class containing scalers and logic to create feature-dataframes/numpy-arrays
    on initialization, fits scalers from the scalers dict for the given crystal-dataframe-parts,
    then builds a compound dataframe.
    could be extended to e.g prebuild kernel matrices!
    """
    def __init__(self, data : AtomisticMLContainer,
                 features : dict = {}, targets : dict = {}) -> (dict, dict):
        self._single_spec = "_"
        if isinstance(features, ScalerSpec):
            features = {self._single_spec : features}
        if isinstance(targets, ScalerSpec):
            targets = {self._single_spec : targets}
        self._feature_wrappers = dict(
            (k, WrappedOutput(data, specs)) for k, specs in features.items()
        )
        self._targets_wrappers = dict(
            (k, WrappedOutput(data, specs)) for k, specs in targets.items()
        )

    def _return_single(self, *args):
        out = []
        for x in args:
            if isinstance(x, dict) and (self._single_spec in x.keys()) and \
               len(x) == 1:
                out.append(x[self._single_spec])
            else:
                out.append(x)
        return tuple(out)

    @property
    def feats_shape(self):
        return self.shapes[0]

    @property
    def targets_shape(self):
        return self.shapes[1]

    @property
    def shapes(self):
        f_s = dict(
            (k, fw.output_shape) for k, fw in self._feature_wrappers.items()
        )
        t_s = dict(
            (k, tw.output_shape) for k, tw in self._targets_wrappers.items()
        )
        return self._return_single(f_s, t_s)

    def scale(self, data, targetscale=True):
        feat_scaled = dict(
            (k, fw.transform(data)) for k,fw in self._feature_wrappers.items()
        )
        if targetscale:
            targets_scaled = dict(
                (k, tw.transform(data)) for k, tw in self._targets_wrappers.items()
            )
        else:
            targets_scaled = None
        return self._return_single(feat_scaled, targets_scaled)

    def reverse_scaling(self, feature_data=None, targets_data=None):
        feat_reversed = None
        targets_reversed = None
        if feature_data is not None:
            if isinstance(feature_data, dict):
                feat_reversed = dict(
                    (k, self._feature_wrappers[k].inverse_transform(feature_data[k]))
                    for k in feature_data.keys()
                )
            else:
                feat_reversed = \
                    self._feature_wrappers[self._single_spec].inverse_transform(feature_data)
        if targets_data is not None:
            if isinstance(targets_data, dict):
                targets_reversed = dict(
                    (k, self._targets_wrappers[k].inverse_transform(targets_data[k]))
                    for k in targets_data.keys()
                )
            else:
                targets_reversed = \
                    self._targets_wrappers[self._single_spec].inverse_transform(targets_data)
        return feat_reversed, targets_reversed


"""
featurizer = FeatureWrapper(p, features={"fps" : [
    ScalerSpec("encoding", sklearn.preprocessing.MaxAbsScaler, globalfit2localscale=True),
    ScalerSpec("fps", sklearn.preprocessing.MaxAbsScaler, pad=True)],
                                         "graphs" : [
                                             ScalerSpec("graphs", sklearn.preprocessing.MaxAbsScaler,
                                                        extra_scaler=sklearn.preprocessing.MaxAbsScaler),
                                         ]},
                            targets=ScalerSpec(["rotA", "rotB"], None))
a = featurizer.scale(p)
print(a[0]["fps"][0])
b = featurizer.reverse_scaling(*a)
print(p.fingerprints[0])
print(b[0]["fps"][1][0])
"""
