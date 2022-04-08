from typing import List
from itertools import product
from functools import partial

from datetime import datetime

import numpy as np
import pandas as pd


class partial_wrap():
    """
    implement functools.partial-behavior for a function which was previouly wrapped in partial
    augment it with information about the differences to the previous wrap function!
    """
    def __init__(self, pfn : partial, *args, **kwargs):
        self._basepartial = pfn
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *newargs, debug=False, **newkwargs):
        if debug:
            print(self._args+newargs, "---", {**self._kwargs, **newkwargs})
        return self._basepartial(*(self._args+newargs), **{**self._kwargs, **newkwargs})

    @property
    def name(self):
        return self._basepartial.func.__name__

    @property
    def varied_opts(self):
        opts = {}
        for i, arg in enumerate(self._args):
            opts[i] = arg
        for key, value in self._kwargs.items():
            opts[key] = value
        return opts

    @property
    def describe(self):
        return {"name": self.name, "opts": self.varied_opts}


def timecp(msg=""):
    print("{}: {}".format(datetime.now().strftime("%H:%M:%S.%f"), msg))

def dict_product(in_dict : dict):
    """ create the product of dicts values as a series of dicts"""
    # TODO: handle empty lists as values like None!
    items = in_dict.items()
    keys = [i[0] for i in items]
    samples = [i[1] if isinstance(i[1], list) else [i[1],] for i in items]
    product_samples = product(*samples)
    dict_product = [dict(zip(keys, ps)) for ps in list(product_samples)]
    return dict_product

def partial_wrap_product(pf: partial, **kwargs):
    pw_functions = [partial_wrap(pf, **product_args) for product_args in dict_product(kwargs)]
    return pw_functions

def dicts2df(strucs : List[dict],
             uid : str,
             columns : List[str]) -> pd.DataFrame:
    strucs_transformed = [(s[uid], [s[c] for c in columns]) for s in strucs]
    labels = [t[0] for t in strucs_transformed]
    features = [t[1] for t in strucs_transformed]
    df = pd.DataFrame.from_records(data=features,
                                   index=labels,
                                   columns=columns)
    return df

def rawfp2df(fp_dict, big_frame=True) -> pd.DataFrame:
    from sne_fingerprints import rdf
    if not isinstance(fp_dict, rdf.FP_dict_raw):
        raise NotImplementedError
    if isinstance(list(list(fp_dict.values())[0].values())[0], rdf.RDF_struc):
        fp_dict_new = []
        for fp_id, fps in fp_dict.items():
            fp_dict_new.append((fp_id,
                                dict((fps_subname, rdf.data) for fps_subname, rdf in fps.items())))
        fp_dict = dict(fp_dict_new)
    else:
        pass    
    all_fps = []
    for fps in fp_dict.values():
        all_fps.extend([(fp_key, len(fps[fp_key])) for fp_key in list(fps.keys())])
    all_fps = sorted(set(all_fps))
    #print("fp2df sorting order!")
    #print(all_fps, len(all_fps))
    if len(all_fps) != len(set(a[0] for a in all_fps)) and not big_frame:
        raise Exception(msg="all fingerprints should be the same for conversion")
    # otherwise: no duplicates, just make a nice long array
    fp_length = sum(p[1] for p in all_fps)
    fps_number = len(fp_dict)
    flat_fps = np.zeros((fps_number, fp_length))
    flat_names = [None,]*fps_number
    fp_idx = 0
    fp_columns = []
    for name, fps in fp_dict.items():
        fp_insertion_idx = 0
        flat_names[fp_idx] = name
        for fp_subpart, fp_sublen in all_fps:
            fp_insertion_idx_new = fp_insertion_idx + fp_sublen
            flat_fps[fp_idx][fp_insertion_idx:fp_insertion_idx_new] = \
                fps[fp_subpart]
            fp_insertion_idx = fp_insertion_idx_new
            if fp_idx == 0:
                fp_columns.extend([(fp_subpart, i) for i in range(fp_sublen)])
        fp_idx += 1

    fp_df = pd.DataFrame(flat_fps, index=flat_names)
    fp_columns = pd.MultiIndex.from_tuples(fp_columns)
    fp_df.columns = fp_columns
    return fp_df

