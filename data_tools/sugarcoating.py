import itertools
import random
from collections import namedtuple as nt

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def dict_product(d, keep=0.5):
    #https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    keys = d.keys()
    vals = d.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


class DfOHE():
    def __init__(self, df, repl_int=False):
        self.column2ohe = dict((c, None) for c in sorted(list(df.columns)))
        for c in self.column2ohe.keys():
            if df[c].dtypes == np.dtype('O'):
                ohe = OneHotEncoder(sparse=False)
                ohe.fit(df[c].to_numpy().reshape(-1,1))
                self.column2ohe[c] = ohe
            if repl_int:
                raise NotImplementedError("Categorize Integer columns")
            pass

    def apply(self, df):
        columns = []
        for c, ohe in self.column2ohe.items():
            if ohe is not None:
                enc = ohe.transform(df[c].to_numpy().reshape(-1,1))
                tot_len = enc.shape[-1]
                columns.append(
                    pd.DataFrame(enc, index=df[c].index, columns=[f"{c}-{i}" for i in range(tot_len)]))
            else:
                columns.append(df[c])
        return pd.concat(columns, axis=1)
                

describe_config = nt("desc_config", ("desc", "config"))
