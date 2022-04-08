import numpy as np
import pandas as pd
import re

import os
from collections import defaultdict



AGGREGATE_COLS=["r2_train", "mae_train", "mse_train", "r2_test", "mae_test", "mse_test"]

def extract_truth(df, db_selector):
    outbool = None
    for k, v in db_selector.items():
        if k not in df.columns:
            print(f"couldn't find {k} - IGNORING!!")
            continue
        if outbool is None:
            outbool = (df[k] == v)
        else:
            outbool = np.logical_and(outbool, df[k] == v)
    return outbool

def aggregate_single_calc(fp, db_selector):
    all_data = pd.read_csv(fp, header=0)
    for col in db_selector.keys():
        if col not in all_data.columns:
            continue
        nones = all_data[col].isna()
        #all_data[col][nones] = "None"
        all_data.loc[nones, col] = "None"
    mask = extract_truth(all_data, db_selector)
    sel_data = all_data[mask]
    means = sel_data[AGGREGATE_COLS].mean()
    stdevs = sel_data[AGGREGATE_COLS].std()
    if not sel_data.empty:
        return means, stdevs
    else:
        return None

def aggregate_data_basic(base_path, outcol="mae_train",
                         outcol_multiplier=1000,
                         outcol_points=2,
                         raw=False):
    aggregate_cols_1 = ["db"]
    aggregate_cols_2 = ["target", "test_portion",
                        "fp_name", "scaler",
                        "data_reducer",
                        "ml_name"]
    df = pd.read_csv(base_path)
    x_cols = set(df[aggregate_cols_1[0]])
    y_cols = set(
        [tuple(cc) for cc in \
         df[aggregate_cols_2].to_numpy()]
    )
    calc_combs = [tuple(cc) for cc in \
                  df[aggregate_cols_1\
                     +aggregate_cols_2].to_numpy()]
    calc_combs = set(calc_combs)
    
    aggregated = pd.DataFrame(index=y_cols, columns=x_cols)

    calc_combs = [dict(zip(aggregate_cols_1+aggregate_cols_2, cc)) for cc in calc_combs]
    for sel in calc_combs:
        agg = aggregate_single_calc(base_path, sel)
        if agg is not None:
            means, stdevs = agg
            means *= outcol_multiplier
            stdevs *= outcol_multiplier
            formatted_mean = format(means[outcol], f'.{outcol_points}f')
            formatted_stdev = format(stdevs[outcol], f'.{outcol_points}f')
            # dictionaries are ordered with novel python
            selkeys = list(sel.values())
            col = selkeys[:len(aggregate_cols_1)][0]
            row = tuple(selkeys[len(aggregate_cols_1):])
            if raw:
                aggregated[col][row] = (means[outcol], stdevs[outcol])
            else:
                aggregated[col][row] \
                    = f"{formatted_mean}±{formatted_stdev}"
    return aggregated


# just calculate the average and std across our data!
#df = pd.read_csv("./fps_output/aggregate_old.csv")
df = aggregate_data_basic("./fps_output/aggregate.csv", outcol="mae_test")
print(df)
