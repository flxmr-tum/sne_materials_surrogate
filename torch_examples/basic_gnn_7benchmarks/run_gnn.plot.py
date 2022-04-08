import numpy as np
import pandas as pd

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

def aggregate_single_calc(df, db_selector : dict, agg_cols):
    all_data = df
    for col in db_selector.keys():
        if col not in all_data.columns:
            continue
        nones = all_data[col].isna()
        #all_data[col][nones] = "None"
        all_data.loc[nones, col] = "None"
    mask = extract_truth(all_data, db_selector)
    sel_data = all_data[mask]
    means = sel_data[agg_cols].mean()
    stdevs = sel_data[agg_cols].std()
    if not sel_data.empty:
        return means, stdevs
    else:
        return None

def aggregate_df(full_df,
                 dbs : list,
                 db_selectors : dict,
                 db_col="db",
                 outcol="mae_test",
                 outcol_multiplier=1,
                 outcol_points=2,
                 rename_cols={},
                 raw=False,):
    selecs = list(db_selectors.keys())
    df_index = []
    for sel in selecs:
        df_index.append(sel)
    aggregated = pd.DataFrame(index=df_index, columns=dbs)
    for db in dbs:
        for sel_name, selspec in db_selectors.items():
            selspec_db = {db_col : db, **selspec}
            agg = aggregate_single_calc(full_df, selspec_db, [outcol])
            if agg is not None:
                means, stdevs = agg
                means *= outcol_multiplier
                stdevs *= outcol_multiplier
                formatted_mean = format(means[outcol], f'.{outcol_points}f')
                formatted_stdev = format(stdevs[outcol], f'.{outcol_points}f')
            else:
                print("no data for ", db, " + ", sel_name)
                continue
            if raw:
                aggregated[db][sel_name] = (means[outcol], stdevs[outcol])
            else:
                aggregated[db][sel_name] \
                    = f"{formatted_mean}±{formatted_stdev}"
    aggregated = aggregated.dropna(axis=0, how="all")
    if raw:
        aggregated = aggregated.fillna(0)
    else:
        aggregated = aggregated.fillna("-")
    
    renamed_cols = [rename_cols.get(n, n) for n in aggregated.columns]
    aggregated.columns = renamed_cols
    return aggregated


BASE_PATH="paper_final_gaps"
DBS = ["KD", "PD", "STANLEY", "CC_nonzero", "CL", "M2D", "SUTTON",]
DBS_ENER = ["KD", "SUTTON", "STANLEY"]
PROPS = ["energy", "gap"]
TTS = [0.2]
GRAPHS = {
    "xie-custom-r6" : {"gnn_arch" : "xie-custom",
                       "graph_technique" : "xie-basic-6-12"},
    "xie-custom-r10" : {"gnn_arch" : "xie-custom",
                        "graph_technique" : "xie-basic-10-12"},
    "xie-r6" : {"gnn_arch" : "xie-main",
                "graph_technique" : "xie-basic-6-12"},
    "xie-r10" : {"gnn_arch" : "xie-main",
                 "graph_technique" : "xie-basic-10-12"},
}
AGGREGATE_COLS=['mae_test', 'mae_train', 'mse_test', 'mse_train', 'r2_test', 'r2_train']

calcs_ener = {}
calcs_gap = {}
for p in PROPS:
    for t in TTS:
        for k, v in GRAPHS.items():
            if p == "gap":
                calcs_gap[f"{k}-{p}-{t}"] = {"test_portion" : t, "target" : p,
                                             **v}
            else:
                calcs_ener[f"{k}-{p}-{t}"] = {"test_portion" : t, "target" : p,
                                              **v}

DBS_RENAME_PLAIN={"PD" : "Pandey,\n 2018",
                  "KD" : "Kim,\n 2017",
                  "CL" : "Castelli,\n 2013",
                  "CC_nonzero" : "Castelli,\n 2012",
                  "SUTTON" : "Sutton,\n 2019",
                  "STANLEY" : "Stanley,\n 2019",
                  "M2D" : "Marchenko,\n 2020"}

regnn_data = pd.read_csv("./gnn_data/aggregate_gaps.csv")[['db', 'gnn_arch', 'graph_technique', 'mae_test', 'mae_train', 'mse_test', 'mse_train', 'r2_test', 'r2_train', 'target', 'test_portion']]
regnn_data["rmse_test"] = np.sqrt(regnn_data["mse_test"])

gap_maes = aggregate_df(regnn_data, DBS, calcs_gap, outcol="mae_test", outcol_multiplier=1000, outcol_points=0)
gap_rmses = aggregate_df(regnn_data, DBS, calcs_gap, outcol="rmse_test", outcol_multiplier=1000, outcol_points=0)
gap_r2s = aggregate_df(regnn_data, DBS, calcs_gap, outcol="r2_test", outcol_multiplier=1, outcol_points=2)
gap_maes.to_latex("gnn_data/gap_maes.tex", escape=False, column_format="l"+"c"*len(gap_maes.columns))
gap_rmses.to_latex("gnn_data/gap_rmses.tex", escape=False, column_format="l"+"c"*len(gap_rmses.columns))
gap_r2s.to_latex("gnn_data/gap_r2s.tex", escape=False, column_format="l"+"c"*len(gap_r2s.columns))

regnn_data_e = pd.read_csv("./gnn_data/aggregate_ener.csv")[['db', 'gnn_arch', 'graph_technique', 'mae_test', 'mae_train', 'mse_test', 'mse_train', 'r2_test', 'r2_train', 'target', 'test_portion']]
regnn_data_e["rmse_test"] = np.sqrt(regnn_data_e["mse_test"])

energy_maes = aggregate_df(regnn_data_e, DBS, calcs_ener, outcol="mae_test", outcol_multiplier=1000, outcol_points=1)
energy_rmses = aggregate_df(regnn_data_e, DBS, calcs_ener, outcol="rmse_test", outcol_multiplier=1000, outcol_points=1)
energy_r2s = aggregate_df(regnn_data_e, DBS, calcs_ener, outcol="r2_test", outcol_multiplier=1, outcol_points=2)
energy_maes.to_latex("gnn_data/energy_maes.tex", escape=False, column_format="l"+"c"*len(energy_maes.columns))
energy_rmses.to_latex("gnn_data/energy_rmses.tex", escape=False, column_format="l"+"c"*len(energy_rmses.columns))
energy_r2s.to_latex("gnn_data/energy_r2s.tex", escape=False, column_format="l"+"c"*len(energy_r2s.columns))
