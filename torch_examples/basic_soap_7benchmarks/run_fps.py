from functools import partial
import importlib as imp

import os
import numpy as np
import pandas as pd

import sklearn
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA, SparsePCA, KernelPCA
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.optim

import torch_tools as tt
import torch_tools.graph_creation
import torch_tools.graph_property_encoding
import torch_tools.pytorch_models
import torch_tools.pytorch_models.simple_deep_nns
import torch_tools.pytorch_models.simple_aes
from torch_tools.pytorch_models.simple_deep_nns import SimpleNetBN, SimpleNetBN_TG
from torch_tools.pytorch_models.simple_aes import BasicAE, Basic2LAE, BasicVAE, vae_loss, ortho_loss
from torch_tools.pytorch_models.simple_graph_predictor import CrystalGraphConvNet_pyggit
from torch_tools.contrib.graph_xie import AtomCustomJSONInitializer, create_xie_graph
from torch_tools.contrib.network_xie import CrystalGraphConvNet

import data_tools as dt
from data_tools import data_utils_pandas as du
import torch_tools.dtos_workflow as dtos
import torch_tools.steps_workflow as steps_wf
import torch_tools.plots_workflow as plots_wf

from sne_ml_databases import dbloader

import dscribe.descriptors as ddesc

import fingerprinting_shims as fps

from copy import deepcopy


DBDIR = os.environ.get("PEROVSKITE_DBDIR")
DB_META = {}
DB_META["CL"] = (f'{DBDIR}/cmr_dbs/low_symmetry_perovskites.db', 'castelli_lowsym', [], {"gap" : "gllbsc_dir_gap", "energy" : "energy_pa"}, {})
DB_META["PD"] = (f'{DBDIR}/cmr_dbs/a2bcx4.db', 'pandey_A2BCX4', [], {"gap": "gllbsc_dir_gap", "energy": "energy_pa"}, {})
DB_META["CC"] = (f'{DBDIR}/cmr_dbs/cubic_perovskites.db', 'castelli_cubic', [],
                 #"heat_of_formation_all",
                 {"gap": "gllbsc_dir_gap", "energy" : "heat_of_formation_all_pa"},
                 {})
DB_META["CC_nonzero"] = (f'{DBDIR}/cmr_dbs/cubic_perovskites.db', 'castelli_cubic', [],
                         {"gap" : "gllbsc_dir_gap", "energy" : "heat_of_formation_all_pa"},
                         {"nonzero_gap": 0.1},
)
DB_META["CC_nonzero_culled"] = (f'{DBDIR}/cmr_dbs/cubic_perovskites.db', 'castelli_cubic', [],
                                {"gap" : "gllbsc_dir_gap", "energy" : "heat_of_formation_all_pa"},
                                {"nonzero_gap": 0.1,
                                 "cull_struc": ["S", "O", "N", "F"],
                                 "replace_pos" : 1},
)
DB_META["M2D"] = (f'{DBDIR}/marchenko_2D', 'marchenko_2D', [],
                  {"gap" : "bandgap_th"},
                  {},
)
DB_META["KD"] = (f"../../../perovskite-data/HOIP-cifs", "kimdb", [
    'r_center', 'r_anion', 'r_cation', 'ea_anion',
    'ea_crystal_cation', 'en_anion', 'en_crystal_cation'],
    {"gap" : "gap_hse", "energy": "E_a"} , {"simplified" : False})
DB_META["SUTTON"] = (f"{DBDIR}/nmd-18/singles", "sutton", [],
                 {"gap" : "gap", "energy":"formation_energy_pa"}, {})
DB_META["STANLEY"] = (f"{DBDIR}/jared/2018-Q3-augmentedDataSet.json", "stanley", [],
                      {"gap" : "opt_gap", "energy" : "formation_energy_pa"}, {})
DB_META["QM9"] = (f"{DBDIR}/qm9/dsgdb9nsd.xyz", "qm9", [],
                  {"gap" : "gap", "energy" : "U0"}, {})
DB_META["QM9_s"] = (f"{DBDIR}/qm9/test.xyz", "qm9", [],
                  {"gap" : "gap", "energy" : "U0"}, {})
DB_META["MP2018"] = (f"{DBDIR}/mp_2018/mp_all.json", "mp2018", [],
                     {"gap" : "gap pbe", "energy" : "e_form", "hull" : "e_hull"}, {"minimal_gap" : 0})
DB_META["MP2018_XieGap"] = (f"{DBDIR}/mp_2018/mp_all.json", "mp2018", [],
                            {"gap" : "gap pbe", "energy" : "e_form", "hull" : "e_hull"},
                            {"minimal_gap" : 0,
                             "subselect_list" : f"{DBDIR}/mp2018/xie-27430.csv"})
DB_META["SCHRAMM"] = (f"{DBDIR}/schramm_2d", "schramm", [],
                      {"gap" : "gap"}, {"simplified" : True})
DB_META["TETRA"] = (f"{DBDIR}/transfer_integrals/tetracene_dimers.agg.cif", "sne_dimer", [],
                    {"ex_int" : "ti"}, {})

db_configs ={
    "db" : [
        ("STANLEY", "gap"),
        ("SUTTON", "gap"),
        ("PD", "gap"),
        ("KD", "gap"),
        ("CL", "gap"),
        ("CC_nonzero", "gap"), ("M2D", "gap"),
        #("KD", "energy"), ("SUTTON", "energy"), ("STANLEY", "energy")
    ],
    "fingerprint" : [
        ("SOAP-nomad", lambda s, species=["H"]: partial(fps.soap_periodic_mean,
                                                        species=species,
                                                        rcut=10, sigma=0.5,
                                                        rbasis=4, sphericals=4)(s)),
        ("P²DDF-fine", lambda s, species=None: fps.p2ddf_fine_paper(s)),
        ("MBTR-k2-rdf", lambda s, species=None: partial(fps.mbtr_simple_rdf,
                                                        species=species,
                                                        sigma=1.0, dist=16,
                                                        grid_n=round(16*1))(s)),
        #("SOAP", partial(fps.soap_fp, rcut=10, rbasis=3, sphericals=4)),
        #("MBTR-k2", partial(fps.mbtr2d_fp, dist=10, grid_n=15)),
        #("PDDF-default", partial(fps.pddf_fp_multi, radius=10, norm="3d",
        #                         binsize=0.3, gaussian=0.9)),
        #("PDDF-small", partial(fps.pddf_fp_multi, radius=10, norm="3d",
        #                       binsize=0.3, gaussian=0.9,
        #                 props=["n",])),
        #("PDDF-extra", partial(fps.pddf_fp_multi, radius=10, norm="3d",
        #                      binsize=0.3, gaussian=0.9,
        #                      props=["n", "ion_2", "ion_1", "val_p", "val_s"])),
    ],
}



configs = {
    "scaler" : [
        #("none", None), # use for SOAP
        #("MaxAbs", MaxAbsScaler),
        ("MinMaxScaler", MaxAbsScaler), # use for SOAP 
        ("StandardScaler", StandardScaler) # Use for PDDF, MBTR
    ],
    "data_reducer" : [
        ("none", None),
        #("varsel", partial(VarianceThreshold, 0.01)),
        #("pca", partial(PCA, n_components=40)),
    ],
    "ml_model" : [
        # used for SOAP in the paper
        #("KRR_nonlinear",
        # partial(sklearn.model_selection.GridSearchCV,
        #         estimator=KernelRidge(),
        #         param_grid={"kernel" : ["linear"],
        #                     "alpha" : np.logspace(-6, 1, num=8)},
        #                     cv=5,
        #         )),
        # used for all others in the paper!
        ("KRR_rbf",
         partial(sklearn.model_selection.GridSearchCV,
                 estimator=KernelRidge(),
                 param_grid={"kernel" : ["rbf"],
                             "gamma" : np.logspace(-6, 1, num=8),
                             "alpha" : np.logspace(-6, 1, num=8)},
                             cv=5,
                 )),
    ],
    "test_split" : [0.2,]
}

def setup_data(dbc):
    db_name = dbc["db"][0]
    db_target = dbc["db"][1]
    fp_name = dbc["fingerprint"][0]

    # load data and fp
    db_tuple = steps_wf.DB_TUPLE(*DB_META[db_name])
    data = dbloader.load_normalized(db_tuple.path,
                                    db_tuple.loader,
                                    db_tuple.targetmapping,
                                    **db_tuple.load_opts)

    chemical_symbols = [x.get_chemical_symbols() for x in data.structures]
    all_elements = set()
    for cs in chemical_symbols:
        all_elements.update(cs)
    data.apply_ase_fp(lambda s: dbc["fingerprint"][1](s, species=all_elements), ncpus=12)
    return data

def run_single_model(c, data, seed):
    db_name = c["db"][0]
    db_target = c["db"][1]
    fp_name = c["fingerprint"][0]
    data_reducer_name = c["data_reducer"][0]
    ml_name = c["ml_model"][0] 
    test_split = c["test_split"]

    outname = f"{db_name}-{db_target}-{fp_name}-{data_reducer_name}-{ml_name}-{test_split}"
    OUTDIR = f"fps_output/{outname}/{seed}/"
    print(f"running in {OUTDIR}")
    os.makedirs(OUTDIR, exist_ok=True)


    # split into train/test/val(=0) and scale
    ttv = data.split3(0.2, 0, shuffle_seed=seed)
    globals().update(**locals())
    featurizer = dt.FeatureWrapper(ttv.train,
                                   features=dt.ScalerSpec("fps", c["scaler"][1]),
                                   targets=dt.ScalerSpec(db_target, None))

    fp_tr, y_tr = featurizer.scale(ttv.train)
    fp_te, y_te = featurizer.scale(ttv.test)

    datrd = c["data_reducer"][1]
    if datrd is not None:
        datrd = datrd()
        fp_tr = datrd.fit_transform(fp_tr)
        fp_te = datrd.transform(fp_te)
        print(data_reducer_name, fp_tr.shape)       
        

    # Q: why is this returning a dictionary?!
    ml_model = steps_wf.train_predictor(fp_tr, y_tr,
                                         featurizer.feats_shape,
                                         constructor=c["ml_model"][1])[0]
    y_tr_pred, y_te_pred = steps_wf.predict(ml_model,
                                            fp_tr,
                                            fp_te)

    _, ye_tr = featurizer.reverse_scaling(targets_data=y_tr_pred)
    _, ye_te = featurizer.reverse_scaling(targets_data=y_te_pred)
    _, y_tr = featurizer.reverse_scaling(targets_data=y_tr)
    _, y_te = featurizer.reverse_scaling(targets_data=y_te)

    plots_wf.plot_prediction_error((y_te[:,0], ye_te[:,0]),
                                   train=(y_tr[:,0], ye_tr[:,0]),
                                   y_label=db_target, title=f"gnn for {outname}, seed={seed}",
                                   use_fig=f"{OUTDIR}/parity_plot.pdf")

    y_tr = y_tr[:,0]
    y_te = y_te[:,0]
    ye_tr = ye_tr[:,0]
    ye_te = ye_te[:,0]

    mae_train = mean_absolute_error(y_tr, ye_tr)
    mse_train = mean_squared_error(y_tr, ye_tr)
    r2_train = r2_score(y_tr, ye_tr)
    
    mae_test = mean_absolute_error(y_te, ye_te)
    mse_test = mean_squared_error(y_te, ye_te)
    r2_test = r2_score(y_te, ye_te)
    
    outdf = None
    outcsv_path = "fps_output/aggregate.csv"
    if os.path.exists(outcsv_path):
        outdf = pd.read_csv(outcsv_path)
    else:
        outdf = pd.DataFrame()

    outdf = outdf.append(
        {
            "db" : db_name,
            "target" : db_target,
            "seed" : seed,
            "test_portion": test_split,
            "fp_name" : fp_name,
            "scaler" : c["scaler"][0],
            "data_reducer" : data_reducer_name,
            "ml_name" : ml_name,
            "mae_train" : mae_train,
            "mse_train" : mse_train,
            "r2_train" : r2_train,
            "mae_test" : mae_test,
            "mse_test" : mse_test,
            "r2_test" : r2_test,
        },
        ignore_index=True
    )
         
    outdf.to_csv(outcsv_path, index=False)

    
SPLIT_SEEDS = [1868, 2020, 1993, 808, 1914, 1681, 616, 8169, 114, 1210]

for dbc in dt.dict_product(db_configs):
    # only apply fp once for all data!
    data_with_fp = setup_data(dbc)
    for c in dt.dict_product(configs):
        c = {**c, **dbc}
        for seed in SPLIT_SEEDS:
            run_single_model(c, deepcopy(data_with_fp), seed)

