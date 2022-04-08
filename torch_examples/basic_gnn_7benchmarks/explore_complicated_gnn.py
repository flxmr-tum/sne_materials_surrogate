from functools import partial
import importlib as imp

import os
import numpy as np
import pandas as pd

import sklearn
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.decomposition import PCA
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


### MIGHT BE HELPFUL FOR USAGE IN JUPYTER-notebooks ###

OUTNAME = "perpetual_testing"
SEED=1868
EPOCHS=100
OUTDIR = f"gnn_exp/{OUTNAME}/{SEED}/"
os.makedirs(OUTDIR, exist_ok=True)

DB_NAME="SUTTON"
DB_TARGET="gap"

db_tuple = steps_wf.DB_TUPLE(*DB_META[DB_NAME])

if "data" not in locals().keys():
    data = dbloader.load_normalized(db_tuple.path,
                                    db_tuple.loader,
                                    db_tuple.targetmapping,
                                    **db_tuple.load_opts)
    #data = data[:100]

graph_func = lambda spec, na: partial(torch_tools.graph_creation.radialconnection_graph,
                                      get_features=[
                                          partial(
                                              torch_tools.graph_property_encoding.get_ohe_atom_features,
                                              per_atom_features=["group_no", "period_no", "en", "cov_r", "ion_1", "block", "atomic_vol"],
                                              max_e=100,
                                              bins={"group_no" : 18 , "period_no" : 9,
                                                    "en" : 10, "cov_r" :10, "ion_1":10, "ea":10,
                                                    "block": 4, "atomic_vol" : 10}
                                          ),
                                          partial(
                                              torch_tools.graph_property_encoding.get_soap_env,
                                              species=all_elements,
                                              rcut=8, nmax=2, lmax=3, rbf="polynomial",
                                          )
                                      ],
                                      r_cut=6, max_nearest=12, weight_distance="basic",
                                      expand_distance="gauss", expand_width=0.2, expand_bins=60)

chemical_symbols = [x.get_chemical_symbols() for x in data.structures]
all_elements = set()
for cs in chemical_symbols:
    all_elements.update(cs)
max_atoms = max(len(x) for x in data.structures)
all_elements = list(all_elements)

data.apply_ase_graph(graph_func(all_elements, max_atoms), ncpus=8)

ttv = data.split3(0.2, 0, shuffle_seed=SEED)

graph_featurizer = dt.FeatureWrapper(ttv.train,
                                     features={"graph" : dt.ScalerSpec("graphs", MinMaxScaler)},
                                     targets=dt.ScalerSpec(DB_TARGET, StandardScaler))

g_tr, y_tr = graph_featurizer.scale(ttv.train)
g_te, y_te = graph_featurizer.scale(ttv.test)

node_shape = graph_featurizer.feats_shape["graph"].node_shape[0]
edge_shape = graph_featurizer.feats_shape["graph"].edge_shape[0]

gnn_model_init = lambda: CrystalGraphConvNet_pyggit(
    node_shape, edge_shape,
    atom_fea_len=64, h_fea_len=32, n_conv=4, n_h=1)

gnn_model = steps_wf.train_predictor(g_tr, y_tr, None,
                                     constructor=gnn_model_init,
                                     run_opts={"epochs" : EPOCHS,
                                               "batch_size" : 32,
                                               "loss_fn" : lambda: torch.nn.MSELoss(),
                                               "optim": partial(torch.optim.SGD, lr=1e-2, momentum=0.9),
                                               "validate": 0.2,
                                               "lr_sched" : partial(torch.optim.lr_scheduler.ReduceLROnPlateau)})
ye_tr, ye_te = steps_wf.predict(gnn_model, g_tr, g_te)

ye_tr = ye_tr.reshape(len(ye_tr), 1)
ye_te = ye_te.reshape(len(ye_te), 1)
_, ye_tr = graph_featurizer.reverse_scaling(targets_data=ye_tr)
_, ye_te = graph_featurizer.reverse_scaling(targets_data=ye_te)
_, y_tr = graph_featurizer.reverse_scaling(targets_data=y_tr)
_, y_te = graph_featurizer.reverse_scaling(targets_data=y_te)

    
plots_wf.plot_prediction_error((y_te[:,0], ye_te[:,0]),
                               train=(y_tr[:,0], ye_tr[:,0]),
                               y_label=DB_TARGET, title=f"gnn-test!",
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
