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
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.optim

import torch_tools as tt
import torch_tools.pytorch_models
import torch_tools.pytorch_models.simple_deep_nns
import torch_tools.pytorch_models.simple_aes
from torch_tools.pytorch_models.simple_deep_nns import SimpleNetBN, SimpleNetBN_TG
from torch_tools.pytorch_models.simple_aes import BasicAE, Basic2LAE, BasicVAE, vae_loss, ortho_loss
from torch_tools.contrib.graph_xie import AtomCustomJSONInitializer, create_xie_graph
from torch_tools.contrib.network_xie import CrystalGraphConvNet



import data_tools as dt
from data_tools import data_utils_pandas as du
import torch_tools.dtos_workflow as dtos
import torch_tools.steps_workflow as steps_wf
import torch_tools.plots_workflow as plots_wf

from sne_ml_databases import dbloader

import dscribe.descriptors as ddesc

import json

DBDIR = os.environ.get("PEROVSKITE_DBDIR")
DB_META = {}

DB_META["SUTTON"] = (f"{DBDIR}/nmd-18/singles", "sutton", [],
                 {"gap" : "gap", "energy":"formation_energy_pa"}, {})
DB_META["CC"] = (f'{DBDIR}/cmr_dbs/cubic_perovskites.db', 'castelli_cubic', [],
                 #"heat_of_formation_all",
                 {"gap": "gllbsc_dir_gap", "energy" : "heat_of_formation_all_pa"},
                 {})
db_name = "SUTTON"

if not "data" in locals():
    db_tuple = steps_wf.DB_TUPLE(*DB_META[db_name])
    data = dbloader.load_normalized(db_tuple.path,
                                    db_tuple.loader,
                                    db_tuple.targetmapping,
                                    **db_tuple.load_opts)
    data, _, _ = data.split3(0.99, 0, shuffle_seed=100)
    data = data[:4]


graphing_func = lambda spec, na: partial(create_xie_graph,
                                  atom_featurizer=AtomCustomJSONInitializer(),
                                  radius=6, nneighbors=12,
                                  gauss_min=0, gauss_step=0.2)

import torch_tools.graph_creation
import importlib as imp
imp.reload(torch_tools.graph_creation)
import torch_tools.graph_property_encoding
imp.reload(torch_tools.graph_property_encoding)

raise Exception
graphing_func = lambda spec, na: \
    partial(torch_tools.graph_creation.radialconnection_graph,
            get_features=partial(
                torch_tools.graph_property_encoding.get_plain_atom_features,
                per_atom_features=["group_no", "period_no"],
            ),
            r_cut=4,
            max_nearest=2, weight_distance="basic",
            expand_width=1.0,
            expand_distance=False, expand_bins=2)


data.apply_ase_graph(graphing_func(1,1))

# apply a second graphing function!
graphing_func2 = lambda spec, na: \
    partial(torch_tools.graph_creation.radialconnection_graph,
            get_features=partial(
                torch_tools.graph_property_encoding.get_soap_env,
                species=all_elements,
                rcut=8, nmax=3, lmax=3, rbf="polynomial",
                ), #weighting={"w0" : 0}), # only new soaps
            r_cut=4,
            max_nearest=0, weight_distance=False,
            expand_width=1.0,
            expand_distance=False, expand_bins=10)

# also atoms and maximal number of atoms
chemical_symbols = [x.get_chemical_symbols() for x in data.structures]
all_elements = set()
for cs in chemical_symbols:
    all_elements.update(cs)
max_atoms = max(len(x) for x in data.structures)
all_elements = list(all_elements)

data.apply_ase_graph(graphing_func2(all_elements,max_atoms), target="soap-graph")


graph_featurizer = dt.FeatureWrapper(
    data,
    # look into featureWrapper/ScalerSpec. This can scale nodes and edge-attributes
    # 
    features={"graph" : dt.ScalerSpec("graphs", None),
              "sgraph" : dt.ScalerSpec("soap-graph", None)},
    targets=dt.ScalerSpec("gap", StandardScaler))

# now get the scaled and ML-ready graphs/features
g_tr, y_tr = graph_featurizer.scale(data)
import importlib as imp
imp.reload(steps_wf)

# → usually you can pass the g_tr-dict to get_torch_geo_dataloaders_from_dict (tuple, because it may create a validation dataloader
dl = steps_wf.get_torch_geo_dataloaders_from_dict(g_tr, targets=y_tr, batchsize=3)[0]
# CARE: this will assign the first graph to the pytorch_geometric.data.Data graph attributes, all other graphs and features will be assigned to their dictionary names (postfixed b _x, _edge.. for graphs!)
for b in dl:
    break
# now you can look at a batch!
# TODO: check why the non-primary graph edge-matrix gets batched!
from torch_tools.contrib.network_xie import ConvLayer, CrystalGraphConvNet
from extra_convnets import CGConv_basic, CrystalGraphConvNet_basic

net = CrystalGraphConvNet(b.x.shape[1], b.edge_attr.shape[1], n_conv=1)
net(b)
net2 = CrystalGraphConvNet_basic(b.x.shape[1], b.edge_attr.shape[1])
net2(b)
raise Exception

struc = data.structures[0]

### SOME VIZ CODE TO PLAY AROUND
from ase.build import bulk
a = 5.64
nacl = bulk('NaCl', 'rocksalt', a=a)
n, e, d = torch_tools.graph_creation.voronoi_graph(
    struc,
    get_features=partial(
        torch_tools.graph_property_encoding.get_gaussian_atom_features,
        per_atom_features=["ion_1", "cov_r"],
        max_e=100,
        bins={"ion_1": 5, "cov_r" : 10}
    ),
    weight_distance="basic",
    expand_distance=False,
    max_distance=5,
    weight_area="1/r2",
    expand_area="ohe"
)
#print(e[:, :6])
#print(d[:6])

from scipy.spatial import Voronoi



import matplotlib as mpl
from matplotlib.gridspec import GridSpec
mpl.use("tkAgg")
import matplotlib.pyplot as plt

#sel = 1
#csurf_x, csurf_y = conn_surfaces_xy[sel][:, 0].T, conn_surfaces_xy[sel][:, 1].T
#plt.plot(csurf_x, csurf_y)
#plt.show()

from scipy.spatial import Voronoi
rep_nacl = nacl.repeat((3,3,3))
nacl.positions += (nacl.cell[0] + nacl.cell[1] + nacl.cell[2])


"""
elem_embedding = json.load(open("./atom_init_xie.json"))
species_features = torch_tools.graph_property_encoding._get_tabulated_ohe(per_atom_features=["group_no", "period_no", "en"],
                                                                           max_e=100,
                                                                           bins={"group_no" : 18 , "period_no" : 9,
                                                                                 "en" : 10 })[:10]

from sne_ml_databases.property_tables import TABLES as atomic_data
from sne_ml_databases.property_tables import TABLES_NUMERIC as atomic_data_numeric
species_features_xie = [s[1:38] for s in list(elem_embedding.values())[:10]]
atomtypes = list(atomic_data["en"].keys())[:10]

grid = GridSpec(len(atomtypes), 1)
fig = plt.figure()
for idx, atom, nodefeature_xie, nodefeature in zip(range(len(atomtypes)), atomtypes,
                                                   species_features_xie, species_features):
    ax = fig.add_subplot(grid[idx])
    ax.plot(nodefeature_xie, label=atom)
    ax.plot(nodefeature)
    ax.legend()
plt.show()
"""

print("")
