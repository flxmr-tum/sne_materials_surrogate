from typing import List,Tuple,Dict
import functools
from functools import partial

import data_tools as dt

from sne_ml_databases import dbloader

import sklearn
import sklearn.base
import sklearn.model_selection

import numpy as np
import pandas as pd

import torch
import torch.nn
import torch.optim
import torch.utils.data as t_data
import torch.nn.utils
# FIXME: https://github.com/rusty1s/pytorch_geometric/issues/1781
import torch_geometric.data as tg_data
import torch_geometric.loader as tg_loader
import torch_geometric.nn as tg_nn

from . import ema_torch

from collections import namedtuple as ntuple
from typing import List

DB_TUPLE = ntuple('DB_TUPLE', ['path', 'loader', 'features', 'targetmapping', 'load_opts'])


CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    CUDA_DEV = torch.device("cuda:0")
else:
    CUDA_DEV = None

def load_ttv_data(predict_dataset : DB_TUPLE,
                  datasplit=(0.2,0.2),
                  encoder_dataset : DB_TUPLE = None,
                  encode_split : float = 0.1,
                  augmentation_function : callable = None,
                  graph_func_wrapper : callable = None,
                  fp_func_wrapper : callable = None, # wrapper returning the fingeprint-function. needs to get all species and max number of atoms.
                  shuffle_seed=14
                  ) -> (dt.TTV_split, dt.TTV_split):
    meta = {}
    predict_dataset_tuple = DB_TUPLE(*predict_dataset)
    predict_data = dbloader.load_normalized(predict_dataset_tuple.path,
                                            predict_dataset_tuple.loader,
                                            predict_dataset_tuple.targetmapping,
                                            **predict_dataset_tuple.load_opts)
    encoder_data = None
    if encoder_dataset:
        encoder_dataset_tuple = DB_TUPLE(*encoder_dataset)
        encoder_data = dbloader.load_normalized(encoder_dataset_tuple.path,
                                                encoder_dataset_tuple.loader,
                                                encoder_dataset_tuple.targetmapping,
                                                **encoder_dataset_tuple.load_opts)
        if augmentation_function:
            print("Augmentation not supported by this loader!")

    chemical_symbols = [
        x.get_chemical_symbols() for x in predict_data.structures
    ]
    max_data_length = max([len(x) for x in predict_data.structures])
    if encoder_data:
        chemical_symbols += [
            x.get_chemical_symbols() for x in encoder_data.structures
        ]
        max_data_length = max(max_data_length, max([len(x) for x in encoder_data.structures]))
    all_elements = set()
    for cs in chemical_symbols:
        all_elements.update(cs)
    meta["agg_props"] = {"elements" : all_elements, "max_atoms" : max_data_length}

    if graph_func_wrapper:
        graph_func = graph_func_wrapper(all_elements, max_data_length)
        predict_data.apply_ase_graph(graph_func, ncpus=8)
        if encoder_data:
            encoder_data.apply_ase_graph(graph_func, ncpus=8)

    if fp_func_wrapper:
        fp_func = fp_func_wrapper(all_elements, max_data_length)
        predict_data.apply_ase_fp(fp_func, ncpus=8)
        if encoder_data:
            encoder_data.apply_ase_fp(fp_func, ncpus=8)


    if isinstance(datasplit, float):
        predict_ttv = predict_data.split3(0, datasplit, shuffle_seed=shuffle_seed)
    else:
        predict_ttv = predict_data.split3(datasplit[1], datasplit[0], shuffle_seed=shuffle_seed)

    if encoder_data:
        encode_ttv = encoder_data.split3(0, encode_split, shuffle_seed=shuffle_seed+31)
    else:
        if predict_ttv.test is None:
            encode_ttv = predict_ttv.train.split3(0, encode_split, shuffle_seed=shuffle_seed+31)
        else:
            encode_ttv = dt.TTV_split(train=predict_ttv.train, test=None, val=predict_ttv.val)

    return predict_ttv, encode_ttv, meta

# special handling for pytorch_geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#frequently-asked-questions
def get_torch_dataset(features, targets=None, use_dev=None):
    if isinstance(targets, pd.DataFrame):
        targets = targets.to_numpy()
      
    torched_targets = []
    torch_featuremap = None
    torched_features = []
    if targets is not None:
        torched_targets = [torch.Tensor(targets).to(use_dev)]

    if isinstance(features, np.ndarray):
        torched_features = [torch.Tensor(features).to(use_dev)]
    elif isinstance(features, dict):
        torch_featuremap = list(features.keys())
        torched_features = [torch.Tensor(f).to(use_dev) for f in torch_featuremap]
    return t_data.TensorDataset(*torched_features+torched_targets), torch_featuremap

def get_torch_dataloaders_from_dataset(torch_data, validation_frac, batchsize=32):
    if validation_frac:
        val_len = np.ceil(validation_frac*len(torch_data))
        print("Validation length:", val_len)
        train_data, val_data = t_data.random_split(torch_data, [int(len(torch_data)-val_len), int(val_len)],
                                                   generator=torch.Generator().manual_seed(42))
        train_dataloader = t_data.DataLoader(train_data, batch_size=batchsize)
        val_dataloader = t_data.DataLoader(val_data, batch_size=batchsize)
    else:
        train_dataloader = t_data.DataLoader(torch_data, batch_size=batchsize)
        val_dataloader = None
    return train_dataloader, val_dataloader

def get_torch_geo_dataloaders_from_dict(features, targets=None, validation_frac=None, batchsize=32,
                                        list_loader = False, use_dev=None):
    # takes a dictionary of features, which can be graphs and prepares torch dataloaders
    # for multiple graphs, the graph in dictionary "graph" will be put as the default graph
    # all others will be prefixed with their dictionary key!
    loader_class = tg_loader.DataLoader
    if list_loader:
        loader_class = tg_loader.DataListLoader
    remaining_features = []
    graph_features = {}
    for k, v in list(features.items()):
        if isinstance(v, dt.GraphBunch):
            if k == "graph":
                x_name = "x"
                edge_index_name = "edge_index"
                edge_attr_name = "edge_attr"
                meta_pref = ""
            else:
                x_name = f"{k}_x"
                edge_index_name = f"{k}_edge_index"
                edge_attr_name = f"{k}_edge_attr"
                meta_pref = f"{k}_"
            graph_np = features[k]
            graph_features[x_name] = [torch.Tensor(gn).to(torch.float).to(use_dev) for gn in graph_np.nodes]
            graph_features[edge_index_name] = [torch.Tensor(ge).to(torch.long).to(use_dev) for ge in graph_np.edges]
            graph_features[edge_attr_name] = [torch.Tensor(geat).to(torch.float).to(use_dev)
                                              for geat in graph_np.edge_attrs]
            if graph_np.meta:
                for k in graph_np.meta[0].keys():
                    graph_features[f"{meta_pref}{k}"] = [
                        torch.Tensor(meta[k]).to(torch.float).to(use_dev)
                        for meta in graph_np.meta
                    ]
        else:
            remaining_features.append(k)
    data_list = []
    features = {**dict((k, features[k]) for k in remaining_features),
                **graph_features}
    features_names = list(features.keys())
    if targets is None:
        base_target = torch.zeros((len(features[features_names[0]]), 1))
        targets_dict = { "pg_target" : base_target }
    elif isinstance(targets, dict):
        base_target = torch.zeros((len(features[features_names[0]]), 1))
        targets_dict = { "pg_target" : base_target,
                         **dict(
                             (f"pg_target_{k}", v) for k, v in targets.items()
                         )}
    else:
        targets_dict = { "pg_target" : targets }
    targets_names = list(targets_dict.keys())
    tlen = len(targets_names)
    for _ in zip(*targets_dict.values(), *features.values()):
        try:
            data_list.append(tg_data.Data(#pg_target=torch.tensor(_[0]),
                **dict([(k, torch.tensor(v, dtype=torch.float32) if not isinstance(v, torch.Tensor) else v)
                        for k, v in zip(targets_names, _[:tlen])]),
                **dict([(k, torch.tensor(v) if not isinstance(v, torch.Tensor) else v)
                        for k, v in zip(features_names, _[tlen:])])
            ).to(use_dev))
        except TypeError as e:
            #return _, data_list
            raise e
    if validation_frac:
        val_len = int(np.ceil(validation_frac*len(data_list)))
        train_dataloader, val_dataloader = [
            loader_class(s, batch_size=batchsize)
            for s in sklearn.model_selection.train_test_split(data_list, test_size=val_len)]
    else:
        train_dataloader = loader_class(data_list, batch_size=batchsize)
        val_dataloader = None
    return train_dataloader, val_dataloader

def create_torch_dataloaders(features, targets=None, validation_frac=None, batchsize=32, model=None, use_dev=None):
    torch_has_tg = False
    if model:
        torch_has_tg = any(map(lambda s: s.startswith("pytorch_geometric"), [c.__module__ for c in model.modules()]))
        torch_has_tg_parallel = isinstance(model, tg_nn.DataParallel)
    if torch_has_tg_parallel or hasattr(model, "_USE_TORCH_GEOMETRIC_LISTLOADER"):
        train_dataloader, val_dataloader = get_torch_geo_dataloaders_from_dict(
            features, targets, validation_frac=validation_frac, batchsize=batchsize,
            list_loader=True,
            use_dev=use_dev)
        torch_featuremap = None
    elif torch_has_tg or hasattr(model, "_USE_TORCH_GEOMETRIC"):
        train_dataloader, val_dataloader = get_torch_geo_dataloaders_from_dict(
            features, targets, validation_frac=validation_frac, batchsize=batchsize, use_dev=use_dev)
        torch_featuremap = None
    else:
        torch_data, torch_featuremap = get_torch_dataset(features, targets, use_dev=use_dev)
        train_dataloader, val_dataloader = get_torch_dataloaders_from_dataset(
            torch_data, validation_frac, batchsize=batchsize)
    return train_dataloader, val_dataloader, torch_featuremap

def get_train_batch(batch_data, model_instance, torch_featuremap=None):
    PYGT = False
    if isinstance(batch_data, list) and isinstance(batch_data[0], tg_data.Data):
        PYGT = True
        TG_LIST = True
        b_features = batch_data
        b_targets = torch.cat([d["pg_target"] for d in batch_data]).reshape(len(batch_data), -1)
        b_pred = model_instance(batch_data)
    elif isinstance(batch_data, tg_data.Batch):
        PYGT = True
        TG_LIST = False
        b_features = batch_data
        b_targets = batch_data["pg_target"]
        b_pred = model_instance(batch_data)
    elif torch_featuremap:
        b_features = batch_data[:len(torch_featuremap)]
        b_targets = batch_data[len(torch_featuremap):]
        b_pred = model_instance(**dict(zip(torch_featuremap, b_features)))
    else:
        b_features = batch_data[0]
        b_targets = batch_data[1]
        b_pred = model_instance(b_features)
    # TODO: check for consequences
    if isinstance(b_pred, dict) and PYGT:
        b_targets = {}
        for k in b_pred.keys():
            data = batch_data[f"pg_target_{k}"]
            if data.dim() == 1:
                b_targets[k] = data.reshape(len(data), 1)
            else:
                b_targets[k] = data
    elif isinstance(b_targets, torch.Tensor):
        if isinstance(b_pred, tuple):
            if b_targets.shape != b_pred[0].shape:
                b_pred = tuple(b_pred[0].reshape(b_targets.shape), *b_pred[1:])
        else:
            if b_targets.shape != b_pred.shape:
                b_pred = b_pred.reshape(b_targets.shape)
    return b_features, b_targets, b_pred

def get_pred_batch(batch_data, model_instance, torch_featuremap=None):
    if isinstance(batch_data, list) and isinstance(batch_data[0], tg_data.Data):
        b_features = batch_data
        b_pred = model_instance(batch_data)
    elif isinstance(batch_data, tg_data.Batch):
        b_features = batch_data
        b_pred = model_instance(batch_data)
    elif torch_featuremap:
        b_features = batch_data[:len(torch_featuremap)]
        b_pred = model_instance(**dict(zip(torch_featuremap, b_features)))
    else:
        b_features = batch_data[0]
        b_pred = model_instance(b_features)
    return b_features, b_pred

def get_pred_batch_ae(batch_data, model_instance, torch_featuremap=None):
    # TODO: the AE should also be feedable with a torch-geometric Batch-object. somehow.
    if torch_featuremap:
        b_features = batch_data[:len(torch_featuremap)]
        b_pred = model_instance.encode(**dict(zip(torch_featuremap, b_features)))
    else:
        b_features = batch_data[0]
        b_pred = model_instance.encode(b_features)
    return b_features, b_pred
    

#- Autoencoder/PCA-stuff: TRAINING
def train_encoder(features, feature_dims, constructor=None, constructor_opts={}, run_opts={}):
    try:
        predictor = constructor(feature_dims=feature_dims, **constructor_opts)
    except TypeError as e:
        print(e)
        print("Couldn't inject feature_dims, trying without...")
        predictor = constructor(**constructor_opts)

    if isinstance(predictor, sklearn.base.BaseEstimator):
        _train_sklearn_reducer(features, predictor, run_opts)
    elif isinstance(predictor, torch.nn.Module):
        _train_torch_autoencoder(features, predictor,run_opts)
    return predictor


def _train_torch_autoencoder(features, model_instance, run_opts, use_dev=CUDA_DEV):
    print(model_instance)
    loss_fn = run_opts.get("loss_fn", partial(torch.nn.MSELoss, reduction="sum"))()
    optimizer = run_opts.get("optim", torch.optim.Adam)
    epochs = run_opts.get("epochs", 100)
    batchsize = run_opts.get("batch_size", 32)
    validation_frac = run_opts.get("validate", None)
    lr_scheduler = run_opts.get("lr_sched", None)
    # TODO: add early stopping/stop on plateau

    # create the optimizer with the parameters of our model
    model_instance.to(use_dev)
    opt_instance = optimizer(model_instance.parameters())
    if lr_scheduler:
        lr_scheduler = lr_scheduler(opt_instance)
    # create a dataset loader
    train_dataloader, val_dataloader, torch_featuremap = create_torch_dataloaders(features,
                                                                                  targets=features,
                                                                                  validation_frac=validation_frac,
                                                                                  batchsize=batchsize,
                                                                                  model=model_instance,
                                                                                  use_dev=use_dev)
    for epoch in range(epochs):
        e_loss = 0
        model_instance.train()
        for batch_data in train_dataloader:
            b_features, b_targets, b_pred_ae = get_train_batch(
                batch_data, model_instance, torch_featuremap)
            # TODO: make a function
            if isinstance(b_pred_ae, tuple) and isinstance(b_pred_ae[-1], dict) and len(b_pred_ae)==2:
                b_pred = b_pred_ae[0]
                loss_kw = b_pred_ae[-1]
            elif isinstance(b_pred_ae, tuple) and isinstance(b_pred_ae[-1], dict):
                b_pred = b_pred_ae[:-1]
                loss_kw = b_pred_ae[-1]
            else:
                b_pred = b_pred_ae
                loss_kw = {}
            loss = loss_fn(b_pred, b_targets, **loss_kw)
            e_loss += loss
            loss.backward()
            opt_instance.step()
            opt_instance.zero_grad()

        model_instance.eval()
        val_loss = 0
        if val_dataloader is not None:
            with torch.no_grad():
                for batch_data in val_dataloader:
                    b_features, b_targets, b_pred_ae = get_train_batch(
                        batch_data, model_instance, torch_featuremap)
                    if isinstance(b_pred_ae, tuple) and isinstance(b_pred_ae[-1], dict) and len(b_pred_ae)==2:
                        b_pred = b_pred_ae[0]
                        loss_kw = b_pred_ae[-1]
                    elif isinstance(b_pred_ae, tuple) and isinstance(b_pred_ae[-1], dict):
                        b_pred = b_pred_ae[:-1]
                        loss_kw = b_pred_ae[-1]
                    else:
                        b_pred = b_pred_ae
                        loss_kw = {}
                    val_loss += loss_fn(b_pred, b_targets, **loss_kw)
            if lr_scheduler:
                lr_scheduler.step(val_loss)

        print(f"TORCH AE {epoch+1}/{epochs}: "
              f"loss {e_loss/len(train_dataloader)}/"
              f"{val_loss/len(val_dataloader) if val_loss != 0 else '-'}")
    
def _train_sklearn_reducer(features, model_instance, run_opts):
    model_instance.fit(features)

#- Autoencoder/PCA-stuff: PREDICTION
def predict_encoder(model_instance, *features, encoding=True):
    pred = []
    for fts in features:
        if isinstance(model_instance, sklearn.base.BaseEstimator):
            pred.append(_predict_sklearn_reducer(fts, model_instance, encoding=encoding))
        elif isinstance(model_instance, torch.nn.Module):
            pred.append(_predict_torch_autoencoder(fts, model_instance, encoding=encoding))
    return pred


def _predict_torch_autoencoder(features, model_instance, encoding=True, use_dev=CUDA_DEV):
    model_instance.to(use_dev)
    torch_data, torch_featuremap = get_torch_dataset(features, use_dev=use_dev)
    test_dataloader = t_data.DataLoader(torch_data, batch_size=64)

    get_batch = None
    if encoding:
        get_batch = get_pred_batch_ae
    else:
        get_batch = get_pred_batch

    prediction = []
    model_instance.eval()
    with torch.no_grad():
        for batch_data in test_dataloader:
            b_features, b_pred = get_batch(
                batch_data, model_instance, torch_featuremap)
            prediction.append(b_pred)

    if isinstance(prediction[0], tuple):
        # quadratic runtime?
        final_prediction = []
        for idx, entrymodel in enumerate(prediction[0]):
            if isinstance(entrymodel, torch.Tensor):
                final_prediction.append(
                    np.array(torch.cat([
                        data[idx] for data in prediction
                    ]).to("cpu")))
            elif isinstance(entrymodel, dict):
                aggdict = {}
                for k in entrymodel.keys():
                    aggdict[k] = np.array(torch.cat([
                        data[idx][k] for data in prediction
                    ]).to("cpu"))
                final_prediction.append(aggdict)
        return tuple(final_prediction)
    elif isinstance(prediction[0], torch.Tensor):
        return np.array(torch.cat(prediction).to("cpu"))
    else:
        raise Exception

def _predict_sklearn_reducer(features, model_instance, encoding=True):
    if encoding:
        return model_instance.transform(features)
    else:
        raise NotImplementedError()

#- predictors: TRAINING, constructor should take a features_dim or just work with the input data...
def train_predictor(features, targets, feature_dims, constructor=None, constructor_opts={}, run_opts={}):
    try:
        predictor = constructor(feature_dims=feature_dims, **constructor_opts)
    except TypeError:
        print("Couldn't inject feature_dims, trying without...")
        predictor = constructor(**constructor_opts)

    other_objects = {}
    if isinstance(predictor, sklearn.base.BaseEstimator):
        _train_sklearn_predictor(features, targets, predictor, run_opts)
    elif isinstance(predictor, torch.nn.Module):
        predictor, other_objects = \
            _train_torch_predictor(features, targets, predictor, run_opts)
        
    return predictor, other_objects

def _train_torch_predictor(features, targets, model_instance, run_opts, use_dev=CUDA_DEV):
    model_instance.to(use_dev)

    epochs = run_opts.get("epochs", 100)
    batchsize = run_opts.get("batch_size", 32)
    validation_frac = run_opts.get("validate", None)

    
    loss_fn = run_opts.get("loss_fn", partial(torch.nn.MSELoss, reduction="sum"))()
    optimizer = run_opts.get("optim", torch.optim.Adam)
    # create the optimizer with the parameters of our model
    opt_instance = optimizer(model_instance.parameters())

    clip_grad = run_opts.get("clip_grad", None)

    # more fancy things....
    ema = run_opts.get("ema_decay", None)
    ema_train = False
    ema_instance = None
    if ema:
        ema_train = run_opts.get("ema_train", False)
        ema_instance = ema_torch.ExponentialMovingAverage(
                model_instance.parameters(),
                ema
            )

    lr_scheduler_val = run_opts.get("lr_sched_val", None)
    lr_scheduler_val_instance = None
    if lr_scheduler_val:
        lr_scheduler_val_instance = lr_scheduler(opt_instance)

    lr_scheduler_step = run_opts.get("lr_micro_sched", None)
    lr_scheduler_step_instance = None
    if lr_scheduler_step:
        lr_scheduler_step_instance = lr_scheduler_step(opt_instance)    

    # saving?
    save_to = run_opts.get("save_to", None)
    save_epochs = run_opts.get("save_epochs", 100)
    
    # create a dataset loader, get a pytorch_geometric one for any model including layers from pytorch_geometric...
    train_dataloader, val_dataloader, torch_featuremap = create_torch_dataloaders(features,
                                                                                  targets=targets,
                                                                                  validation_frac=validation_frac,
                                                                                  batchsize=batchsize,
                                                                                  model=model_instance,
                                                                                  use_dev=use_dev)
    tot_train = len(train_dataloader.dataset)
    tot_val = len(val_dataloader.dataset)
    print(f"Training on {tot_train}/{tot_val} training/validation-sample")
    for epoch in range(epochs):
        e_loss = 0
        part_losses = []
        part_count = []
        model_instance.train()
        for batch_data in train_dataloader:
            b_features, b_targets, b_pred = get_train_batch(
                batch_data, model_instance, torch_featuremap)
                
            loss = loss_fn(b_pred, b_targets)
            loss_c = float(loss.item())
            e_loss += loss_c
            part_losses.append(loss_c)
            if isinstance(batch_data, tg_data.Batch):
                part_count.append(int(max(batch_data.batch)+1))
            else:
                part_count.append(len(batch_data))
#https://github.com/Open-Catalyst-Project/ocp/blob/599ba4fcb69ab2bb648f98b34a781a2ee8852a35/ocpmodels/trainers/base_trainer.py#L607
            if hasattr(model_instance.modules, "shared_parameters"):
                for p, factor in model_instance.modules.shared_parameters:
                    if hasattr(p, "grad") and p.grad is not None:
                        p.grad.detach().div_(factor)
                    else:
                        if not WARN_SHARED_PARAM in locals():
                            WARN_SHARED_PARAM = True
                            print("Shared param without gradient!")
            
            opt_instance.zero_grad()            
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model_instance.parameters(), clip_grad)
            opt_instance.step()
            if ema_instance:
                ema_instance.update()
            if ema_train:
                ema_instance.copy_to()
            if lr_scheduler_step_instance:
                lr_scheduler_step_instance.step()

        model_instance.eval()
        val_loss = 0
        part_vlosses = []
        part_vcount = []
        if val_dataloader is not None:
            if ema_instance:
                ema_instance.store()
                ema_instance.copy_to()
            for batch_data in val_dataloader:
                b_features, b_targets, b_pred = get_train_batch(
                    batch_data, model_instance, torch_featuremap)
                vloss_c = loss_fn(b_pred, b_targets).item()
                val_loss += float(vloss_c)
                part_vlosses.append(loss_c)
                if isinstance(batch_data, tg_data.Batch):
                    part_vcount.append(int(max(batch_data.batch)+1))
                else:
                    part_vcount.append(len(batch_data))
            if lr_scheduler_val:
                lr_scheduler_val_instance.step(val_loss)
            if ema_instance:
                ema_instance.restore()
        if save_to:
            if epoch % save_epochs == 0:
                try:
                    torch.save(model_instance.state_dict(), save_to.format(epoch))
                except:
                    print("SAVING FAILED")
        if lr_scheduler_step_instance:
            print("Learning Rate: ", lr_scheduler_step_instance.get_last_lr())
        part_losses = np.array(part_losses)
        part_weights = np.array(part_count)
        part_weights = part_weights/sum(part_weights)
        final_loss = sum(part_weights*part_losses)
        if part_vlosses:
            part_vlosses = np.array(part_vlosses)
            part_vweights = np.array(part_vcount)
            part_vweights = part_vweights/sum(part_vweights)
            final_vloss = sum(part_vweights*part_vlosses)

        print(f"TORCH {epoch+1}/{epochs}:"
              f"loss {final_loss}/"
              f"{final_vloss if val_loss != 0 else '-'}", flush=True)

    model_instance_no_ema_dict = model_instance.state_dict()
    if ema_instance:
        try:
            torch.save(model_instance_no_ema_dict, save_to.format("final_no_ema"))
        except:
            print("SAVING FAILED")
    # put the moving averages into the model
    ema_instance.copy_to()
    try:
        torch.save(model_instance.state_dict(), save_to.format("final"))
    except:
        print("SAVING FAILED")
    return model_instance, {
        "gnn_without_ema_state" : model_instance_no_ema_dict,
        "optimizer" : opt_instance,
        "ema" : ema_instance,
        "lr_val": lr_scheduler_val_instance,
        "lr_step" : lr_scheduler_step_instance}
    

def _train_sklearn_predictor(features, targets, model_instance, run_opts):
    model_instance.fit(features, targets, **run_opts)


#- predictors: PREDICTION
def predict(model_instance, *features,):
    pred = []
    for fts in features:
        if isinstance(model_instance, sklearn.base.BaseEstimator):
            pred.append(_predict_sklearn(fts, model_instance))
        elif isinstance(model_instance, torch.nn.Module):
            pred.append(_predict_torch(fts, model_instance))
    return pred

def _predict_torch(features, model_instance, use_dev=CUDA_DEV):
    model_instance.to(use_dev)
    # create a dataset loader
    test_dataloader, _, torch_featuremap = create_torch_dataloaders(features,
                                                                    batchsize=64,
                                                                    model=model_instance,
                                                                    use_dev=use_dev)

    prediction = None
    model_instance.eval()
    no_concat = set()
    for batch_data in test_dataloader:
        b_features, b_pred = get_pred_batch(
            batch_data, model_instance, torch_featuremap)
        if prediction is None:
            if isinstance(b_pred, dict):
                prediction = dict((k, []) for k in b_pred.keys())
            else:
                prediction = []
        if isinstance(b_pred, dict):
            for k, v in b_pred.items():
                if isinstance(batch_data, tg_data.Batch):
                    nodes = len(batch_data.x)
                    samples = max(batch_data.batch)+1
                    if len(v) == nodes:
                        prediction[k].extend(
                            [v[batch_data.batch == idx].cpu().detach().numpy()
                             for idx in range(samples)]
                        )
                        no_concat.add(k)
                    else:
                        prediction[k].append(v.to("cpu").detach().numpy())
                else:
                    prediction[k].append(v.to("cpu").detach().numpy())
        else:
            pred_app = b_pred.to("cpu").detach().numpy()
            prediction.append(pred_app)
    if isinstance(prediction, dict):
        return dict(
            [(k, np.concatenate(v)) if (k not in no_concat) else (k, v)
             for k, v in prediction.items()]
        )
    else:
        return np.concatenate(prediction)

def _predict_sklearn(features, model_instance):
    return model_instance.predict(features)
