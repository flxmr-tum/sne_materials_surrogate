from functools import partial
import itertools

import random

import numpy as np

import ase
import torch_tools
import data_tools as dt

from joblib import Parallel, delayed

from sne_ml_databases.util import dicts2df
from sne_ml_databases.property_tables import TABLES as prop_table

import dscribe.descriptors as ddesc

import sne_fingerprints.rdf as rdf


def soap_fp(struc, species=None, rcut=10, rbasis=8, sphericals=6, sigma=1.0):
    soap = ddesc.SOAP(rcut=rcut, nmax=rbasis, lmax=sphericals, species=species,
                      sigma=sigma, rbf='gto', periodic=False, sparse=False)
    fp = soap.create(struc, struc.positions)[0]
    return fp

def soap_periodic_mean(struc, species=['H'], rcut=10, rbasis=8, sphericals=6, sigma=1.0):
    soap = ddesc.SOAP(rcut=rcut, nmax=rbasis, lmax=sphericals, species=species,
                      sigma=sigma, rbf='gto', periodic=True, sparse=False)
    # soap is per atom!
    soaps = soap.create(struc)
    return np.sum(soaps, axis=0)/len(struc)

def mbtr2d_fp(struc, species=None, sigma=1.0, dist=10, grid_n=10):
    mspecies = list(species)
    mbtr = ddesc.MBTR(mspecies, periodic=False,
                      k2={
                          "geometry": {"function": "distance"},
                          "grid": {"min": 0.1, "max": dist, "sigma": sigma, "n": grid_n},
                          "weighting": {"function": "exp", "scale": 0.5, "cutoff": 1e-3}
                      },
                      normalization="n_atoms", # better than "l2_each" according to Obi
                      flatten=True)
    fp = mbtr.create(struc)[0]
    return fp

def p2ddf_fine_paper(struc):
    props=[
                  "ion_2", "ion_1", "r",  "en", "ea",  "e", "val_p", "val_s"
    ]
    fp = PDDF(radius=16, binsize=0.1,
              gaussian=1, weight="prod",
              props=props)[0](struc)
    fp_parts = []
    for p in props:
        if np.nan in fp[p].data:
            print(fp)
            raise Exception
        fp_parts.append(fp[p].data)
    return np.concatenate(fp_parts)
    

def PDDF(radius=16, binsize=0.1,
         gaussian=None, norm="3D",
         props=["ion_2", "ion_1", "en", "ea", "val_p", "val_s"], # also vwd_r, cov_r
         cumulate=False, weight="prod"):
    fp_weight = None
    if weight == "prod":
        fp_weight = rdf.w_product
    elif weight == "stanley":
        fp_weight = rdf.w_stanley_density
    elif weight == "square":
        fp_weight = rdf.w_square
    
    fp_desc = f"pddf_r{radius}_bin{binsize}"\
        f"-{gaussian}_{norm}"\
        f"_{'+'.join(props)}"\
        f"{'_cum' if cumulate else ''}_{weight}"
    return partial(rdf.calc_property_fp,
                   radius=radius, binsize=binsize,
                   gaussian=gaussian, norm_vol=norm,
                   properties=dict((k, prop_table[k]) for k in props),
                   property_mixer=fp_weight), fp_desc


def pddf_fp_mean(struc, species=None, radius=10, binsize=1,
                 gaussian=None, norm="3D", weight="stanley",
                 props=["ion_2", "ion_1", "cov_r", "en", "ea", "val_p", "val_s"]):
    fp_weight = None
    if weight == "prod":
        fp_weight = rdf.w_product
    elif weight == "stanley":
        fp_weight = rdf.w_stanley_density
    elif weight == "square":
        fp_weight = rdf.w_square
    else:
        fp_weight = rdf.w_unit

    fp = rdf.calc_property_fp(struc,
                              radius=radius, binsize=binsize, gaussian=gaussian,
                              norm_vol=norm, properties=dict((k, prop_table[k]) for k in props),
                              property_mixer=fp_weight)
    fp_parts = []
    for p in props:
        if np.nan in fp[p].data:
            print(fp)
            raise Exception
        fp_parts.append(fp[p].data)
    return np.concatenate(fp_parts)

def pddf_fp_multi(struc, species=None, radius=10, binsize=1,
                  gaussian=None, norm="3D", weight="stanley",
                  props=["ion_2", "ion_1", "cov_r", "en", "ea", "val_p", "val_s"]):
    fp_weight = None
    if weight == "prod":
        fp_weight = rdf.w_product
    elif weight == "stanley":
        fp_weight = rdf.w_stanley_density
    elif weight == "square":
        fp_weight = rdf.w_square
    else:
        fp_weight = rdf.w_unit

    fps = []
    for idx in range(len(struc)):
        fp = rdf.calc_property_fp_per_atom(struc, idx, ignorespec=['X'],
                                           radius=radius, binsize=binsize, gaussian=gaussian,
                                           norm_vol=norm, properties=dict((k, prop_table[k]) for k in props),
                                           property_mixer=fp_weight)
        fps.append(fp)

    

    per_atom_totals = []
    for idx in range(len(struc)):
        fp = fps[idx]
        fp_parts = []
        for p in props:
            if np.nan in fp[p]:
                print(fp)
                raise Exception
            fp_parts.append(fp[p])
        per_atom_totals.append(np.concatenate(fp_parts))
    return np.array(per_atom_totals)


def mbtr_simple_rdf(struc, species=['H'],
                    sigma=1.0, dist=10,
                    grid_n=10):
    mbtr = ddesc.MBTR(species, periodic=True,
                      #k1 = {
                      #    "geometry": {"function": "atomic_number"},
                      #    "grid": {"min": 1, "max": 100, "sigma": 0.4, "n": grid_n[0] }
                      #},
                      #k2 = {
                      #    "geometry": {"function": "distance"},
                      #              "grid": {"min": 0.1, "max": dist, "sigma": sigma, "n": grid_n },
                      #    "weighting": {"function": "exp", "scale": 0.75, "cutoff": 1e-2}
                      #},
                      k2 = {
                          "geometry": {"function": "distance"},
                          "grid": {"min": 0.1, "max": dist, "sigma": sigma, "n": grid_n },
                          "weighting": {"function": "exp", "scale": 0.75, "cutoff": 1e-2}
                      },
                      #k3 = {
                      #    "geometry": {"function": "angle"},
                      #    "grid": {"min": 0, "max": 180, "sigma": 5, "n": grid_n[2] },
                      #    "weighting" : {"function": "exp", "scale": 0.5, "cutoff": 1e-3}
                      #},
                      normalization = "n_atoms", # better than "l2_each" according to Obi
                      flatten=True)
                      #flatten=False)
    return mbtr.create(struc)[0]


"""
## COPIED FROM THE PAPER CODE!
# all these functions should not return a dictionary but the plain "per-molecule"-fingerprint
# changed the logic since then

FINGERPRINT_RADIUS = 16
FPS = {}
FPS["sinematrix_eigen"] = lambda spec, na: partial(dscw.sinematrix_eigen, n_atoms=na)
FPS["average_soap_low"] = lambda spec, na: partial(dscw.soap_periodic_mean,
                                                   species=spec,
                                                   rcut=6, #FINGERPRINT_RADIUS,
                                                   rbasis=8, sphericals=6)
FPS["average_soap_low_big"] = lambda spec, na: partial(dscw.soap_periodic_mean,
                                                   species=spec,
                                                   rcut=FINGERPRINT_RADIUS,
                                                   rbasis=8, sphericals=6)
# like in the original paper
FPS["average_soap_high"] = lambda spec, na: partial(dscw.soap_periodic_mean,
                                                    species=spec,
                                                    rcut=6,
                                                    rbasis=4, sphericals=8)
# like in the winner of the Nomad 2016 challenge
FPS["average_soap_nomad"] = lambda spec, na: partial(dscw.soap_periodic_mean,
                                                     species=spec,
                                                     rcut=10, sigma=0.5,
                                                     rbasis=4, sphericals=4)
FPS["average_soap_nomad_fine"] = lambda spec, na: partial(dscw.soap_periodic_mean,
                                                     species=spec,
                                                     rcut=10, sigma=0.1,
                                                     rbasis=4, sphericals=4)

FPS["MBTR-k2-inv"] = lambda spec, na: partial(dscw.mbtr_simple_inverse,
                                              species=spec,
                                              max_radius=FINGERPRINT_RADIUS,
                                              sigma=0.05, grid_n=16)
FPS["MBTR-k2-inv-broad"] = lambda spec, na: partial(dscw.mbtr_simple_inverse,
                                              species=spec,
                                              max_radius=FINGERPRINT_RADIUS,
                                              sigma=1, grid_n=16)
FPS["MBTR-k2-inv-100"] = lambda spec, na: partial(dscw.mbtr_simple_inverse,
                                                   species=spec,
                                                   max_radius=FINGERPRINT_RADIUS,
                                                   sigma=0.05, grid_n=100)
FPS["MBTR-k2-rdf"] = lambda spec, na: partial(dscw.mbtr_simple_rdf,
                                              species=spec,
                                              sigma=1.0, dist=FINGERPRINT_RADIUS,
                                              grid_n=round(FINGERPRINT_RADIUS*1))
FPS["MBTR-k2-rdf-broad"] = lambda spec, na: partial(dscw.mbtr_simple_rdf,
                                              species=spec,
                                              sigma=4, dist=FINGERPRINT_RADIUS,
                                              grid_n=round(FINGERPRINT_RADIUS*1))
FPS["MBTR-k2-rdf-100"] = lambda spec, na: partial(dscw.mbtr_simple_rdf,
                                                  species=spec,
                                                  sigma=1.0, dist=FINGERPRINT_RADIUS,
                                                  grid_n=100)
FPS["MBTR-full"] = lambda spec, na: partial(dscw.mbtr_simple_full,
                                            species=spec,
                                            max_radius=FINGERPRINT_RADIUS,
                                            sigma_r=0.05, grid_r=16,
                                            sigma_angle=5, grid_angle=10)


def PDDF(radius=FINGERPRINT_RADIUS, binsize=0.1,
         gaussian=None, norm="3D",
         props=["ion_2", "ion_1", "r", "en", "ea", "val_p", "val_s"], # also vwd_r, cov_r
         cumulate=False, weight="prod"):
    fp_weight = None
    if weight == "prod":
        fp_weight = rdf.w_product
    elif weight == "stanley":
        fp_weight = rdf.w_stanley_density
    elif weight == "square":
        fp_weight = rdf.w_square
    
    fp_desc = f"pddf_r{radius}_bin{binsize}"\
        f"-{gaussian}_{norm}"\
        f"_{'+'.join(props)}"\
        f"{'_cum' if cumulate else ''}_{weight}"
    return partial(rdf.calc_property_fp,
                   radius=radius, binsize=binsize,
                   gaussian=gaussian, norm_vol=norm,
                   properties=dict((k, prop_table[k]) for k in props),
                   property_mixer=fp_weight), fp_desc


## the fingerprinting functions from  dscw
import numpy as np
import pandas as pd

import dscribe
import dscribe.descriptors as ddesc

def sinematrix_eigen(struc, n_atoms=2):
    # TODO: maybe try sorted_l2 as well
    sm = ddesc.SineMatrix(n_atoms, permutation='eigenspectrum', flatten=True)
    return {"sm_e" : sm.create(struc)[0]}

def ewaldmatrix_eigen(struc, rcut=10, gcut=10, gaussian=0.1, n_atoms=2):
    em = ddesc.EwaldSumMatrix(n_atoms, permutation='eigenspectrum', flatten=True)
    return {"em_e" : em.create(struc, rcut=rcut, gcut=gcut, a=gaussian)[0]}

def soap_periodic_mean(struc, species=['H'], rcut=10, rbasis=8, sphericals=6, sigma=1.0):
    soap = ddesc.SOAP(rcut=rcut, nmax=rbasis, lmax=sphericals, species=species,
                      sigma=sigma, rbf='gto', periodic=True, sparse=False)
    # soap is per atom!
    soaps = soap.create(struc)
    return {"soap" : np.sum(soaps, axis=0)/len(struc)}

def mbtr_simple_rdf(struc, species=['H'],
                    sigma=1.0, dist=10,
                    grid_n=10):
    mbtr = ddesc.MBTR(species, periodic=True,
                      #k1 = {
                      #    "geometry": {"function": "atomic_number"},
                      #    "grid": {"min": 1, "max": 100, "sigma": 0.4, "n": grid_n[0] }
                      #},
                      #k2 = {
                      #    "geometry": {"function": "distance"},
                      #              "grid": {"min": 0.1, "max": dist, "sigma": sigma, "n": grid_n },
                      #    "weighting": {"function": "exp", "scale": 0.75, "cutoff": 1e-2}
                      #},
                      k2 = {
                          "geometry": {"function": "distance"},
                          "grid": {"min": 0.1, "max": dist, "sigma": sigma, "n": grid_n },
                          "weighting": {"function": "exp", "scale": 0.75, "cutoff": 1e-2}
                      },
                      #k3 = {
                      #    "geometry": {"function": "angle"},
                      #    "grid": {"min": 0, "max": 180, "sigma": 5, "n": grid_n[2] },
                      #    "weighting" : {"function": "exp", "scale": 0.5, "cutoff": 1e-3}
                      #},
                      normalization = "n_atoms", # better than "l2_each" according to Obi
                      flatten=True)
                      #flatten=False)
    #return mbtr.create(struc)
    return {"mbtr" : mbtr.create(struc)[0]}

def mbtr_simple_inverse(struc, species=['H'],
                        max_radius=10, min_radius=1,
                        sigma=0.02,  grid_n=10):
    mbtr = ddesc.MBTR(species, periodic=True,
                      #k1 = {
                      #    "geometry": {"function": "atomic_number"},
                      #    "grid": {"min": 1, "max": 100, "sigma": 0.4, "n": grid_n[0] }
                      #},
                      #k2 = {
                      #    "geometry": {"function": "distance"},
                      #              "grid": {"min": 0.1, "max": dist, "sigma": sigma, "n": grid_n },
                      #    "weighting": {"function": "exp", "scale": 0.75, "cutoff": 1e-2}
                      #},
                      k2 = {
                          "geometry": {"function": "inverse_distance"},
                                    "grid": {"min": 1/max_radius, "max": 1/min_radius, "sigma": sigma, "n": grid_n },
                          "weighting": {"function": "exp", "scale": 1.0, "cutoff": 1e-3}
                      },
                      #k3 = {
                      #    "geometry": {"function": "angle"},
                      #    "grid": {"min": 0, "max": 180, "sigma": 5, "n": grid_n[2] },
                      #    "weighting" : {"function": "exp", "scale": 0.5, "cutoff": 1e-3}
                      #},
                      normalization = "n_atoms", # better than "l2_each" according to Obi
                      flatten=True)
                      #flatten=False)
    #return mbtr.create(struc)
    return {"mbtr" : mbtr.create(struc)[0]}


def mbtr_simple_full(struc, species=['H'],
                     max_radius=10, min_radius=1,
                     sigma_r=0.02,  grid_r=10,
                     sigma_angle=5, grid_angle=10
):
    mbtr = ddesc.MBTR(species, periodic=True,
                      k1 = {
                          "geometry": {"function": "atomic_number"},
                          "grid": {"min": 1, "max": len(species), "sigma": 1.0, "n": len(species) }
                      },
                      k2 = {
                          "geometry": {"function": "inverse_distance"},
                          "grid": {"min": 1/max_radius, "max": 1/min_radius, "sigma": sigma_r, "n": grid_r },
                          "weighting": {"function": "exp", "scale": 1.0, "cutoff": 1e-3}
                      },
                      k3 = {
                         "geometry": {"function": "angle"},
                         "grid": {"min": 0, "max": 180, "sigma": sigma_angle, "n": grid_angle },
                         "weighting" : {"function": "exp", "scale": 0.5, "cutoff": 1e-3}
                      },
                      normalization = "n_atoms", # better than "l2_each" according to Obi
                      flatten=True)
                      #flatten=False)
    #return mbtr.create(struc)
    return {"mbtr" : mbtr.create(struc)[0]}


def ascf_periodic(struc, species=['H'], rcut=10):
    #https://singroup.github.io/dscribe/tutorials/acsf.html
    raise NotImplementedError
"""
