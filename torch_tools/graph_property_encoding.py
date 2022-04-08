# Graph node encoding functions
import ase

import numpy as np

import pandas as pd

from dscribe.descriptors import SOAP

from sne_ml_databases.property_tables import TABLES as atomic_data
from sne_ml_databases.property_tables import TABLES_NUMERIC as atomic_data_numeric

# TODO: split up in parts AND TEST!
def _weight_and_expand(numbers_in, min_number, max_number,
                       weight_method : str = "basic", # or False or 1/r2 or ...
                       expand_method : str = False, # or "ohe" or "gauss"
                       expand_bins : int = 50, expand_width=0.2,
                       ohe_include_extremes=False):
    numbers = np.array(numbers_in)
    bins = None
    if weight_method:
        if weight_method == "basic":
            numbers = numbers
            bins = np.linspace(min_number, max_number, num=expand_bins+1)
        elif weight_method == "1/r2":
            numbers = 1/numbers
            bins = np.geomspace(1/max_number, 1/max(min_number, 1e-10), num=expand_bins+1)
        else:
            raise NotImplementedError
    else:
        numbers = np.ones((len(numbers)))
        bins = np.array([0, 2])

    numbers_raw = numbers.copy()
    if expand_method == "ohe":
        numbers = np.zeros((len(numbers_raw), len(bins)-1))
        # bins are 1-indexed! I should have taken care of errors by modifying bounds for the binning
        bin_positions = np.digitize(numbers_raw, bins) - 1
        for row, c in enumerate(bin_positions):
            if ohe_include_extremes:
                numbers[row, min(max(c, 0), expand_bins-1)] = 1
            else:
                if c > -1 and c < expand_bins:
                    numbers[row, c] = 1
    elif expand_method == "gauss":
        if weight_method != "basic":
            raise NotImplementedError
        real_x = bins[:-1] # remove the last element, as we want to evaluate a gaussian
        real_x = real_x + (bins[1] - bins[0])/2 # move evaluation center to center of bins!
        # finally evaluate the gaussian
        numbers = np.exp(-(numbers_raw[..., np.newaxis] - real_x)**2/expand_width**2)
    elif expand_method is False:
        numbers = numbers.reshape((len(numbers), 1))
    else:
        raise NotImplementedError
    return numbers

def get_plain_atom_features(struc, per_atom_features=["e", "r"]):
    #atom_df = pd.DataFrame(np.zeros((len(struc), len(per_atom_features))),
    #                       columns=per_atom_features, index=struc.get_chemical_symbols())
    atom_data = np.zeros((len(struc), len(per_atom_features)))
    for idx, spec in enumerate(struc.get_chemical_symbols()):
        atom_data[idx] = np.array([atomic_data[p][spec] for p in per_atom_features])
    return atom_data


def _get_tabulated_ohe(per_atom_features=["e", "r"], max_e=54, bins : (int, dict) = 10):
    bins2do = {} # {"e": 10, "r" : 5}
    if isinstance(bins, int):
        bins2do = dict([(f, bins) for f in per_atom_features])
    else:
        bins2do = bins
    total_bins = sum(bins2do.values())

    table = np.zeros((max_e, total_bins))
    bin_offset = 0
    for prop in per_atom_features:
        valid_properties = atomic_data_numeric[prop][:max_e]
        max_prop = np.nanmax(valid_properties)
        min_prop = np.nanmin(valid_properties)

        bin_edges = np.linspace(min_prop, max_prop, num=bins2do[prop]+1)
        bin_edges[-1] += 1e-10
        bin_positions = np.digitize(valid_properties, bin_edges)
        # TODO: think about nans
        for atom_num, bin_pos in enumerate(bin_positions):
            if bin_pos == 0 or bin_pos == bins2do[prop]+1:
                pass
            else:
                table[atom_num, bin_offset+bin_pos-1] = 1
        bin_offset += bins2do[prop]
    return table
    

def get_ohe_atom_features(struc, per_atom_features=["e", "r"],
                          max_e=80, bins : (int, dict) = 10):
    ohe_table = _get_tabulated_ohe(per_atom_features=per_atom_features, max_e=max_e, bins=bins)
    node_feature_data = np.zeros((len(struc), ohe_table.shape[1]))
    for idx, anumber in enumerate(struc.numbers):
        node_feature_data[idx] = ohe_table[anumber-1]
    return node_feature_data

def _get_tabulated_gaussians(per_atom_features=["e", "r"], max_e=54, bins : (int, dict) = 10):
    bins2do = {} # {"e": 10, "r" : 5}
    if isinstance(bins, int):
        bins2do = dict([(f, bins) for f in per_atom_features])
    else:
        bins2do = bins
    total_bins = sum(bins2do.values())

    all_encs = []
    for prop in per_atom_features:
        valid_properties = atomic_data_numeric[prop][:max_e]
        max_prop = np.nanmax(valid_properties)
        min_prop = np.nanmin(valid_properties)
        expand_width = 0.5*((max_prop - min_prop)/bins2do[prop])

        all_encs.append(_weight_and_expand(valid_properties, min_prop, max_prop,
                                           weight_method="basic",
                                           expand_method="gauss",
                                           expand_bins=bins2do[prop],
                                           expand_width=expand_width,
                                           ohe_include_extremes=True))
    table = np.hstack(all_encs)
    return table

def get_gaussian_atom_features(struc, per_atom_features=["e", "r"],
                               max_e=80, bins : (int, dict) = 10):
    g_table = _get_tabulated_gaussians(per_atom_features=per_atom_features,
                                        max_e=max_e, bins=bins)
    node_features = []
    for idx, anumber in enumerate(struc.numbers):
        node_features.append(g_table[anumber-1])
    return np.vstack(node_features)


def get_soap_env(struc, species=["H", "O"], force_periodic=True,
                 # takes in all options from dscribe.soap!
                 **soapopts):
    if "average" in soapopts.keys():
        raise NotImplementedError("No averaging supported for node-feature-soap")
    soap_op = SOAP(species=species, periodic=force_periodic, **soapopts)

    per_node_soap = soap_op.create(struc)
    return per_node_soap

def get_simple_rdf_env(struc, r_cut=5, n_bins=10):
    pass
