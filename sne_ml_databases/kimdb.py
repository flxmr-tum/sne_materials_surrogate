import os
import re

import numpy as np
import numpy.random

import ase.io

from .special_features import get_ionradius, get_electron_affinity, get_pauling_en

from joblib import Parallel, delayed

UNIQUE_LABEL = "hoip"
CORE_PROPERTIES = [
    "anion", "crystal_cation", "center_cation_name", # basic names
    "E_a", "gap_pbe", "gap_hse", # basic targets E_a in eV/atom
    "r_center", "r_anion", "r_cation", # ionic radii
    "ea_anion", "ea_crystal_cation", #electron_affinity
    "en_anion", "en_crystal_cation", #electronegativity
]
REALSTRUC = ["struc",]
PEROVSKITESTRUC = {True: ["simplified_struc", "override_spec"],
                   False: ["struc",]}

def kimcif_to_dict(path):
    info = {}
    # read the file
    info["struc"] = ase.io.read(path)
    contents = None
    with open(path, 'r') as commented_cif:
        contents = commented_cif.read()
    # parse the file

    def extract_property(pname):
        m = re.search(r'#.*{0}.*:\s+(.*)'.format(re.escape(pname)), contents)
        prop = m.group(1)
        return prop

    info["hoip"] = extract_property("HOIP entry ID")
    info["label"] = extract_property("Label")
    info["center_cation"] = extract_property("Organic cation chemical formula")
    info["center_cation_name"] = info["label"].split()[0].lower()
    info["r_center"] = get_ionradius(info["center_cation_name"], 1)

    typelist = extract_property("Atom types").split(" ")
    #numberlist = extract_property("Number of each atom").split(" ")
    info["anion"] = typelist[-1]
    info["r_anion"] = get_ionradius(info["anion"], -1)
    info["crystal_cation"] = typelist[-2]
    info["r_cation"] = get_ionradius(info["crystal_cation"], +2)

    # get more data from mendeleev
    # electron_affinity, either: en_[allen|ghosh|mulliken|pauling]
    # maybe:
    # group, period, econf
    info["ea_anion"] = get_electron_affinity(info["anion"])
    info["ea_crystal_cation"] = get_electron_affinity(info["crystal_cation"])
    info["en_anion"] = get_pauling_en(info["crystal_cation"])
    info["en_crystal_cation"] = get_pauling_en(info["crystal_cation"])

    # extract the molecular structure (thankfully the DB only includes CHN-molecules...)
    center_cation_struc = info["struc"].copy()
    for sym in [info["anion"], info["crystal_cation"]]:
        del center_cation_struc[
            np.where(np.array(
                center_cation_struc.get_chemical_symbols()) == sym)[0]
        ]
    info["center_com"] = center_cation_struc.get_center_of_mass()
    info["center_cation_struc"] = center_cation_struc
    simplified_struc = info["struc"].copy()
    for sym in ["C", "N", "H", "O"]:
        del simplified_struc[
            np.where(np.array(
                simplified_struc.get_chemical_symbols()) == sym)[0]
        ]
    info["simplified_struc"] = simplified_struc + ase.Atom("Pu", position=info["center_com"])
    # again Pu as a Placehulder
    override_spec = {"Pu": "A", info["crystal_cation"] : "B", info["anion"] : "X"}
    info["override_spec"] = override_spec
    info["general_radii"] = {"A" : info["r_center"], "B": info["r_cation"], "X": info["r_anion"]}

    info["gap_pbe"] = float(extract_property("Bandgap, GGA"))
    info["gap_hse"] = float(extract_property("Bandgap, HSE"))

    info["k_VBM"] = [float(x) for x in extract_property("Kpoint for VBM").split(",")]
    info["k_CBM"] = [float(x) for x in extract_property("Kpoint for CBM").split(",")]

    info["E_a"] = float(extract_property("Atomization"))
    info["E_1"] = float(extract_property("energy1"))
    info["E_2"] = float(extract_property("energy2"))

    # maybe add dielectric constants
    return info


def load(db_folder, parallel=4):
    filenames = ["{}/{}".format(db_folder, f)
                for f in os.listdir(db_folder) if f.endswith(".cif")]

    structures = Parallel(n_jobs=parallel)(delayed(kimcif_to_dict)(fn) for fn in filenames)
    """ see the use of partial. seems very similar to the application of "delayed" in joblib
    with mp.Pool() as p:
        print("loading structures")
            cifreader = functools.partial(
                kimcif_to_dict,
                fp_func=fpfunc,
                fpopts=fpopts
            )
            structures = list(p.map(
                cifreader, filenames
            ))
            pickle.dump(structures, open(strucstore, "wb"))
        else:
            structures = pickle.load(open(strucstore, "rb"))
    """
    structures.sort(key=lambda x: x["hoip"])
    return structures
