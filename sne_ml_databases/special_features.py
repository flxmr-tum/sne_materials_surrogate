import json
import numpy as np
import pandas as pd
from sqlalchemy.orm.exc import NoResultFound
import os.path

from mendeleev import element

# get the ionic radii
"""
except otherwise noted (see comments)
data from "Revised Effective Ionic Radii and Systematic Studies of Interatomic Distances in Halides and Chalcogenides" By R. D. Shannon. Central Research and Development Department, Experimental Station, E. I. Du Pont de Nemours and Company, Wilmington, Delaware 19898, U.S.A.
Published in Acta Crystallographica. (1976). A32, Pages 751-767.
digital: https://github.com/prtkm/ionic-radii
"""
def get_ionradius(ionstring, charge, coordination=None):
    radius = None
    # read the radius for the ion from Shannon's table
    shannon_table = json.load(open(
        "{}/atomic_features.data/shannon-radii.json".format(
            os.path.dirname(__file__))
        , 'r'))
    if ionstring in shannon_table.keys():
        try:
            radii = shannon_table[ionstring][str(charge)]
            if coordination:
                radius = radii[coordination]
            else:
                radius = np.average([r["r_crystal"] for r in radii.values()])
        except Exception as e:
            # handle exceptions
            if ionstring == "Sn" and charge == 2:
                radius = 1.35 # https://pubs.rsc.org/en/content/articlepdf/2018/ra/c8ra00809d
            else:
                print(e, "Problem loading the {} ionradius for charge"
                  "{}/coordination {}".format(ionstring, charge, coordination))
                pass
    elif isinstance(ionstring, str) and charge == 1:
        # Kieslich, 2014 in Angstrøm (except otherwise noted) https://pubs.rsc.org/en/content/articlepdf/2014/sc/c4sc02211d
        # [1]: https://pubs.rsc.org/en/content/articlepdf/2016/sc/c5sc04845a
        # [2]: alternative to Kieslich: 10.1139/v84-052
        # [3]: review: https://link.springer.com/content/pdf/10.1007%2Fs00706-017-1933-9.pdf
        # [4]: apparently what is in the kimdatabase use: https://pubs.rsc.org/en/content/articlepdf/2017/dt/c6dt04796c
        #molecular_cation_radii = {
        #    "acetamidinium": 2.77, #CH3C(NH2)2 [1]
        #    "ammonium": 1.46, # NH4
        #    "azetidinium": 2.50, #(CH2)3NH2
        #    "butylammonium": 4.94, #CH3CH2CH2CH2NH3
        #    "dimethylammonium": 2.72, #(CH3)2NH2
        #    "ethylammonium": 2.74, #(C2H5)NH5
        #     "formamidinium": 2.53, #NH2(CH)NH2
        #     "guanidinium": 2.78, #C(NH2)3
        #    "hydrazinium": 2.17, #H3N-NH2
        #    "hydroxylammonium": 2.16, # H3NOH
        #    "imidazolium": 2.58, #C3N2H5
        #    "isopropylammonium": None, #(CH3)2CHNH3 
        #    "methylammonium": 2.17, #(CH3)NH3 2.71
        #    "propylammonium": None, #CH3CH2CH2NH3
        #    "trimethylammonium": None, #(CH3)3NH
        #    "tetramethylammonium": 2.92, #(CH3)4N
        #}
        ## everything from 4
        molecular_cation_radii = {
            "acetamidinium": 3.00, #CH3C(NH2)2 [1]
            "ammonium": 1.70, # NH4
            "azetidinium": 2.84 , #(CH2)3NH2
            "butylammonium": 3.60, #CH3CH2CH2CH2NH3
            "dimethylammonium": 2.96, #(CH3)2NH2
            "ethylammonium": 2.99, #(C2H5)NH5
            "formamidinium": 2.77, #NH2(CH)NH2
            "guanidinium": 2.80, #C(NH2)3
            "hydrazinium": 2.20, #H3N-NH2
            "hydroxylammonium": 2.26, # H3NOH
            "imidazolium": 3.03, #C3N2H5
            "isopropylammonium": 3.07, #(CH3)2CHNH3
            "methylammonium": 2.38, #(CH3)NH3 2.71
            "propylammonium": 3.07, #CH3CH2CH2NH3
            "trimethylammonium": 3.04, #(CH3)3NH
            "tetramethylammonium": 3.01, #(CH3)4N
        }
        radius = molecular_cation_radii.get(ionstring, None)

    return radius

def get_electron_affinity(chem_exp):
    try:
        return element(chem_exp).electron_affinity
    except NoResultFound:
        print("using mendeleev. for molecules add own property-db")

def get_pauling_en(chem_exp):
    try:
        return element(chem_exp).en_pauling
    except NoResultFound:
        print("using mendeleev. for molecules add own property-db")

def get_outer_shell(chem_exp, type='p'):
    pass
