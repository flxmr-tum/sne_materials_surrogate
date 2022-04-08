import mendeleev as ml
from itertools import chain

import numpy as np

def average_countdict(d, proptable):
    total = 0
    tot_count = 0
    for spec, count in d.items():
        total += count*proptable[spec]
        tot_count += count
    return total/tot_count

def mendeleev_get_valence_orbitals(ec, name="s"):
    try:
        valence = ec.get_valence()
    except TypeError as e:
        if ec.confstr == "1s":
        # only with H!
            if name == "s":
                return 1
            else:
                return 0
        else:
            raise e

    valence_count = 0
    for orb, occ in valence.conf.items():
        if orb[1] == name:
            valence_count += occ
    return valence_count

# JARED: Pauling EN, first and second ionization energies, s and p-orbital radii, empirical covalent radius, LUMO, HOMO, Fermi-Level, ionic radius, valence s/p-occupations for the neutral atom
# → MISSING: s,p-orbital radii, LUMO, HOMO, Fermi-level, ionic_radii without coordination/table
_block_lookup = {"s" : 1, "p" : 2, "d" : 3, "f" : 4}
_TABLES_RAW = [(el.symbol,
                {"n" : el.atomic_number,
                 "e" : el.electrons,
                 "ea" : el.electron_affinity,
                 "en" : el.en_pauling,
                 "ion_1" : el.ionenergies.get(1, -1),
                 "ion_2" : el.ionenergies.get(2, -1),
                 "r" : el.atomic_radius,
                 "ions_table" : el.ionic_radii,
                 "vdw_r" : el.vdw_radius,
                 "cov_r" : el.covalent_radius,
                 "block" : _block_lookup.get("block", 5),
                 "val_tot" : el.ec.get_valence().ne() if el.atomic_number > 1 else 1,
                 "val_s" : mendeleev_get_valence_orbitals(el.ec, "s") ,
                 "val_p" : mendeleev_get_valence_orbitals(el.ec, "p"),
                 "group_no" : el.group_id,
                 "period_no" : el.period,
                 "atomic_vol" : el.atomic_volume})
               for el
               in ml.elements.get_all_elements()]
_TABLES_RAW.sort(key=lambda x: x[1]["n"])
_TABLES_RAW.append(("X",
                    {
                        "n" : -1,
                        "e" : None,
                        "ea" : None,
                        "en" : None,
                        "ion_1" : None,
                        "ion_2" : None,
                        "r" : None,
                        "ions_table" : None,
                        "vdw_r" : None,
                        "cov_r" : None,
                        "block" : None,
                        "val_tot" : None,
                        "val_s" : None,
                        "val_p" : None,
                        "period_no" : None,
                        "group_no" : None,
                        "atomic_vol": None}))
TABLES_names = set(chain(*(props[1].keys() for props in _TABLES_RAW)))
TABLES = dict((propname,
               dict([(k, v[propname])
                     if v[propname] is not None else (k, np.nan)
                     for k, v in _TABLES_RAW]))
              for propname in TABLES_names)

# TODO: CARE FOR SORTING THE RAW TABLE!
TABLES_NUMERIC = dict((propname,
                       np.array([v[propname]
                                 if v[propname] is not None else np.nan
                                 for k, v in _TABLES_RAW[:-1]], dtype=object))
                       for propname in TABLES_names)
                      
