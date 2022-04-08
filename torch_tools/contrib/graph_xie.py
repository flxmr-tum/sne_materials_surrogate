import json
import numpy as np

from pathlib import Path
FILELOC = Path(__file__).parent.absolute()

import ase
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

## Xie:
# tried 2,3,9 atom features: no difference appearing for eq5-conv function
#     - specific props:
# for bond-features used 
# same for layers (though 2-3 show marginal improvement)
# optimizer: very coarse stepsize doesn't work, everything else does

# GaussianDistance and AtomInitializer copied verbatim
class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file=FILELOC.joinpath("data/atom_init_xie.json")):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)




def create_xie_graph(struc : ase.Atoms, atom_featurizer=None,
                     radius=8,nneighbors=12, gauss_min=0, gauss_step=0.2,
                     warn=False,
                     ):
    if atom_featurizer is None:
        atom_featurizer = AtomCustomJSONInitializer()
    try:
        crystal = AseAtomsAdaptor.get_structure(struc)
    except Exception as e:
        print(struc)
        raise e
    atom_fea = np.vstack([atom_featurizer.get_atom_fea(crystal[i].specie.number)
                          for i in range(len(crystal))])
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    max_num_nbr = nneighbors
    firstwarn = warn
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            if firstwarn:
                print('WARNING: {} not find enough neighbors to build graph. '
                      'If it happens frequently, consider increase '
                      'radius.'.format(struc))
                firstwarn = False
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                               [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                           [radius + 1.] * (max_num_nbr - len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2],
                                        nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1],
                                    nbr[:max_num_nbr])))
    nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)

    nbr_fea_idx_list = np.array([[i,]*max_num_nbr for i in range(len(crystal))])
    nbr_connectivity = np.vstack([nbr_fea_idx_list.flatten(), nbr_fea_idx.flatten()])

    gauss_expander = GaussianDistance(gauss_min, radius, gauss_step)
    #print(nbr_fea, np.concatenate(nbr_fea).reshape((len(crystal)*max_num_nbr,1)))
    #print(gauss_expander.expand(nbr_fea).shape)
    nbr_fea = gauss_expander.expand(np.concatenate(nbr_fea).reshape((len(crystal)*max_num_nbr,1)))[:,0,:]

    
    atom_adj_tensor = nbr_connectivity
    atom_adj_features = nbr_fea
    return atom_fea, atom_adj_tensor, atom_adj_features
