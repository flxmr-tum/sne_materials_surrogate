# Xie et Grossman: expansion of the distances in an Gaussian basis
import numpy as np

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    Unit: angstrom
    taken from https://github.com/txie-93/cgcnn/blob/master/cgcnn/data.py
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

"""
Xies take on the graph convolution (the catted vector is passed through a [cat-vector -> 2x atom_fea]-
>>> atom_fea
tensor([[1, 2],
        [2, 3]])
>>> torch.cat([atom_fea.unsqueeze(1).expand(2,2, 2), nbr_fea], dim=2)
tensor([[[ 1.0000,  2.0000,  0.1000,  0.2000],
         [ 1.0000,  2.0000,  0.5000,  0.4000]],

        [[ 2.0000,  3.0000, -0.1000,  0.1500],
         [ 2.0000,  3.0000,  0.2000,  0.4000]]])
"""
