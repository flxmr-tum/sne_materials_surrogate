import torch
import torch.nn as nn
from torch.nn import Linear, BatchNorm1d

import torch_geometric
from torch_geometric.nn import MessagePassing, CGConv, \
    global_mean_pool as gap, global_max_pool as gmp
import torch_geometric.data as tg_data

### Xies CGCN using pytorch_geometric-messagepassing!
# taken from https://github.com/pyg-team/pytorch_geometric/issues/1381

class CGConv_pyggit(MessagePassing):
    def __init__(self, orig_atom_fea_len, nbr_fea_len, atom_fea_len=64,
                 aggr: str = 'add', bias: bool = True, **kwargs):
        super(CGConv_pyggit, self).__init__(aggr=aggr, flow='target_to_source', **kwargs)
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.orig_atom_fea_len = orig_atom_fea_len

        self.BatchNorm1 = BatchNorm1d(atom_fea_len)
        self.BatchNorm2 = BatchNorm1d(atom_fea_len)
        self.embedding = Linear(orig_atom_fea_len, atom_fea_len)
        self.lin_f = Linear(2*atom_fea_len + nbr_fea_len, atom_fea_len, bias=bias)
        self.lin_s = Linear(2*atom_fea_len + nbr_fea_len, atom_fea_len, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.softplus2 = nn.Softplus() 
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()

    def forward(self, x, edge_index, edge_attr, size=None):
        """"""
        if x.shape[1] == self.orig_atom_fea_len:
            x = self.embedding(x)
        x = self.BatchNorm1(x)
        if isinstance(x, torch.Tensor):
            x: torch_geometric.PairTensor = (x, x)
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.BatchNorm2(self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size))
        out = out + x[1]
        return self.softplus1(out)

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        # print(z.shape)
        return self.sigmoid(self.lin_f(z)) * self.softplus2(self.lin_s(z))

                                       
class CrystalGraphConvNet_pyggit(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1):

        super(CrystalGraphConvNet_pyggit, self).__init__()
        self._USE_TORCH_GEOMETRIC = True
        self.convs = nn.ModuleList([CGConv_pyggit(orig_atom_fea_len=orig_atom_fea_len
                                   ,atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])        
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        
        self.fc_out = nn.Linear(h_fea_len, 1)


    def forward(self, data, ret_intermediate=False):
        atom_fea, bond_index, bond_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv_func in self.convs:
            atom_fea = conv_func(x=atom_fea, edge_index=bond_index, edge_attr=bond_attr)

        crys_fea = torch.cat([gap(atom_fea, batch)], dim=1) #, gmp(atom_fea, batch)], dim=1)
        #print("pooled", crys_fea.shape)
        crys_fea_inter = crys_fea
        
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        
        out = self.fc_out(crys_fea)

        if ret_intermediate:
            return out, crys_fea_inter
        else:
            return out 




class CrystalGraphConvNet_simplified(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1):

        super(CrystalGraphConvNet_pyggit, self).__init__()
        self._USE_TORCH_GEOMETRIC = True
        self.convs = nn.ModuleList([CGConv(orig_atom_fea_len=orig_atom_fea_len,
                                           atom_fea_len=atom_fea_len,
                                           nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        
        self.fc_out = nn.Linear(h_fea_len, 1)
        self.conv_to_fc = nn.Linear(2*atom_fea_len, h_fea_len)

    def forward(self, data, ret_intermediate=False):
        atom_fea, bond_index, bond_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv_func in self.convs:
            atom_fea = conv_func(x=atom_fea, edge_index=bond_index, edge_attr=bond_attr)

        crys_fea = torch.cat([gap(atom_fea, batch), gmp(atom_fea, batch)], dim=1)
        crys_fea_inter = crys_fea
        
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        
        out = self.fc_out(crys_fea)

        if ret_intermediate:
            return out, crys_fea_inter
        else:
            return out

