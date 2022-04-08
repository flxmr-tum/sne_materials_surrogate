from torch import nn

class CrystalGraphConvNetPYG(nn.Module):
    def __init__(self, atom_feat_size, edge_feat_size,
                 atom_embed_size=64,
                 conv_ops=4, skip_conv=None,
                 final_dense=1):
        super(CrystalGraphConvNetPYG, self).__init__()
        self.embedding = nn.Linear (atom_feat_size, atom_embed_size)

        self.convs = nn.ModuleList([
            i for i in range(conv_ops)
        ])
