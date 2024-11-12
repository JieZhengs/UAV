from .aagcn import Model as aagcn_model
from .tegcn import Model as tegcn_model
from .mstgcn import Model as mstgcn_model
from .ctrgcn import Model as ctrgcn_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kan import KAN, KANLinear

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=6,
                 drop_out=0, adaptive=True, attention=True):
        super(Model, self).__init__()
        self.aagcn = aagcn_model(num_class, num_point, num_person, graph, graph_args, in_channels,
                 drop_out, adaptive, attention)
        self.tegcn = tegcn_model(num_class, num_point, num_person, graph, graph_args, in_channels,
                 drop_out, adaptive, attention)
        self.mstgcn = mstgcn_model(num_class, num_point, num_person, graph, graph_args, in_channels,
                 drop_out, adaptive)
        self.ctrgcn = ctrgcn_model(num_class, num_point, num_person, graph, graph_args, in_channels,
                 drop_out, adaptive)
        self.kan_linear = KANLinear(256, num_class)

    def forward(self, x):
        # outA = self.tegcn(x)
        # outB = self.aagcn(x)
        # outC = self.mstgcn(x)
        outD = self.ctrgcn(x)

        combined = outD
        out = self.kan_linear(combined)
        return combined
