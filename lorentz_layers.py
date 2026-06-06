import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
class LorentzLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dropout=0.1,
                 scale=10, fixscale=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(in_features, out_features + 1, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(
            torch.ones(()) * math.log(scale),
            requires_grad=not fixscale
        )
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            self.weight.weight[:, 0] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)
    def forward(self, x):
        x = self.weight(self.dropout(x))  
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)  
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1  
        scale = (time * time - 1) / (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x
class LorentzAgg(nn.Module):
    def __init__(self, in_features, dropout=0.1, use_att=False):
        super().__init__()
        self.in_features = in_features
        self.dropout = dropout
        self.use_att = use_att
        if use_att:
            self.query_linear = LorentzLinear(in_features, in_features, dropout=dropout)
            self.key_linear = LorentzLinear(in_features, in_features, dropout=dropout)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))
    def lorentz_inner(self, x, y):
        return -x[..., 0:1] * y[..., 0:1] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
    def cinner(self, x, y):
        x_time = x[..., 0:1]  
        y_time = y[..., 0:1]  
        x_space = x[..., 1:]  
        y_space = y[..., 1:]  
        return -x_time @ y_time.transpose(-1, -2) + x_space @ y_space.transpose(-1, -2)
    def forward(self, x, edge_index, edge_attr):
        N = x.size(0)
        row, col = edge_index
        if self.use_att:
            query = self.query_linear(x)  
            key = self.key_linear(x)      
            query_edges = query[row]  
            key_edges = key[col]      
            att_scores = 2 + 2 * self.lorentz_inner(query_edges, key_edges).squeeze(-1)  
            att_scores = att_scores / self.scale + self.bias
            att_weights = torch.sigmoid(att_scores) * edge_attr  
            support = torch.zeros_like(x)
            support.index_add_(0, row, att_weights.unsqueeze(-1) * x[col])
        else:
            support = torch.zeros_like(x)
            support.index_add_(0, row, edge_attr.unsqueeze(-1) * x[col])
        denom = (-self.lorentz_inner(support, support)).abs().clamp_min(1e-8).sqrt()
        output = support / denom
        return output
class LorentzGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1,
                 use_att=False, use_bias=True):
        super().__init__()
        self.linear = LorentzLinear(in_features, out_features,
                                    bias=use_bias, dropout=dropout)
        self.agg = LorentzAgg(out_features, dropout=dropout, use_att=use_att)
    def forward(self, x, edge_index, edge_attr):
        h = self.linear(x)
        h = self.agg(h, edge_index, edge_attr)
        return h
class FullyHyperbolicKnnGNN(nn.Module):
    def __init__(self, encoding_size, embedding_size, usePE, useSI, dataset,
                 alpha1, alpha2, beta=1.0, use_att=False):
        super().__init__()
        self.usePE = usePE
        self.useSI = useSI
        self.dataset = dataset
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.use_att = use_att
        if usePE:
            anchor_dim = {'beijing': 98, 'tdrive': 112, 'porto': 162}.get(dataset, 98)
            self.pe_linear = LorentzLinear(encoding_size + anchor_dim + 1, encoding_size)
        self.conv1 = LorentzGraphConvolution(encoding_size + 1, embedding_size,
                                            dropout=0.3, use_att=use_att)
        self.conv2 = LorentzGraphConvolution(encoding_size + 1, embedding_size,
                                            dropout=0.3, use_att=use_att)
        self.conv3 = LorentzGraphConvolution(embedding_size + 1, embedding_size,
                                            dropout=0.3, use_att=use_att)
        self.conv4 = LorentzGraphConvolution(embedding_size + 1, embedding_size,
                                            dropout=0.3, use_att=use_att)
    def cosh_projection(self, x, beta=1.0):
        sqrt_beta = math.sqrt(beta)
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        x_hat = x / norm
        r = norm / sqrt_beta
        x0 = sqrt_beta * torch.cosh(r)
        x_rest = sqrt_beta * torch.sinh(r) * x_hat
        return torch.cat([x0, x_rest], dim=-1)
    def logmap_spatial(self, x_hyp, beta=1.0):
        sqrt_beta = math.sqrt(beta)
        x0 = x_hyp[..., 0:1]
        x_rest = x_hyp[..., 1:]
        alpha = (x0 / sqrt_beta).clamp(min=1.0 + 1e-6)
        scale = sqrt_beta * torch.acosh(alpha) / (alpha ** 2 - 1).clamp(min=1e-10).sqrt()
        return x_rest * scale
    def lorentz_midpoint(self, x0_hyp, x1_hyp):
        weighted = self.alpha1 * x0_hyp + self.alpha2 * x1_hyp
        inner = -weighted[..., 0:1] ** 2 + (weighted[..., 1:] ** 2).sum(dim=-1, keepdim=True)
        denom = (-inner).abs().clamp_min(1e-8).sqrt()
        return weighted / denom
    def forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x = data[0].x
        edge_index_l0, edge_weight_l0 = data[0].edge_index, data[0].edge_attr
        edge_index_l1, edge_weight_l1 = data[1].edge_index, data[1].edge_attr
        if self.usePE and d2an is not None:
            x_combined = torch.cat([x, d2an], dim=-1)  
            x_hyp = self.cosh_projection(x_combined, self.beta)  
            x_hyp = self.pe_linear(x_hyp)  
        else:
            x_hyp = self.cosh_projection(x, self.beta)  
        x0_hyp = F.dropout(self.conv1(x_hyp, edge_index_l0, edge_weight_l0),
                          p=0.3, training=self.training)
        x1_hyp = F.dropout(self.conv2(x_hyp, edge_index_l1, edge_weight_l1),
                          p=0.3, training=self.training)
        x_hyp = self.lorentz_midpoint(x0_hyp, x1_hyp)
        x0_hyp = F.dropout(self.conv3(x_hyp, edge_index_l0, edge_weight_l0),
                          p=0.3, training=self.training)
        x1_hyp = F.dropout(self.conv4(x_hyp, edge_index_l1, edge_weight_l1),
                          p=0.3, training=self.training)
        x_hyp = self.lorentz_midpoint(x0_hyp, x1_hyp)
        return self.logmap_spatial(x_hyp, self.beta)
    def noSI_forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x = data.x
        edge_index_l0, edge_weight_l0 = data.edge_index, data.edge_attr
        if self.usePE and d2an is not None:
            x_combined = torch.cat([x, d2an], dim=-1)
            x_hyp = self.cosh_projection(x_combined, self.beta)
            x_hyp = self.pe_linear(x_hyp)
        else:
            x_hyp = self.cosh_projection(x, self.beta)
        x_hyp = F.dropout(self.conv1(x_hyp, edge_index_l0, edge_weight_l0),
                         p=0.3, training=self.training)
        x_hyp = F.dropout(self.conv3(x_hyp, edge_index_l0, edge_weight_l0),
                         p=0.3, training=self.training)
        return self.logmap_spatial(x_hyp, self.beta)
