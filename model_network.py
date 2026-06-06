import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import logging
try:
    from lorentz_layers import FullyHyperbolicKnnGNN, LorentzGraphConvolution, LorentzLinear
    FULLY_HYPERBOLIC_AVAILABLE = True
except ImportError:
    FULLY_HYPERBOLIC_AVAILABLE = False
    logging.warning("lorentz_layers not found, using standard hyperbolic implementation")
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, device, dropout: float = 0.1, max_len: int = 3700):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)  
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float, device):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, device, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)
        self.init_weights()
    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    def forward(self, src, src_mask):
        src = src * math.sqrt(self.d_model)  
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        output = self.decoder(output)
        return output
class PosKnnGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, usePE, useSI, dataset):
        super().__init__(aggr='add')  
        self.usePE = usePE
        self.useSI = useSI
        logging.info('usePE: ' + str(self.usePE) + ', useSI: ' + str(self.useSI))
        self.dataset = dataset
        if dataset == 'beijing':
            self.nodeLin = torch.nn.Linear(in_channels + 98, in_channels, bias=False)
        elif dataset == 'tdrive':
            self.nodeLin = torch.nn.Linear(in_channels + 112, in_channels, bias=False)
        elif dataset == 'porto':
            self.nodeLin = torch.nn.Linear(in_channels + 162, in_channels, bias=False)
        self.lin1 = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.lin2 = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
    def reset_parameters(self):
        self.nodeLin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bias.data.zero_()
    def forward(self, x, input_edge_index, input_edge_attr, d2an, firstLayer):
        edge_index, edge_attr = add_self_loops(input_edge_index, input_edge_attr, num_nodes=x.size(0), fill_value=1.0)
        if firstLayer and self.usePE:
            combined_input = torch.cat((x, d2an), dim=1)
            x = self.nodeLin(combined_input)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        edge_inv_sqrt = edge_attr.pow(-0.5)
        edge_inv_sqrt[edge_inv_sqrt == float('inf')] = 0
        edge_inv_sqrt[edge_inv_sqrt > 1.0] = 1.0
        edge_norm = edge_inv_sqrt
        out = self.propagate(edge_index, x=x, norm=[deg_norm, edge_norm])
        return out
    def message(self, x_j, norm):
        if self.useSI:
            return norm[0].view(-1, 1) * (self.lin1(x_j)) + norm[1].view(-1, 1) * (self.lin2(x_j))
        else:
            return norm.view(-1, 1) * (self.lin1(x_j))
    def gcn_forward(self, x, input_edge_index, input_edge_attr, d2an, firstLayer):
        edge_index, edge_attr = add_self_loops(input_edge_index, input_edge_attr, num_nodes=x.size(0), fill_value=1.0)
        if firstLayer and self.usePE:
            combined_input = torch.cat((x, d2an), dim=1)
            x = self.nodeLin(combined_input)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        edge_inv_sqrt = edge_attr.pow(-0.5)
        edge_inv_sqrt[edge_inv_sqrt == float('inf')] = 0
        edge_inv_sqrt[edge_inv_sqrt > 1.0] = 1.0
        edge_norm = edge_inv_sqrt
        out = self.propagate(edge_index, x=x, norm=deg_norm)
        return out
class SimpleGCN(nn.Module):
    def __init__(self, encoding_size, embedding_size, usePE, dataset):
        super(SimpleGCN, self).__init__()
        self.usePE = usePE
        self.dataset = dataset
        anchor_dim = {'beijing': 98, 'tdrive': 112, 'porto': 162}.get(dataset, 98)
        if usePE:
            self.pe_lin = nn.Linear(encoding_size + anchor_dim, encoding_size, bias=False)
        self.gc1 = nn.Linear(encoding_size, embedding_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(embedding_size))
        nn.init.xavier_uniform_(self.gc1.weight)
        nn.init.zeros_(self.bias)
    def forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.usePE and hasattr(self, 'pe_lin'):
            x = self.pe_lin(torch.cat([x, d2an], dim=1))
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0), fill_value=1.0)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        edge_inv_sqrt = edge_attr.pow(-0.5)
        edge_inv_sqrt[edge_inv_sqrt == float('inf')] = 0
        edge_norm = edge_inv_sqrt
        msg = deg_norm.view(-1, 1) * edge_norm.view(-1, 1) * self.gc1(x[col])
        aggr = torch.zeros(x.size(0), msg.size(-1), device=x.device, dtype=x.dtype)
        aggr.index_add_(0, row, msg)
        return F.relu(aggr + self.bias)
    def noSI_forward(self, input_data):
        return self.forward(input_data)
class SimpleGAT(nn.Module):
    def __init__(self, encoding_size, embedding_size, usePE, dataset, heads=4):
        super(SimpleGAT, self).__init__()
        self.usePE = usePE
        self.dataset = dataset
        self.heads = heads
        self.embedding_size = embedding_size
        anchor_dim = {'beijing': 98, 'tdrive': 112, 'porto': 162}.get(dataset, 98)
        if usePE:
            self.pe_lin = nn.Linear(encoding_size + anchor_dim, encoding_size, bias=False)
        head_dim = embedding_size // heads
        self.query = nn.Linear(encoding_size, embedding_size, bias=False)
        self.key = nn.Linear(encoding_size, embedding_size, bias=False)
        self.value = nn.Linear(encoding_size, embedding_size, bias=False)
        self.out_proj = nn.Linear(embedding_size, embedding_size, bias=False)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    def forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.usePE and hasattr(self, 'pe_lin'):
            x = self.pe_lin(torch.cat([x, d2an], dim=1))
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0), fill_value=1.0)
        num_nodes = x.size(0)
        row, col = edge_index
        q = self.query(x)  
        k = self.key(x)
        v = self.value(x)
        q_row = q[row]  
        k_col = k[col]  
        att_raw = (q_row * k_col).sum(dim=-1)  
        att_raw = att_raw * edge_attr  
        deg = degree(col, num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        att_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  
        att_scores = att_raw * att_norm  
        att_scores_max = att_scores.max()
        att_scores_exp = torch.exp(att_scores - att_scores_max)
        denom = torch.zeros(num_nodes, device=x.device)
        denom.scatter_add_(0, row, att_scores_exp)
        att_weights = att_scores_exp / (denom[row] + 1e-8)  
        v_col = v[col]  
        aggr = torch.zeros(num_nodes, self.embedding_size, device=x.device, dtype=x.dtype)
        aggr = aggr.index_add_(0, row, v_col * att_weights.unsqueeze(-1))  
        out = self.out_proj(aggr)
        return F.relu(out)
    def noSI_forward(self, input_data):
        return self.forward(input_data)
class SimpleRNN(nn.Module):
    def __init__(self, encoding_size, embedding_size, usePE, dataset, rnn_type='LSTM'):
        super(SimpleRNN, self).__init__()
        self.usePE = usePE
        self.dataset = dataset
        self.rnn_type = rnn_type
        anchor_dim = {'beijing': 98, 'tdrive': 112, 'porto': 162}.get(dataset, 98)
        if usePE:
            self.pe_lin = nn.Linear(encoding_size + anchor_dim, encoding_size, bias=False)
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(encoding_size, embedding_size, num_layers=1, batch_first=True, bidirectional=False)
        else:
            self.rnn = nn.LSTM(encoding_size, embedding_size, num_layers=1, batch_first=True, bidirectional=False)
    def forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.usePE and hasattr(self, 'pe_lin'):
            x = self.pe_lin(torch.cat([x, d2an], dim=1))
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0), fill_value=1.0)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_norm = deg_inv_sqrt[row]
        aggr = torch.zeros(x.size(0), x.size(1), device=x.device, dtype=x.dtype)
        aggr.index_add_(0, row, deg_norm.view(-1, 1) * x[col])
        aggr = aggr / deg.view(-1, 1).clamp(min=1)  
        aggr_expanded = aggr.unsqueeze(1)  
        if self.rnn_type == 'GRU':
            _, h_n = self.rnn(aggr_expanded)
            node_embeddings = h_n.squeeze(0)
        else:
            _, (h_n, _) = self.rnn(aggr_expanded)
            node_embeddings = h_n.squeeze(0)
        return node_embeddings
    def noSI_forward(self, input_data):
        return self.forward(input_data)
class SimpleTransformer(nn.Module):
    def __init__(self, encoding_size, embedding_size, usePE, dataset):
        super(SimpleTransformer, self).__init__()
        self.usePE = usePE
        self.dataset = dataset
        anchor_dim = {'beijing': 98, 'tdrive': 112, 'porto': 162}.get(dataset, 98)
        if usePE:
            self.pe_lin = nn.Linear(encoding_size + anchor_dim, encoding_size, bias=False)
        self.query = nn.Linear(encoding_size, encoding_size, bias=False)
        self.key = nn.Linear(encoding_size, encoding_size, bias=False)
        self.value = nn.Linear(encoding_size, encoding_size, bias=False)
        self.out_proj = nn.Linear(encoding_size, embedding_size, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    def forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.usePE and hasattr(self, 'pe_lin'):
            x = self.pe_lin(torch.cat([x, d2an], dim=1))
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0), fill_value=1.0)
        num_nodes = x.size(0)
        row, col = edge_index
        q = self.query(x)  
        k = self.key(x)    
        v = self.value(x)  
        att_raw = (q[row] * k[col]).sum(dim=-1) * edge_attr  
        deg = degree(col, num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        att_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  
        att_scores = att_raw * att_norm  
        att_scores_exp = torch.exp(att_scores)
        denom = torch.zeros(num_nodes, device=x.device)
        denom.scatter_add_(0, row, att_scores_exp)
        att_weights = att_scores_exp / (denom[row] + 1e-8)  
        aggr = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
        aggr.index_add_(0, row, v[col] * att_weights.view(-1, 1))
        out = self.out_proj(aggr)  
        return F.relu(out)
    def noSI_forward(self, input_data):
        return self.forward(input_data)
class SimpleGTS(nn.Module):
    def __init__(self, encoding_size, embedding_size, usePE, dataset):
        super(SimpleGTS, self).__init__()
        self.usePE = usePE
        self.dataset = dataset
        anchor_dim = {'beijing': 98, 'tdrive': 112, 'porto': 162}.get(dataset, 98)
        if usePE:
            self.pe_lin = nn.Linear(encoding_size + anchor_dim, encoding_size, bias=False)
        self.q_lin = nn.Linear(encoding_size, encoding_size, bias=False)
        self.k_lin = nn.Linear(encoding_size, encoding_size, bias=False)
        self.v_lin = nn.Linear(encoding_size, encoding_size, bias=False)
        self.edge_proj = nn.Linear(1, encoding_size, bias=False)
        self.out_proj = nn.Linear(encoding_size, embedding_size, bias=False)
        self.rw_bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.q_lin.weight)
        nn.init.xavier_uniform_(self.k_lin.weight)
        nn.init.xavier_uniform_(self.v_lin.weight)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    def forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.usePE and hasattr(self, 'pe_lin'):
            x = self.pe_lin(torch.cat([x, d2an], dim=1))
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0), fill_value=1.0)
        num_nodes = x.size(0)
        row, col = edge_index
        q = self.q_lin(x)  
        k = self.k_lin(x)  
        v = self.v_lin(x)  
        edge_emb = self.edge_proj(edge_attr.view(-1, 1))  
        k_enhanced = k[col] + self.rw_bias * edge_emb  
        att_scores = (q[row] * k_enhanced).sum(dim=-1)  
        deg = degree(col, num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        att_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]  
        att_scores = att_scores * att_norm
        att_scores_exp = torch.exp(att_scores - att_scores.max())
        denom = torch.zeros(num_nodes, device=x.device)
        denom.scatter_add_(0, row, att_scores_exp)
        att_weights = att_scores_exp / (denom[row] + 1e-8)  
        v_enhanced = v[col] + self.rw_bias * edge_emb  
        aggr = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
        aggr.index_add_(0, row, v_enhanced * att_weights.view(-1, 1))
        out = self.out_proj(aggr)
        return F.relu(out)
    def noSI_forward(self, input_data):
        return self.forward(input_data)
class HierarchicalFullyHyperbolicGNN(nn.Module):
    def __init__(self, encoding_size, embedding_size, usePE, dataset, beta=1.0, use_att=False):
        super().__init__()
        self.usePE = usePE
        self.dataset = dataset
        self.beta = beta
        self.use_att = use_att
        anchor_dim = {'beijing': 98, 'tdrive': 112, 'porto': 162}.get(dataset, 98)
        if usePE:
            self.pe_lin = LorentzLinear(encoding_size + anchor_dim + 1, encoding_size)
        self.layer1_encoder = LorentzGraphConvolution(encoding_size + 1, embedding_size, dropout=0.3, use_att=use_att)
        self.layer2_encoder = LorentzGraphConvolution(embedding_size + 1, embedding_size, dropout=0.3, use_att=use_att)
        self.layer3_encoder = LorentzGraphConvolution(embedding_size + 1, embedding_size, dropout=0.3, use_att=use_att)
        self.layer4_encoder = LorentzGraphConvolution(embedding_size + 1, embedding_size, dropout=0.3, use_att=use_att)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.Sigmoid()
        )
        self.num_virt2 = 4   
        self.num_virt3 = 3   
        self.num_virt4 = 2   
        self.virtual_anchor_k = 64
    def cosh_projection(self, x, beta=1.0):
        sqrt_beta = math.sqrt(beta)
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-10)
        x_hat = x / norm
        r = norm / sqrt_beta
        x0 = sqrt_beta * torch.cosh(r)
        x_rest = sqrt_beta * torch.sinh(r) * x_hat
        return torch.cat([x0, x_rest], dim=-1)
    def lorentz_normalize(self, x):
        x_space = x[..., 1:]
        space_norm_sq = (x_space * x_space).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        x0 = (1.0 + space_norm_sq).sqrt()
        return torch.cat([x0, x_space], dim=-1)
    def logmap_spatial(self, x_hyp, beta=1.0):
        sqrt_beta = math.sqrt(beta)
        x0 = x_hyp[..., 0:1]
        x_rest = x_hyp[..., 1:]
        alpha = (x0 / sqrt_beta).clamp(min=1.0 + 1e-6)
        scale = sqrt_beta * torch.acosh(alpha) / (alpha ** 2 - 1).clamp(min=1e-10).sqrt()
        return x_rest * scale
    def _make_virtual_nodes(self, ref_x, num_virtuals, device):
        mean_feat = ref_x.mean(dim=0, keepdim=True)        
        mean_feat = self.lorentz_normalize(mean_feat)       
        virtual_nodes = mean_feat.expand(num_virtuals, -1).clone()
        noise = torch.randn(num_virtuals, ref_x.size(1) - 1, device=device) * 0.01
        virtual_nodes[:, 1:] = virtual_nodes[:, 1:] + noise
        virtual_nodes = self.lorentz_normalize(virtual_nodes)  
        return virtual_nodes
    def _build_edges_virt_to_lower(self, num_lower, num_virtuals, device, k=None):
        if k is None or k >= num_lower:
            sampled = list(range(num_lower))
        else:
            step = max(1, num_lower // k)
            sampled = list(range(0, num_lower, step))[:k]
        src, dst, attrs = [], [], []
        for v_offset in range(num_virtuals):
            v_idx = num_lower + v_offset
            for r_idx in sampled:
                src.append(v_idx); dst.append(r_idx); attrs.append(1.0)
                src.append(r_idx); dst.append(v_idx); attrs.append(1.0)
        edge_index = torch.tensor([src, dst], device=device, dtype=torch.long)
        edge_attr = torch.tensor(attrs, device=device, dtype=torch.float)
        return edge_index, edge_attr
    def forward(self, input_data, d2an=None):
        data = input_data[0]
        if isinstance(data, list):
            x = data[0].x
            edge_index = data[0].edge_index
            edge_attr = data[0].edge_attr
        else:
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
        d2an = input_data[1] if len(input_data) > 1 else None
        num_nodes = x.size(0)
        device = x.device
        if self.usePE and d2an is not None:
            x_combined = torch.cat([x, d2an], dim=-1)
            x_hyp = self.pe_lin(self.cosh_projection(x_combined, self.beta))
        else:
            x_hyp = self.cosh_projection(x, self.beta)   
        x1_out = F.dropout(
            self.layer1_encoder(x_hyp, edge_index, edge_attr),
            p=0.3, training=self.training
        )  
        virt2 = self._make_virtual_nodes(x1_out, self.num_virt2, device)  
        edge2, attr2 = self._build_edges_virt_to_lower(
            num_nodes, self.num_virt2, device, k=self.virtual_anchor_k
        )
        x2_input = torch.cat([x1_out, virt2], dim=0)   
        x2_out = F.dropout(
            self.layer2_encoder(x2_input, edge2, attr2),
            p=0.3, training=self.training
        )  
        x2_virt_out = x2_out[num_nodes:]                
        virt3 = self._make_virtual_nodes(x2_virt_out, self.num_virt3, device)  
        edge3, attr3 = self._build_edges_virt_to_lower(
            self.num_virt2, self.num_virt3, device, k=None
        )
        x3_input = torch.cat([x2_virt_out, virt3], dim=0)  
        x3_out = F.dropout(
            self.layer3_encoder(x3_input, edge3, attr3),
            p=0.3, training=self.training
        )  
        x3_virt_out = x3_out[self.num_virt2:]               
        virt4 = self._make_virtual_nodes(x3_virt_out, self.num_virt4, device)  
        edge4, attr4 = self._build_edges_virt_to_lower(
            self.num_virt3, self.num_virt4, device, k=None
        )
        x4_input = torch.cat([x3_virt_out, virt4], dim=0)  
        x4_out = F.dropout(
            self.layer4_encoder(x4_input, edge4, attr4),
            p=0.3, training=self.training
        )  
        x4_virt_out = x4_out[self.num_virt3:]               
        x1_tan = self.logmap_spatial(x1_out, self.beta)              
        x4_tan = self.logmap_spatial(x4_virt_out, self.beta)         
        x4_global = x4_tan.mean(dim=0, keepdim=True).expand(num_nodes, -1)  
        gate = self.fusion_gate(torch.cat([x1_tan, x4_global], dim=-1))  
        h_fused = gate * x1_tan + (1 - gate) * x4_global                 
        return h_fused
    def noSI_forward(self, input_data, d2an=None):
        return self.forward(input_data, d2an)
class KnnGNN(nn.Module):
    def __init__(self, encoding_size, embedding_size, usePE, useSI, dataset, alpha1, alpha2):
        super(KnnGNN, self).__init__()
        self.usePE = usePE
        self.useSI = useSI
        self.dataset = dataset
        self.posconv1 = PosKnnGNNLayer(encoding_size, embedding_size, self.usePE, self.useSI, self.dataset)
        self.posconv2 = PosKnnGNNLayer(encoding_size, embedding_size, self.usePE, self.useSI, self.dataset)
        self.posconv3 = PosKnnGNNLayer(embedding_size, embedding_size, self.usePE, self.useSI, self.dataset)
        self.posconv4 = PosKnnGNNLayer(embedding_size, embedding_size, self.usePE, self.useSI, self.dataset)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
    def forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x, edge_index_l0, edge_weight_l0 = data[0].x, data[0].edge_index, data[0].edge_attr
        _, edge_index_l1, edge_weight_l1 = data[1].x, data[1].edge_index, data[1].edge_attr
        _, edge_index_l2, edge_weight_l2 = data[2].x, data[2].edge_index, data[2].edge_attr
        x0 = F.relu(self.posconv1(x, edge_index_l0, edge_weight_l0, d2an, True))
        x0 = F.dropout(x0, p=0.3, training=self.training)
        x1 = F.relu(self.posconv2(x, edge_index_l1, edge_weight_l1, d2an, True))
        x1 = F.dropout(x1, p=0.3, training=self.training)
        x = self.alpha1 * x0 + self.alpha2 * x1
        x0 = F.relu(self.posconv3(x, edge_index_l0, edge_weight_l0, d2an, False))
        x0 = F.dropout(x0, p=0.3, training=self.training)
        x1 = F.relu(self.posconv4(x, edge_index_l1, edge_weight_l1, d2an, False))
        x1 = F.dropout(x1, p=0.3, training=self.training)
        x = self.alpha1 * x0 + self.alpha2 * x1
        return x
    def noSI_forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x, edge_index_l0, edge_weight_l0 = data.x, data.edge_index, data.edge_attr
        x0 = F.relu(self.posconv1.gcn_forward(x, edge_index_l0, edge_weight_l0, d2an, True))
        x0 = F.dropout(x0, p=0.3, training=self.training)
        x = x0
        x0 = F.relu(self.posconv3.gcn_forward(x, edge_index_l0, edge_weight_l0, d2an, False))
        x0 = F.dropout(x0, p=0.3, training=self.training)
        x = x0
        return x
def lorentz_inner_product(a, b):
    return -a[..., 0] * b[..., 0] + (a[..., 1:] * b[..., 1:]).sum(dim=-1)
def cosh_projection(x, beta=1.0):
    sqrt_beta = math.sqrt(beta)
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    x_hat = x / norm                              
    r = norm / sqrt_beta                          
    x0 = sqrt_beta * torch.cosh(r)               
    x_rest = sqrt_beta * torch.sinh(r) * x_hat   
    return torch.cat([x0, x_rest], dim=-1)        
def logmap_spatial(x_hyp, beta=1.0):
    sqrt_beta = math.sqrt(beta)
    x0 = x_hyp[..., 0:1]                                
    x_rest = x_hyp[..., 1:]                             
    alpha = (x0 / sqrt_beta).clamp(min=1.0 + 1e-6)
    scale = sqrt_beta * torch.acosh(alpha) / (alpha ** 2 - 1).clamp(min=1e-10).sqrt()
    return x_rest * scale                                
def lorentz_distance(a, b, beta=1.0):
    inner = lorentz_inner_product(a, b)
    return torch.abs(inner) - beta
def hyperbolic_attention_pooling(seq_outputs, w_omega, u_omega, mask, beta=1.0):
    seq_hyp = cosh_projection(seq_outputs, beta)  
    seq_tangent = logmap_spatial(seq_hyp, beta)   
    u = torch.tanh(torch.matmul(seq_tangent, w_omega))
    att = torch.matmul(u, u_omega).squeeze(-1)     
    att = att.masked_fill(mask == True, -1e10)
    att_score = F.softmax(att, dim=1).unsqueeze(2)  
    scored_tangent = seq_tangent * att_score
    out_tangent = torch.sum(scored_tangent, dim=1)  
    out_hyp = cosh_projection(out_tangent, beta)    
    return out_hyp, out_tangent
class FactorVectorLayer(nn.Module):
    def __init__(self, input_dim, factor_dim=16):
        super().__init__()
        hidden_dim = max(factor_dim * 4, 64)
        self.v_lo_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, factor_dim),
            nn.Softplus()
        )
        self.v_eu_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, factor_dim),
            nn.Softplus()
        )
    def forward(self, x):
        return self.v_lo_layer(x), self.v_eu_layer(x)
class HypLorentzGNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, beta, usePE=False, useSI=False, dataset='beijing'):
        super().__init__()
        self.beta = beta
        self.usePE = usePE
        self.useSI = useSI
        if usePE:
            anchor_dim = {'beijing': 98, 'tdrive': 112, 'porto': 162}.get(dataset, 98)
            self.nodeLin = nn.Linear(in_channels + anchor_dim, in_channels, bias=False)
        self.lin1 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin2 = nn.Linear(in_channels, out_channels, bias=False) if useSI else None
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.lin1.weight)
        if self.lin2 is not None:
            nn.init.xavier_uniform_(self.lin2.weight)
        if usePE:
            nn.init.xavier_uniform_(self.nodeLin.weight)
    def forward(self, x_hyp, edge_index, edge_attr, d2an, firstLayer):
        N = x_hyp.size(0)
        v = logmap_spatial(x_hyp, self.beta)          
        if firstLayer and self.usePE and d2an is not None:
            v = self.nodeLin(torch.cat([v, d2an], dim=1))  
        edge_index_sl, edge_attr_sl = add_self_loops(
            edge_index, edge_attr, num_nodes=N, fill_value=1.0)
        row, col = edge_index_sl
        deg = degree(col, N, dtype=v.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        deg_norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        edge_inv_sqrt = edge_attr_sl.pow(-0.5)
        edge_inv_sqrt[edge_inv_sqrt == float('inf')] = 0
        edge_inv_sqrt[edge_inv_sqrt > 1.0] = 1.0
        if self.useSI:
            msg = (deg_norm.view(-1, 1) * self.lin1(v[col])
                   + edge_inv_sqrt.view(-1, 1) * self.lin2(v[col]))
        else:
            msg = deg_norm.view(-1, 1) * self.lin1(v[col])
        aggr = torch.zeros(N, msg.size(-1), device=v.device, dtype=v.dtype)
        aggr.index_add_(0, row, msg)
        aggr = F.relu(aggr + self.bias)
        return cosh_projection(aggr, self.beta)           
class HyperbolicKnnGNN(nn.Module):
    def __init__(self, encoding_size, embedding_size, usePE, useSI, dataset, alpha1, alpha2, beta=1.0):
        super().__init__()
        self.usePE = usePE
        self.useSI = useSI
        self.dataset = dataset
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.hyp_conv1 = HypLorentzGNNLayer(encoding_size, embedding_size, beta, usePE, useSI, dataset)
        self.hyp_conv2 = HypLorentzGNNLayer(encoding_size, embedding_size, beta, usePE, useSI, dataset)
        self.hyp_conv3 = HypLorentzGNNLayer(embedding_size, embedding_size, beta, False, useSI, dataset)
        self.hyp_conv4 = HypLorentzGNNLayer(embedding_size, embedding_size, beta, False, useSI, dataset)
    def _mix_in_tangent(self, x0_hyp, x1_hyp):
        v0 = logmap_spatial(x0_hyp, self.beta)
        v1 = logmap_spatial(x1_hyp, self.beta)
        return cosh_projection(self.alpha1 * v0 + self.alpha2 * v1, self.beta)
    def forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x = data[0].x
        edge_index_l0, edge_weight_l0 = data[0].edge_index, data[0].edge_attr
        edge_index_l1, edge_weight_l1 = data[1].edge_index, data[1].edge_attr
        x_hyp = cosh_projection(x, self.beta)            
        x0_hyp = F.dropout(
            self.hyp_conv1(x_hyp, edge_index_l0, edge_weight_l0, d2an, True),
            p=0.3, training=self.training)
        x1_hyp = F.dropout(
            self.hyp_conv2(x_hyp, edge_index_l1, edge_weight_l1, d2an, True),
            p=0.3, training=self.training)
        x_hyp = self._mix_in_tangent(x0_hyp, x1_hyp)    
        x0_hyp = F.dropout(
            self.hyp_conv3(x_hyp, edge_index_l0, edge_weight_l0, None, False),
            p=0.3, training=self.training)
        x1_hyp = F.dropout(
            self.hyp_conv4(x_hyp, edge_index_l1, edge_weight_l1, None, False),
            p=0.3, training=self.training)
        x_hyp = self._mix_in_tangent(x0_hyp, x1_hyp)    
        return logmap_spatial(x_hyp, self.beta)           
    def noSI_forward(self, input_data):
        data, d2an = input_data[0], input_data[1]
        x = data.x
        edge_index_l0, edge_weight_l0 = data.edge_index, data.edge_attr
        x_hyp = cosh_projection(x, self.beta)
        x_hyp = F.dropout(
            self.hyp_conv1(x_hyp, edge_index_l0, edge_weight_l0, d2an, True),
            p=0.3, training=self.training)
        x_hyp = F.dropout(
            self.hyp_conv3(x_hyp, edge_index_l0, edge_weight_l0, None, False),
            p=0.3, training=self.training)
        return logmap_spatial(x_hyp, self.beta)           
class SMNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, stard_LSTM=False, stard_GRU=False, incell=True, device=0):
        super(SMNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stard_LSTM = stard_LSTM
        self.stard_GRU = stard_GRU
        self.device = device
        self.mlp_ele = torch.nn.Linear(2, int(hidden_size/2))
        self.nonLeaky = torch.nn.LeakyReLU(0.1)
        self.nonTanh = torch.nn.Tanh()
        self.point_pooling = torch.nn.AvgPool1d(10)
        self.seq_model_layer = 1
        self.device = device
        if stard_GRU:
            logging.info('Using GRU for sequence modeling')
            self.t2s_model = torch.nn.GRU(self.input_size, hidden_size,
                                          num_layers=self.seq_model_layer)
        else:
            logging.info('Using LSTM for sequence modeling')
            self.t2s_model = torch.nn.LSTM(self.input_size, hidden_size,
                                            num_layers=self.seq_model_layer)
        self.res_linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.res_linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.res_linear3 = torch.nn.Linear(hidden_size, hidden_size)
    def forward(self, inputs_a):
        input_a, input_len_a = inputs_a  
        if self.stard_GRU:
            outputs_a, hn_a = self.t2s_model(input_a.permute(1, 0, 2))
        else:
            outputs_a, (hn_a, cn_a) = self.t2s_model(input_a.permute(1, 0, 2))
        outputs_ca = F.sigmoid(self.res_linear1(outputs_a)) * F.tanh(self.res_linear2(outputs_a))
        outputs_hata = F.sigmoid(self.res_linear3(outputs_a)) * F.tanh(outputs_ca)
        outputs_fa = outputs_a + outputs_hata
        mask_out_a = []
        for b, v in enumerate(input_len_a):
            mask_out_a.append(outputs_fa[v - 1][b, :].view(1, -1))
        fa_outputs = torch.cat(mask_out_a, dim=0)
        return fa_outputs, outputs_fa
class GraphTrajSimEncoder(nn.Module):
    def __init__(self, feature_size, embedding_size, hidden_size, num_layers, dropout_rate, concat, device, usePE, useSI, useLSTM, useGRU=False, dataset=None, alpha1=1.0, alpha2=1.0, use_hyperbolic=False, hyp_beta=1.0, hyp_factor_dim=16, use_fully_hyperbolic=False, hyp_use_att=False, hyp_gcn_type='hyperbolic', use_time=False, date2vec_size=64):
        super(GraphTrajSimEncoder, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.usePE = usePE
        self.useSI = useSI
        self.useLSTM = useLSTM
        self.useGRU = useGRU
        self.use_hyperbolic = use_hyperbolic
        self.use_time = use_time
        self.dataset = dataset if dataset else 'beijing'
        self.hyp_gcn_type = hyp_gcn_type
        if use_hyperbolic:
            if hyp_gcn_type == 'euclidean':
                logging.info('Ablation: Using Euclidean KnnGNN (hyp_gcn_type=euclidean) for hyperbolic modeling')
                self.graph_embedding = KnnGNN(feature_size, embedding_size, self.usePE,
                                              self.useSI, self.dataset, alpha1, alpha2)
            elif hyp_gcn_type == 'simple_gcn':
                logging.info('Ablation: Using SimpleGCN (hyp_gcn_type=simple_gcn) for hyperbolic modeling')
                self.graph_embedding = SimpleGCN(feature_size, embedding_size, self.usePE, self.dataset)
            elif hyp_gcn_type == 'simple_rnn':
                logging.info('Ablation: Using SimpleRNN (hyp_gcn_type=simple_rnn) for hyperbolic modeling')
                self.graph_embedding = SimpleRNN(feature_size, embedding_size, self.usePE, self.dataset, rnn_type='LSTM')
            elif hyp_gcn_type == 'simple_transformer':
                logging.info('Ablation: Using SimpleTransformer (hyp_gcn_type=simple_transformer) for hyperbolic modeling')
                self.graph_embedding = SimpleTransformer(feature_size, embedding_size, self.usePE, self.dataset)
            elif hyp_gcn_type == 'simple_gts':
                logging.info('Ablation: Using SimpleGTS (hyp_gcn_type=simple_gts) for hyperbolic modeling')
                self.graph_embedding = SimpleGTS(feature_size, embedding_size, self.usePE, self.dataset)
            elif hyp_gcn_type == 'simple_gat':
                logging.info('Ablation: Using SimpleGAT (hyp_gcn_type=simple_gat) for hyperbolic modeling')
                self.graph_embedding = SimpleGAT(feature_size, embedding_size, self.usePE, self.dataset)
            elif hyp_gcn_type == 'hierarchical_hyperbolic':
                logging.info('Ablation: Using HierarchicalFullyHyperbolicGNN (hyp_gcn_type=hierarchical_hyperbolic) for hyperbolic modeling')
                self.graph_embedding = HierarchicalFullyHyperbolicGNN(
                    feature_size, embedding_size, self.usePE, self.dataset,
                    beta=hyp_beta, use_att=hyp_use_att
                )
            elif use_fully_hyperbolic and FULLY_HYPERBOLIC_AVAILABLE:
                logging.info('Using FullyHyperbolicKnnGNN (Fully Hyperbolic NN, beta=%.2f, att=%s)' % (hyp_beta, hyp_use_att))
                self.graph_embedding = FullyHyperbolicKnnGNN(
                    feature_size, embedding_size, self.usePE, self.useSI,
                    self.dataset, alpha1, alpha2, beta=hyp_beta, use_att=hyp_use_att
                )
            else:
                if use_fully_hyperbolic and not FULLY_HYPERBOLIC_AVAILABLE:
                    logging.warning('FullyHyperbolicKnnGNN not available, falling back to HyperbolicKnnGNN')
                logging.info('Using HyperbolicKnnGNN (Lorentz manifold GNN, beta=%.2f)' % hyp_beta)
                self.graph_embedding = HyperbolicKnnGNN(feature_size, embedding_size, self.usePE,
                                                        self.useSI, self.dataset, alpha1, alpha2, beta=hyp_beta)
        else:
            logging.info('Using Euclidean KnnGNN')
            self.graph_embedding = KnnGNN(feature_size, embedding_size, self.usePE,
                                          self.useSI, self.dataset, alpha1, alpha2)
        if use_hyperbolic:
            self.hyp_beta = hyp_beta
            emb_dim = 2 * hidden_size if (useLSTM or useGRU) else hidden_size
            self.factor_layer = FactorVectorLayer(emb_dim, factor_dim=hyp_factor_dim)
            logging.info(f'Lorentz Projection enabled: beta={self.hyp_beta}, factor_dim={hyp_factor_dim}')
        else:
            self.factor_layer = None
        self.trm_encoder = TransformerModel(hidden_size, embedding_size, 4, 512, 1, 0.3, device)
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        logging.info('useLSTM: ' + str(self.useLSTM) + ', useGRU: ' + str(self.useGRU))
        if useLSTM or useGRU:
            self.smn = SMNEncoder(
                input_size=2,
                hidden_size=hidden_size,
                stard_LSTM=useLSTM and not useGRU,  
                stard_GRU=useGRU,
                incell=True,
                device=self.device
            ).to(self.device)
            self.out_linear = nn.Linear(2*hidden_size, 2*hidden_size)
        else:
            self.smn = None
        if use_time:
            self.time_embed = nn.Sequential(
                nn.Linear(date2vec_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, embedding_size)
            )
            logging.info(f'Time embedding enabled: date2vec_size={date2vec_size}')
        if use_fully_hyperbolic or use_hyperbolic:
            for attr in ['posconv1', 'posconv2', 'posconv3', 'posconv4']:
                if hasattr(self, attr):
                    delattr(self, attr)
                    logging.warning(f'Removed legacy attribute: {attr}')
    def obtain_trm_src_mask(self, seq_lengths):
        max_len = int(seq_lengths.max())
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)
        for i, l in enumerate(seq_lengths):
            if l <= max_len:
                mask[i, :l] = 0
        return mask.bool()
    def forward(self, network_data, traj_seqs, coor_seqs, seq_lengths, time_seqs=None):
        seq_lengths = seq_lengths.to(self.device)
        traj_seqs = traj_seqs.to(self.device)
        coor_seqs = coor_seqs.to(self.device)
        if time_seqs is not None:
            time_seqs = time_seqs.to(self.device)
        if self.useSI:
            graph_node_embeddings = self.graph_embedding(network_data)  
        else:
            graph_node_embeddings = self.graph_embedding.noSI_forward(network_data)
        embedded_seq_tensor = graph_node_embeddings[traj_seqs].to(self.device)
        trm_encoder_src_mask = self.obtain_trm_src_mask(seq_lengths)
        trm_outputs = self.trm_encoder(embedded_seq_tensor.transpose(1, 0), trm_encoder_src_mask).transpose(1, 0)
        if self.use_time and time_seqs is not None:
            time_emb = self.time_embed(time_seqs)  
            trm_outputs = trm_outputs + time_emb  
        if self.use_hyperbolic:
            attrn_out_hyp, attrn_out_tangent = hyperbolic_attention_pooling(
                trm_outputs, self.w_omega, self.u_omega, trm_encoder_src_mask, self.hyp_beta
            )
        else:
            u = torch.tanh(torch.matmul(trm_outputs, self.w_omega))
            att = torch.matmul(u, self.u_omega).squeeze()
            att = att.masked_fill(trm_encoder_src_mask == True, -1e10)
            att_score = F.softmax(att, dim=1).unsqueeze(2)
            scored_outputs = trm_outputs * att_score
            attrn_out = torch.sum(scored_outputs, dim=1)
        if self.useLSTM or self.useGRU:
            anchor_embedding, _ = self.smn([coor_seqs, seq_lengths])
            if self.use_hyperbolic:
                anchor_hyp = cosh_projection(anchor_embedding, self.hyp_beta)
                anchor_tangent = logmap_spatial(anchor_hyp, self.hyp_beta)
                out_tangent = torch.cat((attrn_out_tangent, anchor_tangent), dim=-1)
                out_tangent = self.out_linear(out_tangent)
                out_hyp = cosh_projection(out_tangent, self.hyp_beta)
            else:
                out = torch.cat((attrn_out, anchor_embedding), dim=-1)
                out = self.out_linear(out)
        else:
            if self.use_hyperbolic:
                out_tangent = attrn_out_tangent
                out_hyp = attrn_out_hyp
            else:
                out = attrn_out
        if self.use_hyperbolic and self.factor_layer is not None:
            v_lo, v_eu = self.factor_layer(out_tangent)  
            return {'emb_eucl': out_tangent, 'emb_hyp': out_hyp, 'v_lo': v_lo, 'v_eu': v_eu}
        return out
class GraphTrajSTEncoder(nn.Module):
    def __init__(self, feature_size, embedding_size, date2vec_size, hidden_size, num_layers, dropout_rate, concat, device, usePE, useSI, dataset):
        super(GraphTrajSTEncoder, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.usePE = usePE
        self.useSI = useSI
        self.dataset = dataset
        self.graph_embedding = KnnGNN(feature_size, embedding_size, self.usePE, self.useSI, self.dataset)
        self.co_attention = Co_Att(date2vec_size).to(device)
        self.encoder_ST = ST_LSTM(embedding_size+date2vec_size, hidden_size, num_layers, dropout_rate, device)
        self.out_linear = nn.Linear(384, 384)
        self.smn = SMNEncoder(2,
                              128,
                              stard_LSTM=True,
                              incell=True,
                              device=self.device).to(self.device)
    def obtain_trm_src_mask(self, seq_lengths):
        max_len = int(seq_lengths.max())
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)
        for i, l in enumerate(seq_lengths):
            if l <= max_len:
                mask[i, :l] = 0
        return mask.bool()
    def forward(self, network_data, traj_seqs, coor_seqs, time_seqs, seq_lengths):
        traj_seqs = traj_seqs.to(self.device)
        coor_seqs = coor_seqs.to(self.device)
        time_seqs = time_seqs.to(self.device)
        graph_node_embeddings = self.graph_embedding(network_data)
        spa_input = graph_node_embeddings[traj_seqs].to(self.device)
        time_input = time_seqs
        att_s, att_t = self.co_attention(spa_input, time_input)
        st_input = torch.cat((att_s, att_t), dim=2)
        seq_lengths = seq_lengths.to('cpu')
        packed_input = pack_padded_sequence(st_input, seq_lengths, batch_first=True, enforce_sorted=False)
        out = self.encoder_ST(packed_input)
        seq_lengths = seq_lengths.to(self.device)
        anchor_embedding,  outputs_ap = self.smn([coor_seqs, seq_lengths])
        all_out = torch.cat((out, anchor_embedding), dim=-1)
        all_out = self.out_linear(all_out)
        return all_out
class Co_Att(nn.Module):
    def __init__(self, dim):
        super(Co_Att, self).__init__()
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.temperature = dim ** 0.5
        self.FFN = nn.Sequential(
            nn.Linear(dim, int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5), dim),
            nn.Dropout(0.1)
        )
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
    def forward(self, seq_s, seq_t):
        h = torch.stack([seq_s, seq_t], 2)  
        q = self.Wq(h)
        k = self.Wk(h)
        v = self.Wv(h)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = F.softmax(attn, dim=-1)
        attn_h = torch.matmul(attn, v)
        attn_o = self.FFN(attn_h) + attn_h
        attn_o = self.layer_norm(attn_o)
        att_s = attn_o[:, :, 0, :]
        att_t = attn_o[:, :, 1, :]
        return att_s, att_t
class ST_LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, dropout_rate, device):
        super(ST_LSTM, self).__init__()
        self.device = device
        self.bi_lstm = nn.LSTM(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout_rate,
                               bidirectional=True)
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
    def getMask(self, seq_lengths):
        max_len = int(seq_lengths.max())
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)
        for i, l in enumerate(seq_lengths):
            if l < max_len:
                mask[i, l:] = 0
        return mask
    def forward(self, packed_input):
        packed_output, _ = self.bi_lstm(packed_input)  
        outputs, seq_lengths = pad_packed_sequence(packed_output, batch_first=True)
        mask = self.getMask(seq_lengths)
        u = torch.tanh(torch.matmul(outputs, self.w_omega))
        att = torch.matmul(u, self.u_omega).squeeze()
        att = att.masked_fill(mask == 0, -1e10)
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        scored_outputs = outputs * att_score
        out = torch.sum(scored_outputs, dim=1)
        return out
