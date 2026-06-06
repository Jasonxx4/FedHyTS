import torch
from torch.nn import Module
from setting import SetParameter
import torch.nn.functional as F
config = SetParameter()
def _dynamic_fusion_dist(emb_a, emb_b, beta=1.0):
    d_eu = torch.norm(emb_a['emb_eucl'] - emb_b['emb_eucl'], p=2, dim=-1)
    a_hyp = emb_a['emb_hyp']   
    b_hyp = emb_b['emb_hyp']   
    inner = -a_hyp[:, 0] * b_hyp[:, 0] + (a_hyp[:, 1:] * b_hyp[:, 1:]).sum(dim=-1)
    d_lo = torch.abs(inner) - beta
    dot_lo = (emb_a['v_lo'] * emb_b['v_lo']).sum(dim=-1)   
    dot_eu = (emb_a['v_eu'] * emb_b['v_eu']).sum(dim=-1)   
    alpha_lo = dot_lo / (dot_lo + dot_eu + 1e-10)           
    d_fu = alpha_lo * d_lo + (1.0 - alpha_lo) * d_eu
    return d_fu
class SpaLossFun(Module):
    def __init__(self, train_batch, distance_type, hyp_beta=1.0):
        super(SpaLossFun, self).__init__()
        self.train_batch = train_batch
        self.distance_type = distance_type
        self.hyp_beta = hyp_beta
    def forward(self, embedding_a, embedding_p, embedding_n, pos_dis, neg_dis, device):
        D_ap = torch.exp(-pos_dis).to(device)
        D_an = torch.exp(-neg_dis).to(device)
        if isinstance(embedding_a, dict):
            v_ap = torch.exp(-_dynamic_fusion_dist(embedding_a, embedding_p, self.hyp_beta))
            v_an = torch.exp(-_dynamic_fusion_dist(embedding_a, embedding_n, self.hyp_beta))
        else:
            v_ap = torch.exp(-(torch.norm(embedding_a - embedding_p, p=2, dim=-1)))
            v_an = torch.exp(-(torch.norm(embedding_a - embedding_n, p=2, dim=-1)))
        loss_entire_ap = (D_ap - v_ap) ** 2
        loss_entire_an = (D_an - v_an) ** 2
        loss = loss_entire_ap + loss_entire_an + (D_ap > D_an) * (F.relu(v_an - v_ap)) ** 2
        loss_mean = loss.mean(dim=-1)
        return loss_mean
