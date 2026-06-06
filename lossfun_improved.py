import torch
from torch.nn import Module
from setting import SetParameter
import torch.nn.functional as F
config = SetParameter()
def _dynamic_fusion_dist_improved(emb_a, emb_b, beta=1.0, eps=1e-8):
    d_eu = torch.norm(emb_a['emb_eucl'] - emb_b['emb_eucl'], p=2, dim=-1)
    a_hyp = emb_a['emb_hyp']   
    b_hyp = emb_b['emb_hyp']   
    inner = -a_hyp[:, 0] * b_hyp[:, 0] + (a_hyp[:, 1:] * b_hyp[:, 1:]).sum(dim=-1)
    inner_normalized = (-inner / beta).clamp(min=1.0 + eps)
    d_lo = torch.sqrt(torch.tensor(beta)) * torch.acosh(inner_normalized)
    v_lo_norm = F.normalize(emb_a['v_lo'], p=2, dim=-1)
    v_eu_norm = F.normalize(emb_a['v_eu'], p=2, dim=-1)
    v_lo_b_norm = F.normalize(emb_b['v_lo'], p=2, dim=-1)
    v_eu_b_norm = F.normalize(emb_b['v_eu'], p=2, dim=-1)
    dot_lo = (v_lo_norm * v_lo_b_norm).sum(dim=-1).clamp(min=0)  
    dot_eu = (v_eu_norm * v_eu_b_norm).sum(dim=-1).clamp(min=0)  
    temperature = 1.0
    alpha_lo = 1.0
    d_fu = alpha_lo * d_lo + (1.0 - alpha_lo) * d_eu
    return d_fu, alpha_lo  
class ImprovedSpaLossFun(Module):
    def __init__(self, train_batch, distance_type, hyp_beta=1.0,
                 margin=0.5, use_contrastive=False, lambda_reg=0.01):
        super(ImprovedSpaLossFun, self).__init__()
        self.train_batch = train_batch
        self.distance_type = distance_type
        self.hyp_beta = hyp_beta
        self.margin = margin  
        self.use_contrastive = use_contrastive
        self.lambda_reg = lambda_reg  
    def forward(self, embedding_a, embedding_p, embedding_n, pos_dis, neg_dis, device):
        D_ap = torch.exp(-pos_dis).to(device)
        D_an = torch.exp(-neg_dis).to(device)
        if isinstance(embedding_a, dict):
            v_ap, alpha_ap = _dynamic_fusion_dist_improved(embedding_a, embedding_p, self.hyp_beta)
            v_an, alpha_an = _dynamic_fusion_dist_improved(embedding_a, embedding_n, self.hyp_beta)
            v_ap = torch.exp(-v_ap)
            v_an = torch.exp(-v_an)
        else:
            v_ap = torch.exp(-(torch.norm(embedding_a - embedding_p, p=2, dim=-1)))
            v_an = torch.exp(-(torch.norm(embedding_a - embedding_n, p=2, dim=-1)))
            alpha_ap = None
            alpha_an = None
        loss_ap = (D_ap - v_ap) ** 2
        loss_an = (D_an - v_an) ** 2
        triplet_loss = F.relu(v_ap - v_an + self.margin)
        ranking_loss = (D_ap > D_an).float() * F.relu(v_an - v_ap) ** 2
        loss = loss_ap + loss_an + triplet_loss + ranking_loss
        if self.use_contrastive and isinstance(embedding_a, dict):
            contrastive_loss = self._contrastive_loss(
                embedding_a, embedding_p, embedding_n, device
            )
            loss = loss + 0.1 * contrastive_loss
        loss_mean = loss.mean(dim=-1)
        return loss_mean
    def _contrastive_loss(self, emb_a, emb_p, emb_n, device, temperature=0.07):
        batch_size = emb_a['emb_eucl'].size(0)
        pos_sim, _ = _dynamic_fusion_dist_improved(emb_a, emb_p, self.hyp_beta)
        pos_sim = torch.exp(-pos_sim / temperature)
        neg_sim, _ = _dynamic_fusion_dist_improved(emb_a, emb_n, self.hyp_beta)
        neg_sim = torch.exp(-neg_sim / temperature)
        a_hyp = emb_a['emb_hyp']  
        inner_matrix = -a_hyp[:, 0:1] @ a_hyp[:, 0:1].T + a_hyp[:, 1:] @ a_hyp[:, 1:].T
        inner_matrix = inner_matrix.fill_diagonal_(float('-inf'))
        inner_normalized = (-inner_matrix / self.hyp_beta).clamp(min=1.0 + 1e-8)
        d_matrix = torch.sqrt(torch.tensor(self.hyp_beta)) * torch.acosh(inner_normalized)
        batch_neg_sim = torch.exp(-d_matrix / temperature).sum(dim=1)
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + batch_neg_sim + 1e-8))
        return loss.mean()
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
def _dynamic_fusion_dist(emb_a, emb_b, beta=1.0):
    d_eu = torch.norm(emb_a['emb_eucl'] - emb_b['emb_eucl'], p=2, dim=-1)
    a_hyp = emb_a['emb_hyp']
    b_hyp = emb_b['emb_hyp']
    inner = -a_hyp[:, 0] * b_hyp[:, 0] + (a_hyp[:, 1:] * b_hyp[:, 1:]).sum(dim=-1)
    d_lo = torch.abs(inner) - beta
    dot_lo = (emb_a['v_lo'] * emb_b['v_lo']).sum(dim=-1)
    dot_eu = (emb_a['v_eu'] * emb_b['v_eu']).sum(dim=-1)
    alpha_lo = dot_lo / (dot_lo + dot_eu + 1e-10)
    d_fu = d_lo
    return d_fu
