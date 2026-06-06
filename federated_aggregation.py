import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import os.path as osp
from tqdm import tqdm
import math
from datetime import datetime
import json
import csv
import time
def lorentz_logmap_spatial(x_hyp, beta=1.0):
    sqrt_beta = math.sqrt(beta)
    x0 = x_hyp[..., 0:1]
    x_rest = x_hyp[..., 1:]
    alpha = (x0 / sqrt_beta).clamp(min=1.0 + 1e-6)
    scale = sqrt_beta * torch.acosh(alpha) / (alpha ** 2 - 1).clamp(min=1e-10).sqrt()
    return x_rest * scale
def lorentz_expmap(v, beta=1.0):
    sqrt_beta = math.sqrt(beta)
    norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    v_hat = v / norm
    r = norm / sqrt_beta
    x0 = sqrt_beta * torch.cosh(r)
    x_rest = sqrt_beta * torch.sinh(r) * v_hat
    return torch.cat([x0, x_rest], dim=-1)
def is_hyperbolic_param(param_name):
    hyperbolic_keywords = [
        'hyp_conv',
        'graph_embedding.hyp',
        'graph_embedding.pe_linear',
        'graph_embedding.conv1',
        'graph_embedding.conv2',
        'graph_embedding.conv3',
        'graph_embedding.conv4',
        'graph_embedding.layer1_encoder',
        'graph_embedding.layer2_encoder',
        'graph_embedding.layer3_encoder',
        'graph_embedding.layer4_encoder',
        'graph_embedding.fusion_gate',
        'graph_embedding.pe_lin',
        'factor_layer',
    ]
    return any(kw in param_name for kw in hyperbolic_keywords)
def should_aggregate_param(param_name, personalized_layers=None):
    if personalized_layers is None:
        personalized_layers = []
    for keyword in personalized_layers:
        if keyword in param_name:
            return False
    return True
def lorentz_weighted_mean(param_list, weights, beta=1.0):
    result = torch.zeros_like(param_list[0], dtype=torch.float32)
    for param, weight in zip(param_list, weights):
        result += weight * param.float()
    param_norm = result.norm()
    max_norm = math.sqrt(beta) * 10.0  
    if param_norm > max_norm:
        result = result * (max_norm / (param_norm + 1e-10))
    return result
def compute_gradient_direction(global_weights, client_weights):
    if global_weights is None:
        return None
    client_grads = []
    for cw in client_weights:
        grad = {}
        for key in global_weights:
            if key in cw:
                grad[key] = cw[key].float() - global_weights[key].float()
        client_grads.append(grad)
    return client_grads
def compute_cosine_similarity(grad1, grad2):
    import torch.nn.functional as F
    vectors1 = []
    vectors2 = []
    for key in grad1:
        if key in grad2:
            vectors1.append(grad1[key].flatten().float())
            vectors2.append(grad2[key].flatten().float())
    if len(vectors1) == 0:
        return 0.0
    vec1 = torch.cat(vectors1)
    vec2 = torch.cat(vectors2)
    cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
    return cos_sim.item()
def compute_communication_cost(state_dict):
    total_bytes = 0
    for p in state_dict.values():
        if torch.is_tensor(p):
            total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 * 1024)
def compute_gradient_similarity_weights(global_weights, client_weights, client_sample_counts, tau=0.1):
    if global_weights is None:
        total = sum(client_sample_counts)
        return [count / total for count in client_sample_counts]
    client_grads = compute_gradient_direction(global_weights, client_weights)
    total_samples = sum(client_sample_counts)
    ref_grad = {}
    for key in global_weights:
        ref_grad[key] = torch.zeros_like(global_weights[key], dtype=torch.float32)
        for cw, count in zip(client_weights, client_sample_counts):
            if key in cw:
                ref_grad[key] += cw[key].float() * (count / total_samples)
    ref_vectors = []
    for key in ref_grad:
        ref_vectors.append(ref_grad[key].flatten().float())
    ref_vector = torch.cat(ref_vectors)
    similarities = []
    for grad in client_grads:
        grad_vectors = []
        for key in grad:
            grad_vectors.append(grad[key].flatten().float())
        grad_vector = torch.cat(grad_vectors)
        cos_sim = F.cosine_similarity(grad_vector.unsqueeze(0), ref_vector.unsqueeze(0))
        similarities.append(cos_sim.item())
    base_weights = [count / total_samples for count in client_sample_counts]
    adjusted_weights = []
    for base_w, sim in zip(base_weights, similarities):
        adjusted_w = base_w * math.exp(sim / tau)
        adjusted_weights.append(adjusted_w)
    total = sum(adjusted_weights)
    normalized_weights = [w / total for w in adjusted_weights]
    logging.debug(f"Gradient similarity weights: {[f'{w:.4f}' for w in normalized_weights]}")
    logging.debug(f"Similarities: {[f'{s:.4f}' for s in similarities]}")
    return normalized_weights
class FederatedAggregator:
    def __init__(self, config, num_clients=20, experiment_dir=None, enable_personalization=True,
                 use_gradient_similarity=True, tau=0.1, lambda_align=0):
        self.config = config
        self.num_clients = num_clients
        self.dataset = str(config.dataset)
        self.distance_type = str(config.distance_type)
        self.device = "cuda:" + str(config.cuda)
        self.use_hyperbolic = config.gtraj.get('use_hyperbolic', False)
        self.hyp_beta = config.gtraj.get('hyp_beta', 1.0) if self.use_hyperbolic else 1.0
        if self.use_hyperbolic:
            logging.info(f"Hyperbolic aggregation enabled: beta={self.hyp_beta}")
        self.enable_personalization = enable_personalization
        if self.enable_personalization:
            logging.info("Personalized Federated Learning enabled: factor_layer and attention weights will NOT be aggregated")
        self.use_gradient_similarity = use_gradient_similarity
        self.tau = tau
        if self.use_gradient_similarity:
            logging.info(f"Gradient Similarity aggregation enabled: tau={tau}")
            logging.info("  - tau small (0.01-0.05): sharp weight distribution")
            logging.info("  - tau large (0.5-1.0): close to uniform weights")
        self.lambda_align = lambda_align
        if self.lambda_align > 0:
            logging.info(f"Direction alignment penalty enabled: lambda_align={lambda_align}")
            logging.info("  - Encourages gradient direction to align with moving toward global model")
            logging.info("  - Larger lambda means stronger alignment pressure")
        if experiment_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_dir = f"{self.distance_type}_{timestamp}"
        else:
            self.experiment_dir = experiment_dir
        logging.info(f"Experiment directory: {self.experiment_dir}")
        self.results_csv_path = osp.join('saved_models', self.dataset, self.experiment_dir, 'federated_results.csv')
        self.results_json_path = osp.join('saved_models', self.dataset, self.experiment_dir, 'federated_results.json')
        self.rounds_results = []  
        self._init_results_csv()
    def FedAvg(self, client_weights):
        total = len(client_weights)
        if total == 0:
            return None
        all_keys = set()
        for client_weight in client_weights:
            all_keys.update(client_weight.keys())
        if len(all_keys) == 0:
            logging.error("No parameters to aggregate! Check model architecture.")
            return None
        global_weights = {}
        for key in all_keys:
            clients_with_key = [cw for cw in client_weights if key in cw]
            if len(clients_with_key) == 0:
                continue
            if self.enable_personalization and not should_aggregate_param(key):
                logging.debug(f"Skipping personalized layer: {key}")
                continue
            global_weights[key] = torch.zeros_like(clients_with_key[0][key], dtype=torch.float32)
            if self.use_hyperbolic and is_hyperbolic_param(key):
                logging.debug(f"Hyperbolic aggregation for: {key}")
                weights = [1.0 / len(clients_with_key)] * len(clients_with_key)
                global_weights[key] = lorentz_weighted_mean(
                    [cw[key] for cw in clients_with_key], weights, self.hyp_beta
                )
            else:
                for client_weight in clients_with_key:
                    global_weights[key] += client_weight[key].float()
                global_weights[key] /= len(clients_with_key)
        return global_weights
    def FedAvg_weighted(self, client_weights, client_sample_counts, prev_global_model=None):
        total_samples = sum(client_sample_counts)
        if total_samples == 0:
            return None
        all_keys = set()
        for client_weight in client_weights:
            all_keys.update(client_weight.keys())
        if len(all_keys) == 0:
            logging.error("No parameters to aggregate! Check model architecture.")
            return None
        if self.use_gradient_similarity and prev_global_model is not None:
            logging.info("Using gradient similarity weighting for aggregation")
            adjusted_weights = compute_gradient_similarity_weights(
                prev_global_model, client_weights, client_sample_counts, tau=self.tau
            )
            logging.info(f"Adjusted weights based on gradient similarity: {[f'{w:.4f}' for w in adjusted_weights]}")
        else:
            adjusted_weights = None
        global_weights = {}
        for key in all_keys:
            clients_with_key = [cw for cw in client_weights if key in cw]
            if len(clients_with_key) == 0:
                continue
            if self.enable_personalization and not should_aggregate_param(key):
                logging.debug(f"Skipping personalized layer: {key}")
                continue
            global_weights[key] = torch.zeros_like(clients_with_key[0][key], dtype=torch.float32)
            total_weight = sum(count for cw, count in zip(client_weights, client_sample_counts) if key in cw)
            normalized_weights = [count / total_weight for cw, count in zip(client_weights, client_sample_counts) if key in cw]
            if adjusted_weights is not None:
                final_weights = []
                for i, (cw, count) in enumerate(zip(client_weights, client_sample_counts)):
                    if key in cw:
                        final_weights.append(adjusted_weights[i])
                total_final = sum(final_weights)
                final_weights = [w / total_final for w in final_weights]
            else:
                final_weights = normalized_weights
            if self.use_hyperbolic and is_hyperbolic_param(key):
                logging.debug(f"Hyperbolic weighted aggregation for: {key}")
                global_weights[key] = lorentz_weighted_mean(
                    [cw[key] for cw in clients_with_key], final_weights, self.hyp_beta
                )
            else:
                for client_weight, weight in zip(clients_with_key, final_weights):
                    global_weights[key] += client_weight[key].float() * weight
        return global_weights
    def aggregate(self, client_weights, client_sample_counts=None, method='FedAvg'):
        if method == 'FedAvg':
            return self.FedAvg(client_weights)
        elif method == 'FedAvg_weighted':
            return self.FedAvg_weighted(client_weights, client_sample_counts)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    def save_global_model(self, global_weights, round_num):
        save_dir = osp.join('saved_models', self.dataset, self.experiment_dir, 'global_model')
        os.makedirs(save_dir, exist_ok=True)
        save_path = osp.join(save_dir, f'global_round_{round_num}.pt')
        torch.save(global_weights, save_path)
        logging.info(f"Global model saved to: {save_path}")
        return save_path
    def load_global_model(self, round_num):
        save_path = osp.join('saved_models', self.dataset, self.experiment_dir,
                            'global_model', f'global_round_{round_num}.pt')
        if os.path.exists(save_path):
            return torch.load(save_path)
        return None
    CSV_FIELDS = ['round', 'loss', 'lr', 'hr10', 'hr50', 'r10@50', 'r1@1', 'r1@10', 'r1@50',
                  'training_time', 'inference_time_ms', 'comm_cost_upload_mb', 'comm_cost_download_mb',
                  'aggregation_time', 'flops_per_forward', 'flops_total',
                  'parameters_m', 'model_size_mb', 'gpu_memory_mb']
    def _init_results_csv(self):
        os.makedirs(osp.dirname(self.results_csv_path), exist_ok=True)
        with open(self.results_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.CSV_FIELDS)
    def save_round_results(self, round_num, client_metrics, client_losses=None, client_lrs=None,
                           extra_metrics_list=None, comm_upload_mb=0, comm_download_mb=0,
                           aggregation_time=0):
        if client_losses is None:
            client_losses = [None] * len(client_metrics)
        if client_lrs is None:
            client_lrs = [None] * len(client_metrics)
        valid_losses = [l for l in client_losses if l is not None]
        valid_lrs = [l for l in client_lrs if l is not None]
        avg_loss = sum(valid_losses) / len(valid_losses) if valid_losses else None
        avg_lr = sum(valid_lrs) / len(valid_lrs) if valid_lrs else None
        hr10_list, hr50_list, r10_50_list, r1_1_list, r1_10_list, r1_50_list = [], [], [], [], [], []
        for m in client_metrics:
            metrics = m.get('hr10', {}) if isinstance(m.get('hr10'), dict) else m
            if isinstance(metrics, dict):
                hr10_list.append(metrics.get('hr10', 0))
                hr50_list.append(metrics.get('hr50', 0))
                r10_50_list.append(metrics.get('r10@50', 0))
                r1_1_list.append(metrics.get('r1@1', 0))
                r1_10_list.append(metrics.get('r1@10', 0))
                r1_50_list.append(metrics.get('r1@50', 0))
        n = len(hr10_list)
        avg_hr10 = sum(hr10_list) / n if n > 0 else 0
        avg_hr50 = sum(hr50_list) / n if n > 0 else 0
        avg_r10_50 = sum(r10_50_list) / n if n > 0 else 0
        avg_r1_1 = sum(r1_1_list) / n if n > 0 else 0
        avg_r1_10 = sum(r1_10_list) / n if n > 0 else 0
        avg_r1_50 = sum(r1_50_list) / n if n > 0 else 0
        avg_training_time = 0
        avg_inference_time = 0
        avg_flops_per_forward = 0
        avg_flops_total = 0
        avg_gpu_memory = 0
        avg_num_params = 0
        avg_model_size_mb = 0
        if extra_metrics_list:
            valid_extras = [e for e in extra_metrics_list if e is not None]
            if valid_extras:
                avg_training_time = sum(e.get('total_training_time', 0) for e in valid_extras) / len(valid_extras)
                avg_inference_time = sum(e.get('avg_inference_time_ms', 0) for e in valid_extras) / len(valid_extras)
                avg_flops_per_forward = sum(e.get('flops_per_forward', 0) for e in valid_extras) / len(valid_extras)
                avg_flops_total = sum(e.get('flops_total', 0) for e in valid_extras) / len(valid_extras)
                avg_gpu_memory = sum(e.get('peak_gpu_memory_mb', 0) for e in valid_extras) / len(valid_extras)
                avg_num_params = sum(e.get('num_params', 0) for e in valid_extras) / len(valid_extras)
                avg_model_size_mb = sum(e.get('model_size_mb', 0) for e in valid_extras) / len(valid_extras)
        row = {
            'round': round_num,
            'loss': avg_loss,
            'lr': avg_lr,
            'hr10': avg_hr10,
            'hr50': avg_hr50,
            'r10@50': avg_r10_50,
            'r1@1': avg_r1_1,
            'r1@10': avg_r1_10,
            'r1@50': avg_r1_50,
            'training_time': f'{avg_training_time:.2f}',
            'inference_time_ms': f'{avg_inference_time:.2f}',
            'comm_cost_upload_mb': f'{comm_upload_mb:.4f}',
            'comm_cost_download_mb': f'{comm_download_mb:.4f}',
            'aggregation_time': f'{aggregation_time:.2f}',
            'flops_per_forward': avg_flops_per_forward,
            'flops_total': avg_flops_total,
            'parameters_m': f'{avg_num_params/1e6:.2f}',
            'model_size_mb': f'{avg_model_size_mb:.2f}',
            'gpu_memory_mb': f'{avg_gpu_memory:.1f}',
        }
        with open(self.results_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writerow(row)
        self.rounds_results.append({
            'round': round_num,
            'avg_loss': avg_loss,
            'avg_lr': avg_lr,
            'avg_metrics': {
                'hr10': avg_hr10,
                'hr50': avg_hr50,
                'r10@50': avg_r10_50,
                'r1@1': avg_r1_1,
                'r1@10': avg_r1_10,
                'r1@50': avg_r1_50,
            },
            'avg_training_time': avg_training_time,
            'avg_inference_time_ms': avg_inference_time,
            'comm_cost_upload_mb': comm_upload_mb,
            'comm_cost_download_mb': comm_download_mb,
            'aggregation_time': aggregation_time,
            'avg_flops_per_forward': avg_flops_per_forward,
            'avg_flops_total': avg_flops_total,
            'avg_num_params': avg_num_params,
            'avg_model_size_mb': avg_model_size_mb,
            'avg_gpu_memory_mb': avg_gpu_memory,
            'num_clients': n
        })
    def save_final_results(self, all_rounds_metrics):
        results = {
            'dataset': self.dataset,
            'distance_type': self.distance_type,
            'config': {
                'use_hyperbolic': self.use_hyperbolic,
                'hyp_beta': self.hyp_beta,
                'num_clients': self.num_clients,
                'enable_personalization': self.enable_personalization,
                'use_gradient_similarity': self.use_gradient_similarity,
                'tau': self.tau,
            },
            'all_rounds': []
        }
        for round_data in all_rounds_metrics:
            r = round_data['round']
            metrics_list = round_data['metrics']
            valid_metrics = [m for m in metrics_list if m.get('hr10') is not None]
            if not valid_metrics:
                continue
            avg_hr10 = sum(m['hr10']['hr10'] if isinstance(m['hr10'], dict) else m['hr10']
                          for m in valid_metrics) / len(valid_metrics)
            avg_hr50 = sum(m['hr10']['hr50'] if isinstance(m['hr10'], dict) else 0
                          for m in valid_metrics) / len(valid_metrics)
            avg_r10_50 = sum(m['hr10']['r10@50'] if isinstance(m['hr10'], dict) else 0
                            for m in valid_metrics) / len(valid_metrics)
            avg_r1_1 = sum(m['hr10']['r1@1'] if isinstance(m['hr10'], dict) else 0
                         for m in valid_metrics) / len(valid_metrics)
            avg_r1_10 = sum(m['hr10']['r1@10'] if isinstance(m['hr10'], dict) else 0
                            for m in valid_metrics) / len(valid_metrics)
            avg_r1_50 = sum(m['hr10']['r1@50'] if isinstance(m['hr10'], dict) else 0
                            for m in valid_metrics) / len(valid_metrics)
            round_extra = next((rr for rr in self.rounds_results if rr['round'] == r), {})
            results['all_rounds'].append({
                'round': r,
                'num_clients': len(valid_metrics),
                'avg_metrics': {
                    'hr10': float(avg_hr10),
                    'hr50': float(avg_hr50),
                    'r10@50': float(avg_r10_50),
                    'r1@1': float(avg_r1_1),
                    'r1@10': float(avg_r1_10),
                    'r1@50': float(avg_r1_50),
                },
                'training_time': round_extra.get('avg_training_time', 0),
                'inference_time_ms': round_extra.get('avg_inference_time_ms', 0),
                'comm_cost_upload_mb': round_extra.get('comm_cost_upload_mb', 0),
                'comm_cost_download_mb': round_extra.get('comm_cost_download_mb', 0),
                'aggregation_time': round_extra.get('aggregation_time', 0),
                'flops_per_forward': round_extra.get('avg_flops_per_forward', 0),
                'flops_total': round_extra.get('avg_flops_total', 0),
                'parameters_m': round_extra.get('avg_num_params', 0) / 1e6,
                'model_size_mb': round_extra.get('avg_model_size_mb', 0),
                'gpu_memory_mb': round_extra.get('avg_gpu_memory_mb', 0),
                'client_details': [
                    {
                        'client_id': m.get('client_id', i),
                        'epoch': m.get('epoch', 0),
                        'metrics': m['hr10'] if isinstance(m['hr10'], dict) else
                                   {'hr10': m['hr10'], 'hr50': 0, 'r10@50': 0,
                                    'r1@1': 0, 'r1@10': 0, 'r1@50': 0}
                    } for i, m in enumerate(metrics_list)
                ]
            })
        with open(self.results_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Final results saved to {self.results_json_path}")
    def get_client_model_path(self, client_id, epoch):
        return osp.join('saved_models', self.dataset, self.experiment_dir,
                       f'client_{client_id}', f'epoch_{epoch}.pt')
    def get_latest_client_model(self, client_id):
        client_dir = osp.join('saved_models', self.dataset, self.experiment_dir, f'client_{client_id}')
        if not os.path.exists(client_dir):
            return None
        files = [f for f in os.listdir(client_dir) if f.endswith('.pt')]
        if not files:
            return None
        epochs = []
        for f in files:
            try:
                epoch = int(f.replace('epoch_', '').replace('.pt', ''))
                epochs.append((epoch, f))
            except:
                continue
        if not epochs:
            return None
        epochs.sort(key=lambda x: x[0])
        latest_epoch, latest_file = epochs[-1]
        return osp.join(client_dir, latest_file), latest_epoch
def federated_train_round(aggregator, client_ids, round_num, epochs_per_round=1):
    from federated_trainer import FederatedTrainer
    logging.info(f"\n{'='*60}")
    logging.info(f"Federated Round {round_num}")
    logging.info(f"{'='*60}")
    client_weights = []
    client_sample_counts = []
    client_metrics = []  
    extra_metrics_list = []  
    prev_global_model = None
    if round_num > 0:
        prev_global_model = aggregator.load_global_model(round_num - 1)
        if prev_global_model is not None:
            logging.info(f"Loaded global model from round {round_num - 1}")
        else:
            logging.warning(f"Could not load global model from round {round_num - 1}")
    for client_id in tqdm(client_ids, desc=f"Round {round_num} - Local Training"):
        try:
            aggregator.config.epochs = epochs_per_round  
            trainer = FederatedTrainer(aggregator.config, client_id, experiment_dir=aggregator.experiment_dir)
            lambda_align = getattr(aggregator, 'lambda_align', 0.1)
            best_metrics, best_epoch, extra_metrics = trainer.Spa_train(
                load_global_model=prev_global_model,
                fed_round=round_num,
                lambda_align=lambda_align
            )
            final_loss = None
            final_lr = None
            if trainer.train_history['loss']:
                final_loss = trainer.train_history['loss'][-1]
            if trainer.train_history['lr']:
                final_lr = trainer.train_history['lr'][-1]
            client_metrics.append({
                'client_id': client_id,
                'hr10': best_metrics,
                'epoch': best_epoch,
                'final_loss': final_loss,
                'final_lr': final_lr
            })
            extra_metrics_list.append(extra_metrics)
        except Exception as e:
            logging.error(f"Client {client_id} training failed: {e}")
            continue
        model_path, _ = aggregator.get_latest_client_model(client_id)
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            client_weights.append(state_dict)
            client_dir = osp.join('data', aggregator.dataset, f'client_{client_id}', 'st_traj')
            train_data = np.load(osp.join(client_dir, 'train_node_list.npy'), allow_pickle=True)
            client_sample_counts.append(len(train_data))
        else:
            logging.warning(f"Client {client_id}: No model found after training")
    if len(client_weights) == 0:
        logging.error("No client weights collected! Aborting aggregation.")
        return None, [], []
    logging.info(f"Collected {len(client_weights)} client models, aggregating...")
    comm_upload_mb = sum(compute_communication_cost(cw) for cw in client_weights)
    start_agg = time.time()
    global_weights = aggregator.FedAvg_weighted(client_weights, client_sample_counts, prev_global_model)
    aggregation_time = time.time() - start_agg
    if global_weights is None:
        logging.error("Aggregation failed!")
        return None, [], []
    aggregator.save_global_model(global_weights, round_num)
    comm_download_mb = compute_communication_cost(global_weights) * len(client_ids)
    logging.info(f"Aggregation time: {aggregation_time:.2f}s")
    logging.info(f"Communication cost: upload={comm_upload_mb:.2f}MB, download={comm_download_mb:.2f}MB")
    for client_id in client_ids:
        client_dir = osp.join('saved_models', aggregator.dataset, aggregator.experiment_dir, f'client_{client_id}')
        os.makedirs(client_dir, exist_ok=True)
        global_model_path = osp.join(client_dir, f'global_round_{round_num}.pt')
        torch.save(global_weights, global_model_path)
        logging.info(f"Client {client_id}: Global model distributed")
    logging.info(f"\n{'='*60}")
    logging.info(f"Round {round_num} - Client Metrics Summary")
    logging.info(f"{'='*60}")
    total_metrics = {'hr10': 0, 'hr50': 0, 'r10@50': 0, 'r1@1': 0, 'r1@10': 0, 'r1@50': 0}
    valid_count = 0
    for m in client_metrics:
        metrics = m['hr10'] if isinstance(m['hr10'], dict) else m['hr10']
        if isinstance(m['hr10'], dict):
            metrics = m['hr10']
        else:
            metrics = {'hr10': m['hr10'], 'hr50': 0, 'r10@50': 0, 'r1@1': 0, 'r1@10': 0, 'r1@50': 0}
        logging.info(f"  Client {m['client_id']:2d}: HR10={metrics['hr10']:.4f}, HR50={metrics['hr50']:.4f}, "
                     f"R10@50={metrics['r10@50']:.4f}, R1@1={metrics['r1@1']:.4f}, "
                     f"R1@10={metrics['r1@10']:.4f}, R1@50={metrics['r1@50']:.4f}, Best Epoch={m['epoch']}")
        total_metrics['hr10'] += metrics['hr10']
        total_metrics['hr50'] += metrics['hr50']
        total_metrics['r10@50'] += metrics['r10@50']
        total_metrics['r1@1'] += metrics['r1@1']
        total_metrics['r1@10'] += metrics['r1@10']
        total_metrics['r1@50'] += metrics['r1@50']
        valid_count += 1
    if valid_count > 0:
        avg_metrics = {k: v / valid_count for k, v in total_metrics.items()}
        logging.info(f"  Average HR10: {avg_metrics['hr10']:.4f}")
        logging.info(f"  Average HR50: {avg_metrics['hr50']:.4f}")
        logging.info(f"  Average R10@50: {avg_metrics['r10@50']:.4f}")
        logging.info(f"  Average R1@1: {avg_metrics['r1@1']:.4f}")
        logging.info(f"  Average R1@10: {avg_metrics['r1@10']:.4f}")
        logging.info(f"  Average R1@50: {avg_metrics['r1@50']:.4f}")
    else:
        logging.info(f"  No valid metrics")
    client_losses = [m.get('final_loss') for m in client_metrics]
    client_lrs = [m.get('final_lr') for m in client_metrics]
    aggregator.save_round_results(round_num, client_metrics, client_losses, client_lrs,
                                   extra_metrics_list=extra_metrics_list,
                                   comm_upload_mb=comm_upload_mb,
                                   comm_download_mb=comm_download_mb,
                                   aggregation_time=aggregation_time)
    return global_weights, client_sample_counts, client_metrics
def federated_test(aggregator, round_num, client_ids=None, use_cpu=True, test_mode='global'):
    from federated_trainer import FederatedTrainer
    if client_ids is None:
        client_ids = list(range(aggregator.num_clients))
    logging.info(f"\n{'='*60}")
    if test_mode == 'global':
        logging.info(f"Federated Test - Round {round_num} Global Model")
    else:
        logging.info(f"Local Test - Round {round_num} Each Client's Best Model")
    logging.info(f"{'='*60}")
    if test_mode == 'global':
        model_weights = aggregator.load_global_model(round_num)
        if model_weights is None:
            logging.error(f"Global model for round {round_num} not found!")
            return
        logging.info(f"Loaded global model from round {round_num}")
    else:
        logging.info(f"Using each client's local best model from round {round_num}")
    import torch
    from model_network import GraphTrajSimEncoder
    import spatial_data_utils
    import test_method
    import federated_data_utils
    all_test_metrics = []
    for client_id in tqdm(client_ids, desc="Testing clients"):
        try:
            trainer = FederatedTrainer(aggregator.config, client_id, experiment_dir=aggregator.experiment_dir)
            test_device = "cpu" if use_cpu else trainer.device
            csv_path = osp.join(trainer.save_folder, 'training_results.csv')
            training_time = 0.0
            if osp.exists(csv_path):
                try:
                    import csv as csv_module
                    with open(csv_path, 'r') as f:
                        reader = csv_module.reader(f)
                        next(reader)  
                        for row in reader:
                            if len(row) >= 3:
                                training_time += float(row[2]) if row[2] else 0.0
                except Exception:
                    pass
            _, test_lenth = federated_data_utils.federated_vali_data_loader(
                client_id, trainer.test_batch)
            net = GraphTrajSimEncoder(
                feature_size=trainer.feature_size,
                embedding_size=trainer.embedding_size,
                hidden_size=trainer.hidden_size,
                num_layers=trainer.num_layers,
                dropout_rate=trainer.dropout_rate,
                concat=trainer.concat,
                device=test_device,
                usePE=trainer.usePE,
                useSI=trainer.useSI,
                useLSTM=trainer.useLSTM,
                dataset=trainer.dataset,
                alpha1=trainer.alpha1,
                alpha2=trainer.alpha2,
                use_hyperbolic=trainer.use_hyperbolic,
                useGRU=trainer.useGRU,
                hyp_beta=trainer.hyp_beta,
                hyp_factor_dim=trainer.hyp_factor_dim,
                use_fully_hyperbolic=trainer.use_fully_hyperbolic,
                hyp_use_att=trainer.hyp_use_att,
                hyp_gcn_type=trainer.hyp_gcn_type,
            )
            if test_mode == 'global':
                missing, unexpected = net.load_state_dict(model_weights, strict=False)
                if unexpected:
                    logging.warning(f"Client {client_id}: {len(unexpected)} unexpected keys in global model: {list(unexpected)[:5]}...")
                if missing:
                    logging.warning(f"Client {client_id}: {len(missing)} missing keys in global model: {list(missing)[:5]}...")
                critical_missing = [
                    k for k in missing
                    if 'graph_embedding' in k and should_aggregate_param(k)
                ]
                if critical_missing:
                    logging.error(f"Client {client_id}: Missing critical parameters: {critical_missing[:5]}..., skipping client")
                    continue
                elif missing:
                    logging.warning(f"Client {client_id}: Non-critical missing keys: {list(missing)[:5]}...")
            else:
                local_model_path = osp.join(trainer.save_folder, "best_model.pt")
                if not osp.exists(local_model_path):
                    logging.error(f"Local model not found: {local_model_path}")
                    continue
                saved_weights = torch.load(local_model_path, map_location=test_device)
                missing, unexpected = net.load_state_dict(saved_weights, strict=False)
                if unexpected:
                    logging.warning(f"Client {client_id}: {len(unexpected)} unexpected keys: {list(unexpected)[:5]}...")
                if missing:
                    logging.warning(f"Client {client_id}: {len(missing)} missing keys: {list(missing)[:5]}...")
                critical_missing = [
                    k for k in missing
                    if 'graph_embedding' in k and should_aggregate_param(k)
                ]
                if critical_missing:
                    logging.error(f"Client {client_id}: Missing critical parameters: {critical_missing[:5]}..., skipping client")
                    continue
                elif missing:
                    logging.warning(f"Client {client_id}: Non-critical missing keys: {list(missing)[:5]}...")
                logging.info(f"Client {client_id}: Loaded local model from {local_model_path}")
            net.to(test_device)
            test_data_loader, _ = federated_data_utils.federated_vali_data_loader(
                client_id, trainer.test_batch)
            if trainer.useSI:
                road_network_data = spatial_data_utils.load_neighbor(trainer.dataset, trainer.num_knn)
                road_network = [item.to(test_device) for item in road_network_data]
            else:
                road_network = spatial_data_utils.load_netowrk(trainer.dataset).to(test_device)
            distance_to_anchor_node = federated_data_utils.load_region_node(
                client_id, trainer.dataset).to(test_device)
            net.eval()
            with torch.no_grad():
                start_inference = time.time()
                if trainer.useLSTM or trainer.useGRU:
                    test_embedding = torch.zeros((test_lenth, trainer.hidden_size * 2), device=test_device)
                else:
                    test_embedding = torch.zeros((test_lenth, trainer.hidden_size), device=test_device)
                test_label = torch.zeros((test_lenth, 50), dtype=torch.long)
                for batch in test_data_loader:
                    (data, coor, label, data_length, idx) = batch
                    output = net([road_network, distance_to_anchor_node], data, coor, seq_lengths=data_length)
                    a_embedding = output['emb_eucl'] if isinstance(output, dict) else output
                    test_embedding[idx] = a_embedding
                    test_label[idx] = label
                acc = test_method.test_spa_model(test_embedding, test_label, test_device)
                acc = acc.mean(axis=0)
                acc[0] = acc[0] / 10.0
                acc[1] = acc[1] / 50.0
                acc[2] = acc[2] / 10.0
                inference_time = time.time() - start_inference
                logging.info(f"  Client {client_id}: HR10={acc[0]:.4f}, HR50={acc[1]:.4f}, "
                           f"R10@50={acc[2]:.4f}, R1@1={acc[3]:.4f}, "
                           f"R1@10={acc[4]:.4f}, R1@50={acc[5]:.4f}, "
                           f"Inference={inference_time:.2f}s")
                all_test_metrics.append({
                    'client_id': client_id,
                    'hr10': acc[0].item() if hasattr(acc[0], 'item') else acc[0],
                    'hr50': acc[1].item() if hasattr(acc[1], 'item') else acc[1],
                    'r10@50': acc[2].item() if hasattr(acc[2], 'item') else acc[2],
                    'r1@1': acc[3].item() if hasattr(acc[3], 'item') else acc[3],
                    'r1@10': acc[4].item() if hasattr(acc[4], 'item') else acc[4],
                    'r1@50': acc[5].item() if hasattr(acc[5], 'item') else acc[5],
                    'inference_time': inference_time,
                    'training_time': training_time,
                })
        except Exception as e:
            logging.error(f"  Client {client_id} test failed: {e}")
            continue
    if all_test_metrics:
        avg_metrics = {'hr10': 0, 'hr50': 0, 'r10@50': 0, 'r1@1': 0, 'r1@10': 0, 'r1@50': 0}
        total_inference_time = 0
        total_training_time = 0
        for m in all_test_metrics:
            avg_metrics['hr10'] += m['hr10']
            avg_metrics['hr50'] += m['hr50']
            avg_metrics['r10@50'] += m['r10@50']
            avg_metrics['r1@1'] += m['r1@1']
            avg_metrics['r1@10'] += m['r1@10']
            avg_metrics['r1@50'] += m['r1@50']
            total_inference_time += m.get('inference_time', 0)
            total_training_time += m.get('training_time', 0)
        n = len(all_test_metrics)
        avg_metrics = {k: v / n for k, v in avg_metrics.items()}
        avg_inference_time = total_inference_time / n
        avg_training_time = total_training_time / n
        logging.info(f"\n{'='*60}")
        logging.info(f"Test Summary - Round {round_num} ({n} clients)")
        logging.info(f"{'='*60}")
        logging.info(f"  Average HR10: {avg_metrics['hr10']:.4f}")
        logging.info(f"  Average HR50: {avg_metrics['hr50']:.4f}")
        logging.info(f"  Average R10@50: {avg_metrics['r10@50']:.4f}")
        logging.info(f"  Average R1@1: {avg_metrics['r1@1']:.4f}")
        logging.info(f"  Average R1@10: {avg_metrics['r1@10']:.4f}")
        logging.info(f"  Average R1@50: {avg_metrics['r1@50']:.4f}")
        logging.info(f"  Average Training Time: {avg_training_time:.2f}s")
        logging.info(f"  Average Inference Time: {avg_inference_time:.2f}s")
    return all_test_metrics
def run_federated_learning(num_rounds=10, clients_per_round=None, epochs_per_round=1, test_after_training=False, use_cpu=False, test_round=-1, experiment_dir=None):
    from setting import SetParameter
    import argparse
    config = SetParameter()
    num_clients = 20
    aggregator = FederatedAggregator(config, num_clients, experiment_dir=experiment_dir)
    if clients_per_round is None or clients_per_round >= num_clients:
        all_clients = list(range(num_clients))
    else:
        all_clients = list(range(clients_per_round))
    logging.info(f"Starting Federated Learning")
    logging.info(f"Total rounds: {num_rounds}")
    logging.info(f"Clients per round: {len(all_clients)}")
    logging.info(f"Epochs per client per round: {epochs_per_round}")
    all_rounds_metrics = []  
    for round_num in range(num_rounds):
        global_weights, sample_counts, client_metrics = federated_train_round(
            aggregator, all_clients, round_num, epochs_per_round
        )
        if global_weights is None:
            logging.error(f"Round {round_num} failed, stopping.")
            break
        all_rounds_metrics.append({
            'round': round_num,
            'metrics': client_metrics
        })
    logging.info(f"\n{'='*60}")
    logging.info("Federated Learning Completed!")
    logging.info(f"{'='*60}")
    logging.info("\nAll Rounds Summary:")
    logging.info(f"{'='*60}")
    for round_data in all_rounds_metrics:
        r = round_data['round']
        metrics = round_data['metrics']
        if metrics:
            avg_hr10 = sum(m['hr10']['hr10'] if isinstance(m['hr10'], dict) else m['hr10'] for m in metrics) / len(metrics)
            avg_hr50 = sum(m['hr10']['hr50'] if isinstance(m['hr10'], dict) else 0 for m in metrics) / len(metrics)
            avg_r10_50 = sum(m['hr10']['r10@50'] if isinstance(m['hr10'], dict) else 0 for m in metrics) / len(metrics)
            avg_r1_1 = sum(m['hr10']['r1@1'] if isinstance(m['hr10'], dict) else 0 for m in metrics) / len(metrics)
            avg_r1_10 = sum(m['hr10']['r1@10'] if isinstance(m['hr10'], dict) else 0 for m in metrics) / len(metrics)
            avg_r1_50 = sum(m['hr10']['r1@50'] if isinstance(m['hr10'], dict) else 0 for m in metrics) / len(metrics)
            logging.info(f"  Round {r}: HR10={avg_hr10:.4f}, HR50={avg_hr50:.4f}, R10@50={avg_r10_50:.4f}, "
                         f"R1@1={avg_r1_1:.4f}, R1@10={avg_r1_10:.4f}, R1@50={avg_r1_50:.4f} ({len(metrics)} clients)")
        else:
            logging.info(f"  Round {r}: No valid metrics")
    if test_after_training:
        actual_test_round = test_round if test_round != -1 else num_rounds - 1
        federated_test(aggregator, actual_test_round, all_clients, use_cpu=use_cpu)
    aggregator.save_final_results(all_rounds_metrics)
def test_only(round_num, clients_per_round=None, use_cpu=True, test_mode='global', experiment_dir=None):
    from setting import SetParameter
    config = SetParameter()
    num_clients = 20
    aggregator = FederatedAggregator(config, num_clients, experiment_dir=experiment_dir)
    if clients_per_round is None or clients_per_round >= num_clients:
        client_ids = list(range(num_clients))
    else:
        client_ids = list(range(clients_per_round))
    federated_test(aggregator, round_num, client_ids, use_cpu=use_cpu, test_mode=test_mode)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Federated Aggregation')
    parser.add_argument('--rounds', type=int, default=1, help='Number of federated rounds')
    parser.add_argument('--clients', type=int, default=None, help='Clients per round (default: all)')
    parser.add_argument('--epochs', type=int, default=1, help='Local epochs per round')
    parser.add_argument('--test', action='store_true', help='Run test evaluation after training')
    parser.add_argument('--test-round', type=int, default=-1, help='Which round to test (-1=last)')
    parser.add_argument('--cpu', action='store_true', help='Use CPU for testing')
    parser.add_argument('--test-only', type=int, default=None, help='Only test, specify round number')
    parser.add_argument('--mode', type=str, default='global', choices=['global', 'local'],
                       help="Test mode: 'global' uses federated aggregated model, 'local' uses each client's best_model.pt")
    parser.add_argument('--exp-dir', type=str, default=None,
                       help='Experiment directory name (e.g., TP_20260424_153000). If not specified, auto-generate with timestamp')
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    if args.test_only is not None:
        test_only(round_num=args.test_only, clients_per_round=args.clients,
                 use_cpu=args.cpu, test_mode=args.mode, experiment_dir=args.exp_dir)
    else:
        run_federated_learning(
            num_rounds=args.rounds,
            clients_per_round=args.clients,
            epochs_per_round=args.epochs,
            test_after_training=args.test,
            use_cpu=args.cpu,
            test_round=args.test_round,
            experiment_dir=args.exp_dir
        )
