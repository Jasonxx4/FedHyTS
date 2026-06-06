from setting import SetParameter
from model_network import GraphTrajSimEncoder, GraphTrajSTEncoder
import federated_data_utils
import spatial_data_utils
import torch
from lossfun import SpaLossFun
from lossfun_improved import ImprovedSpaLossFun
import time
from tqdm import tqdm
import numpy as np
import test_method
import logging
import os
import os.path as osp
import json
import csv
def _cuda_sync(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
def _measure_model_flops(net, road_network, distance_to_anchor_node, train_data_loader, device):
    total_flops = 0
    def _count_linear_flops(module, input, output):
        nonlocal total_flops
        in_f = input[0].shape[-1]
        out_f = output.shape[-1]
        num_elements = input[0].numel() // in_f
        total_flops += num_elements * in_f * out_f * 2  
    def _count_lstm_flops(module, input, output):
        nonlocal total_flops
        input_size = module.input_size
        hidden_size = module.hidden_size
        num_layers = module.num_layers
        seq_len = input[0].shape[0] if not module.batch_first else input[0].shape[1]
        batch = input[0].shape[1] if not module.batch_first else input[0].shape[0]
        bidirectional = 2 if module.bidirectional else 1
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * bidirectional
            flops_per_step = 4 * (layer_input_size * hidden_size + hidden_size * hidden_size) * 2
            total_flops += flops_per_step * seq_len * batch * bidirectional
    def _count_gru_flops(module, input, output):
        nonlocal total_flops
        input_size = module.input_size
        hidden_size = module.hidden_size
        num_layers = module.num_layers
        seq_len = input[0].shape[0] if not module.batch_first else input[0].shape[1]
        batch = input[0].shape[1] if not module.batch_first else input[0].shape[0]
        bidirectional = 2 if module.bidirectional else 1
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * bidirectional
            flops_per_step = 3 * (layer_input_size * hidden_size + hidden_size * hidden_size) * 2
            total_flops += flops_per_step * seq_len * batch * bidirectional
    def _count_conv1d_flops(module, input, output):
        nonlocal total_flops
        batch = input[0].shape[0]
        in_ch = input[0].shape[1]
        out_ch = output.shape[1]
        k = module.kernel_size[0]
        seq_out = output.shape[-1]
        total_flops += batch * out_ch * in_ch * k * seq_out * 2
    def _count_bmm_flops(module, input, output):
        nonlocal total_flops
        if len(input) >= 2 and input[0].dim() == 3 and input[1].dim() == 3:
            b, n, m = input[0].shape
            p = input[1].shape[-1]
            total_flops += b * n * m * p * 2
    hooks = []
    for name, module in net.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(_count_linear_flops))
        elif isinstance(module, torch.nn.LSTM):
            hooks.append(module.register_forward_hook(_count_lstm_flops))
        elif isinstance(module, torch.nn.GRU):
            hooks.append(module.register_forward_hook(_count_gru_flops))
        elif isinstance(module, torch.nn.Conv1d):
            hooks.append(module.register_forward_hook(_count_conv1d_flops))
    net.eval()
    try:
        with torch.no_grad():
            for batch in train_data_loader:
                (data, coor, data_pos, coor_pos, data_neg, coor_neg,
                 data_pos_dis, data_neg_dis,
                 data_length, pos_length, neg_length) = batch
                _ = net([road_network, distance_to_anchor_node], data, coor, seq_lengths=data_length)
                break  
    except Exception as e:
        logging.warning(f"FLOPs measurement failed: {e}, using parameter-based estimate")
        num_params = sum(p.numel() for p in net.parameters())
        total_flops = num_params * 2  
    finally:
        for h in hooks:
            h.remove()
    return total_flops
class FederatedTrainer:
    def __init__(self, config, client_id, experiment_dir=None):
        self.client_id = client_id
        self.feature_size = config.feature_size
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.dropout_rate = config.dropout_rate
        self.concat = config.concat
        self.device = "cuda:" + str(config.cuda)
        self.learning_rate = config.learning_rate
        self.epochs = config.epochs
        self.train_batch = config.gtraj["train_batch"]
        self.test_batch = config.gtraj["test_batch"]
        self.usePE = config.gtraj["usePE"]
        self.useSI = config.gtraj["useSI"]
        self.useLSTM = config.gtraj["useLSTM"]
        self.use_hyperbolic = config.gtraj.get("use_hyperbolic", False)
        self.useGRU = config.gtraj.get("useGRU", False)  
        self.hyp_beta = config.gtraj.get("hyp_beta", 1.0)
        self.hyp_factor_dim = config.gtraj.get("hyp_factor_dim", 16)
        self.use_fully_hyperbolic = config.gtraj.get("use_fully_hyperbolic", False)
        self.hyp_use_att = config.gtraj.get("hyp_use_att", False)
        self.hyp_gcn_type = config.gtraj.get("hyp_gcn_type", 'hyperbolic')
        self.use_improved_loss = config.gtraj.get("use_improved_loss", False)
        self.triplet_margin = config.gtraj.get("triplet_margin", 0.5)
        self.use_contrastive = config.gtraj.get("use_contrastive", False)
        self.lambda_reg = config.gtraj.get("lambda_reg", 0.01)
        self.use_time = config.gtraj.get("use_time", False)
        self.date2vec_size = config.date2vec_size
        self.dataset = str(config.dataset)
        self.distance_type = str(config.distance_type)
        self.early_stop = config.early_stop
        self.alpha1 = config.alpha1
        self.alpha2 = config.alpha2
        self.num_knn = config.num_knn
        self.config = config
        self.experiment_dir = experiment_dir if experiment_dir else config.distance_type
        self.save_folder = osp.join('saved_models', self.dataset,
                                     self.experiment_dir,
                                     f'client_{client_id}')
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        logging.info(f"Client {client_id} - Save folder: {self.save_folder}")
        self.train_history = {
            'epochs': [],
            'loss': [],
            'lr': [],
            'vali_metrics': {
                'hr10': [], 'hr50': [], 'r10@50': [], 'r1@1': [], 'r1@10': [], 'r1@50': []
            },
            'training_time': [],      
            'inference_time': [],     
            'gpu_memory_mb': [],      
        }
        self.csv_path = osp.join(self.save_folder, 'training_results.csv')
        self.json_path = osp.join(self.save_folder, 'training_results.json')
    def Spa_train(self, load_global_model=None, fed_round=0, lambda_align=0.01,
              update_scale=1.0, distance_penalty_alpha=0.0):
        logging.info(f"Client {self.client_id} - FedHyTS on {self.dataset} with {self.distance_type}")
        logging.info(f"Client {self.client_id} - positive num: {self.config.pos_num}")
        logging.info(f"Client {self.client_id} - GPU: {self.config.cuda}")
        net = GraphTrajSimEncoder(
            feature_size=self.feature_size,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            concat=self.concat,
            device=self.device,
            usePE=self.usePE,
            useSI=self.useSI,
            useLSTM=self.useLSTM,
            dataset=self.dataset,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
            use_hyperbolic=self.use_hyperbolic,
            useGRU=self.useGRU,
            hyp_beta=self.hyp_beta,
            hyp_factor_dim=self.hyp_factor_dim,
            use_fully_hyperbolic=self.use_fully_hyperbolic,
            hyp_use_att=self.hyp_use_att,
            hyp_gcn_type=self.hyp_gcn_type,
            use_time=self.use_time,
            date2vec_size=self.date2vec_size,
        ).to(self.device)
        self.global_model_ref = None
        self.use_direction_penalty = (lambda_align > 0 and load_global_model is not None)
        if load_global_model is not None:
            self.global_model_ref = {k: v.clone() for k, v in load_global_model.items()}
            def _detect_arch(state_dict):
                if any('graph_embedding.layer1_encoder' in k for k in state_dict):
                    return 'hierarchical_hyperbolic'
                if any('graph_embedding.conv1.linear' in k for k in state_dict):
                    return 'fully_hyperbolic'
                if any('graph_embedding.hyp_conv1' in k for k in state_dict):
                    return 'hyperbolic'
                return 'euclidean'
            global_arch = _detect_arch(load_global_model)
            local_arch = self.hyp_gcn_type if self.use_hyperbolic else 'euclidean'
            if global_arch != local_arch:
                logging.warning(f"Client {self.client_id} - Architecture mismatch: "
                                f"global={global_arch}, local={local_arch}, skipping global model init")
                self.global_model_ref = None
            else:
                try:
                    missing_keys, unexpected_keys = net.load_state_dict(load_global_model, strict=False)
                    from federated_aggregation import should_aggregate_param
                    critical_missing = [
                        k for k in missing_keys
                        if 'graph_embedding' in k and should_aggregate_param(k)
                    ]
                    if critical_missing:
                        logging.error(f"Client {self.client_id} - Missing critical params: "
                                      f"{critical_missing[:3]}..., skipping global model init")
                        self.global_model_ref = None
                    else:
                        if missing_keys:
                            logging.warning(f"Client {self.client_id} - Missing keys: {len(missing_keys)} "
                                            f"(personalized layers, using local init)")
                        if unexpected_keys:
                            logging.warning(f"Client {self.client_id} - Unexpected keys: {len(unexpected_keys)} "
                                            f"(will be ignored)")
                        logging.info(f"Client {self.client_id} - Initialized from global model (arch={global_arch})")
                except Exception as e:
                    logging.error(f"Client {self.client_id} - Failed to load global model: {e}")
                    logging.info(f"Client {self.client_id} - Starting from random initialization")
                    self.global_model_ref = None
        else:
            logging.info(f"Client {self.client_id} - Starting from random initialization (no global model)")
        self.lambda_align = lambda_align
        if self.use_direction_penalty:
            logging.info(f"Client {self.client_id} - Direction alignment penalty enabled with lambda={lambda_align}")
        self.update_scale = update_scale
        self.distance_penalty_alpha = distance_penalty_alpha
        if self.update_scale < 1.0 or distance_penalty_alpha > 0:
            logging.info(f"Client {self.client_id} - Conservative update enabled: update_scale={update_scale}, distance_penalty_alpha={distance_penalty_alpha}")
        elif fed_round > 0:
            self.update_scale = 1
            self.distance_penalty_alpha = 0.01
            logging.info(f"Client {self.client_id} - Default conservative update for round {fed_round}: update_scale={self.update_scale}, distance_penalty_alpha={self.distance_penalty_alpha}")
        if fed_round > 0:
            adjusted_lr = self.learning_rate
            logging.info(f"Client {self.client_id} - Fed round 0 (initial), LR: {adjusted_lr:.6f}")
        else:
            adjusted_lr = self.learning_rate
            logging.info(f"Client {self.client_id} - Fed round 0 (initial), LR: {adjusted_lr:.6f}")
        optimizer = torch.optim.Adam(
            [p for p in net.parameters() if p.requires_grad],
            lr=adjusted_lr,
            weight_decay=0.0001
        )
        if self.epochs > 10:
            milestones_list = [15, 30, 45, 60, 75, 90, 115]
            logging.info(f"Client {self.client_id} - milestones: {milestones_list}")
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones_list, gamma=0.2)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(3, self.epochs // 2), gamma=0.5)
            logging.info(f"Client {self.client_id} - Using StepLR with step_size={max(3, self.epochs // 2)}")
        if self.use_improved_loss:
            logging.info(f"Client {self.client_id} - Using ImprovedSpaLossFun with margin={self.triplet_margin:.2f}")
            lossfunction = ImprovedSpaLossFun(
                self.train_batch, self.distance_type, hyp_beta=self.hyp_beta,
                margin=self.triplet_margin, use_contrastive=self.use_contrastive,
                lambda_reg=self.lambda_reg
            ).to(self.device)
        else:
            logging.info(f"Client {self.client_id} - Using original SpaLossFun")
            lossfunction = SpaLossFun(self.train_batch, self.distance_type, hyp_beta=self.hyp_beta).to(self.device)
        distance_to_anchor_node = federated_data_utils.load_region_node(self.client_id, self.dataset).to(self.device)
        if self.useSI:
            road_network = spatial_data_utils.load_neighbor(self.dataset, self.num_knn)
            road_network = [item.to(self.device) for item in road_network]
        else:
            road_network = spatial_data_utils.load_netowrk(self.dataset).to(self.device)
        train_data_loader = federated_data_utils.federated_train_data_loader(self.client_id, self.train_batch)
        vali_data_loader, vali_lenth = federated_data_utils.federated_vali_data_loader(self.client_id, self.train_batch)
        if vali_lenth == 0:
            logging.warning(f"Client {self.client_id} - No validation data! Skipping validation.")
            vali_data_loader = None
        best_epoch = 0
        best_hr10 = 0
        best_metrics = {'hr10': 0, 'hr50': 0, 'r10@50': 0, 'r1@1': 0, 'r1@10': 0, 'r1@50': 0}
        self.best_hr10 = 0  
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        for epoch in range(0, self.epochs):
            net.train()
            losses = []
            _cuda_sync(self.device)
            start_train = time.time()
            for batch in tqdm(train_data_loader, desc=f"Client {self.client_id} Train"):
                optimizer.zero_grad()
                (data, coor, data_pos, coor_pos, data_neg, coor_neg,
                 data_pos_dis, data_neg_dis,
                 data_length, pos_length, neg_length) = batch
                a_embedding = net([road_network, distance_to_anchor_node], data, coor, seq_lengths=data_length)
                p_embedding = net([road_network, distance_to_anchor_node], data_pos, coor_pos, seq_lengths=pos_length)
                n_embedding = net([road_network, distance_to_anchor_node], data_neg, coor_neg, seq_lengths=neg_length)
                task_loss = lossfunction(a_embedding, p_embedding, n_embedding,
                                    data_pos_dis, data_neg_dis, self.device)
                alignment_penalty = 0.0
                if self.use_direction_penalty and self.global_model_ref is not None:
                    for name, param in net.named_parameters():
                        if name in self.global_model_ref and param.grad is not None:
                            delta = param.data - self.global_model_ref[name].to(param.device)
                            grad = param.grad.data
                            delta_norm = delta.norm()
                            grad_norm = grad.norm()
                            if delta_norm > 1e-8 and grad_norm > 1e-8:
                                cos_align = (grad * delta).sum() / (grad_norm * delta_norm)
                                alignment_penalty += cos_align
                    total_loss = task_loss - self.lambda_align * alignment_penalty
                else:
                    total_loss = task_loss
                distance_penalty = 0.0
                if self.distance_penalty_alpha > 0 and self.global_model_ref is not None:
                    for name, param in net.named_parameters():
                        if name in self.global_model_ref:
                            init_param = self.global_model_ref[name].to(param.device)
                            distance_penalty += torch.sum((param - init_param) ** 2)
                    total_loss = total_loss + self.distance_penalty_alpha * distance_penalty
                losses.append(task_loss.item())
                total_loss.backward()
                if self.update_scale < 1.0:
                    for param in net.parameters():
                        if param.grad is not None:
                            param.grad.data = param.grad.data * self.update_scale
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
            _cuda_sync(self.device)
            end_train = time.time()
            epoch_train_time = end_train - start_train
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            current_loss = np.mean(losses)
            self.train_history['epochs'].append(epoch)
            self.train_history['loss'].append(float(current_loss))
            self.train_history['lr'].append(float(current_lr))
            self.train_history['training_time'].append(epoch_train_time)
            for metric in ['hr10', 'hr50', 'r10@50', 'r1@1', 'r1@10', 'r1@50']:
                self.train_history['vali_metrics'][metric].append(None)
            self.train_history['inference_time'].append(None)
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                self.train_history['gpu_memory_mb'].append(peak_mem)
            else:
                self.train_history['gpu_memory_mb'].append(0)
            logging.info(f"Client {self.client_id} - Epoch {epoch}: "
                        f"Train time={epoch_train_time:.2f}s, "
                        f"loss={current_loss:.4f}, "
                        f"lr={current_lr:.6f}")
            if vali_data_loader is not None and epoch % 1 == 0:
                net.eval()
                with torch.no_grad():
                    _cuda_sync(self.device)
                    start_vali = time.time()
                    if self.useLSTM or self.useGRU:
                        vali_embedding = torch.zeros((vali_lenth, self.hidden_size * 2),
                                                    device=self.device, requires_grad=False)
                    else:
                        vali_embedding = torch.zeros((vali_lenth, self.hidden_size),
                                                    device=self.device, requires_grad=False)
                    vali_label = torch.zeros((vali_lenth, 50),
                                              requires_grad=False, dtype=torch.long)
                    for batch in tqdm(vali_data_loader, desc=f"Client {self.client_id} Vali"):
                        (data, coor, label, data_length, idx) = batch
                        output = net([road_network, distance_to_anchor_node],
                                     data, coor, seq_lengths=data_length)
                        a_embedding = output['emb_eucl'] if isinstance(output, dict) else output
                        vali_embedding[idx] = a_embedding
                        vali_label[idx] = label
                    acc = test_method.test_spa_model(vali_embedding, vali_label, self.device)
                    acc = acc.mean(axis=0)
                    acc[0] = acc[0] / 10.0
                    acc[1] = acc[1] / 50.0
                    acc[2] = acc[2] / 10.0
                    _cuda_sync(self.device)
                    end_vali = time.time()
                    epoch_vali_time = end_vali - start_vali
                    avg_inference_time_ms = epoch_vali_time * 1000 / max(vali_lenth, 1)
                    logging.info(f"Client {self.client_id} - Dataset: {self.dataset}, "
                               f"Distance: {self.distance_type}, f_num: {vali_lenth}")
                    logging.info(f"Client {self.client_id} - Vali time: {epoch_vali_time:.2f}s, "
                               f"per-sample: {avg_inference_time_ms:.2f}ms")
                    logging.info(f"Client {self.client_id} - Results: HR10={acc[0]:.4f}, "
                               f"HR50={acc[1]:.4f}, R10@50={acc[2]:.4f}")
                    metrics_list = [acc[0], acc[1], acc[2], acc[3], acc[4], acc[5]]
                    metric_names = ['hr10', 'hr50', 'r10@50', 'r1@1', 'r1@10', 'r1@50']
                    for metric_name, metric_val in zip(metric_names, metrics_list):
                        val = metric_val.item() if hasattr(metric_val, 'item') else metric_val
                        self.train_history['vali_metrics'][metric_name][epoch] = float(val)
                    self.train_history['inference_time'][epoch] = avg_inference_time_ms
                    save_modelname = osp.join(self.save_folder, f"epoch_{epoch}.pt")
                    torch.save(net.state_dict(), save_modelname)
                    if acc[0] > best_hr10:
                        best_hr10 = acc[0]
                        best_epoch = epoch
                        best_metrics = {
                            'hr10': acc[0].item() if hasattr(acc[0], 'item') else acc[0],
                            'hr50': acc[1].item() if hasattr(acc[1], 'item') else acc[1],
                            'r10@50': acc[2].item() if hasattr(acc[2], 'item') else acc[2],
                            'r1@1': acc[3].item() if hasattr(acc[3], 'item') else acc[3],
                            'r1@10': acc[4].item() if hasattr(acc[4], 'item') else acc[4],
                            'r1@50': acc[5].item() if hasattr(acc[5], 'item') else acc[5],
                        }
                        best_modelname = osp.join(self.save_folder, "best_model.pt")
                        torch.save(net.state_dict(), best_modelname)
                        logging.info(f"Client {self.client_id} - New best HR10: {best_hr10:.4f} at epoch {best_epoch}")
                    if epoch - best_epoch >= self.early_stop:
                        logging.info(f"Client {self.client_id} - Early stopping at epoch {epoch}")
                        break
            self.best_hr10 = best_hr10
        logging.info(f"Client {self.client_id} - Training completed! Best HR10={best_hr10:.4f} at epoch {best_epoch}")
        total_training_time = sum(t for t in self.train_history['training_time'])
        valid_infer_times = [t for t in self.train_history['inference_time'] if t is not None]
        avg_inference_time_ms = sum(valid_infer_times) / len(valid_infer_times) if valid_infer_times else 0
        peak_gpu_memory_mb = max(self.train_history['gpu_memory_mb']) if self.train_history['gpu_memory_mb'] else 0
        num_params = sum(p.numel() for p in net.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in net.parameters()) / (1024 ** 2)
        flops_per_forward = _measure_model_flops(net, road_network, distance_to_anchor_node,
                                                  train_data_loader, self.device)
        flops_per_step = flops_per_forward * 3 * 3
        num_epochs_actual = len(self.train_history['training_time'])
        num_batches = len(train_data_loader) if hasattr(train_data_loader, '__len__') else 0
        total_flops = num_epochs_actual * num_batches * flops_per_step
        extra_metrics = {
            'total_training_time': total_training_time,
            'avg_inference_time_ms': avg_inference_time_ms,
            'peak_gpu_memory_mb': peak_gpu_memory_mb,
            'flops_per_forward': flops_per_forward,
            'flops_total': total_flops,
            'num_params': num_params,
            'model_size_mb': model_size_mb,
        }
        logging.info(f"Client {self.client_id} - Extra metrics: "
                    f"train_time={total_training_time:.2f}s, "
                    f"inference/ms={avg_inference_time_ms:.2f}, "
                    f"GPU_mem={peak_gpu_memory_mb:.1f}MB, "
                    f"FLOPs/forward={flops_per_forward/1e6:.2f}MFLOPs, "
                    f"Total_FLOPs={total_flops/1e9:.2f}GFLOPs, "
                    f"params={num_params/1e6:.2f}M, "
                    f"model_size={model_size_mb:.2f}MB")
        self._save_results_csv()
        self._save_results_json()
        logging.info(f"Client {self.client_id} - Training history saved to {self.csv_path} and {self.json_path}")
        return best_metrics, best_epoch, extra_metrics
    def _save_results_csv(self):
        if not self.train_history['epochs']:
            return
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'lr', 'hr10', 'hr50', 'r10@50', 'r1@1', 'r1@10', 'r1@50'])
            for i in range(len(self.train_history['epochs'])):
                row = [
                    self.train_history['epochs'][i],
                    self.train_history['loss'][i],
                    self.train_history['lr'][i],
                ]
                for metric in ['hr10', 'hr50', 'r10@50', 'r1@1', 'r1@10', 'r1@50']:
                    val = self.train_history['vali_metrics'][metric][i]
                    row.append(val if val is not None else '')
                writer.writerow(row)
    def _save_results_json(self):
        if not self.train_history['epochs']:
            return
        results = {
            'client_id': self.client_id,
            'dataset': self.dataset,
            'distance_type': self.distance_type,
            'config': {
                'use_hyperbolic': self.use_hyperbolic,
                'hyp_gcn_type': self.hyp_gcn_type,
                'hyp_beta': self.hyp_beta,
                'hyp_factor_dim': self.hyp_factor_dim,
                'useLSTM': self.useLSTM,
                'useGRU': self.useGRU,
            },
            'train_history': self.train_history,
            'best_metrics': {
                'hr10': float(self.best_hr10) if hasattr(self, 'best_hr10') else 0,
            }
        }
        with open(self.json_path, 'w') as f:
            json.dump(results, f, indent=2)
    def Spa_eval(self, load_model=None, use_cpu=False):
        logging.info(f"Client {self.client_id} - FedHyTS on {self.dataset} with {self.distance_type}")
        if use_cpu:
            test_device = "cpu"
            logging.info(f"Client {self.client_id} - Using CPU for testing")
        else:
            test_device = self.device
            logging.info(f"Client {self.client_id} - Using GPU for testing")
        net = GraphTrajSimEncoder(
            feature_size=self.feature_size,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout_rate=self.dropout_rate,
            concat=self.concat,
            device=test_device,
            usePE=self.usePE,
            useSI=self.useSI,
            useLSTM=self.useLSTM,
            dataset=self.dataset,
            alpha1=self.alpha1,
            alpha2=self.alpha2,
            use_hyperbolic=self.use_hyperbolic,
            useGRU=self.useGRU,
            hyp_beta=self.hyp_beta,
            hyp_factor_dim=self.hyp_factor_dim,
        ).to(test_device)
        if load_model is not None:
            logging.info(f"Client {self.client_id} - Loading model: {load_model}")
            net.load_state_dict(torch.load(load_model, map_location=test_device))
            net.to(test_device)
            test_data_loader, test_lenth = federated_data_utils.federated_vali_data_loader(
                self.client_id, self.test_batch)
            logging.info(f"Client {self.client_id} - Test size: {test_lenth}")
            distance_to_anchor_node = federated_data_utils.load_region_node(self.client_id, self.dataset).to(test_device)
            if self.useSI:
                road_network = spatial_data_utils.load_neighbor(self.dataset, self.num_knn)
                for item in road_network:
                    item = item.to(test_device)
            else:
                road_network = spatial_data_utils.load_netowrk(self.dataset).to(test_device)
            net.eval()
            with torch.no_grad():
                start_test = time.time()
                if self.useLSTM or self.useGRU:
                    test_embedding = torch.zeros((test_lenth, self.hidden_size * 2),
                                                device=test_device, requires_grad=False)
                else:
                    test_embedding = torch.zeros((test_lenth, self.hidden_size),
                                                device=test_device, requires_grad=False)
                test_label = torch.zeros((test_lenth, 50),
                                         requires_grad=False, dtype=torch.long)
                for batch in tqdm(test_data_loader, desc=f"Client {self.client_id} Test"):
                    (data, coor, label, data_length, idx) = batch
                    output = net([road_network, distance_to_anchor_node],
                                 data, coor, data_length)
                    a_embedding = output['emb_eucl'] if isinstance(output, dict) else output
                    test_embedding[idx] = a_embedding
                    test_label[idx] = label
                end_test = time.time()
                acc = test_method.test_spa_model(test_embedding, test_label, test_device)
                acc = acc.mean(axis=0)
                acc[0] = acc[0] / 10.0
                acc[1] = acc[1] / 50.0
                acc[2] = acc[2] / 10.0
                logging.info(f"Client {self.client_id} - Dataset: {self.dataset}, "
                           f"Distance: {self.distance_type}, f_num: {test_lenth}")
                logging.info(f"Client {self.client_id} - Test time: {end_test - start_test:.2f}s")
                logging.info(f"Client {self.client_id} - Results: HR10={acc[0]:.4f}, "
                           f"HR50={acc[1]:.4f}, R10@50={acc[2]:.4f}, "
                           f"R1@1={acc[3]:.4f}, R1@10={acc[4]:.4f}, R1@50={acc[5]:.4f}")
def train_client(client_id, config):
    logging.info(f"=" * 50)
    logging.info(f"Starting training for Client {client_id}")
    logging.info(f"=" * 50)
    trainer = FederatedTrainer(config, client_id)
    best_metrics, best_epoch, extra_metrics = trainer.Spa_train()
    return best_metrics, best_epoch, extra_metrics
def train_all_clients(num_clients=20, total_hr10=None):
    config = SetParameter()
    results = []
    for client_id in range(num_clients):
        try:
            best_metrics, best_epoch, extra_metrics = train_client(client_id, config)
            results.append({'client_id': client_id, 'metrics': best_metrics, 'epoch': best_epoch})
        except Exception as e:
            logging.error(f"Client {client_id} - Training failed: {e}")
            results.append({'client_id': client_id, 'metrics': None, 'epoch': 0, 'error': str(e)})
    logging.info(f"\n{'=' * 50}")
    logging.info("Training Summary - All Clients")
    logging.info(f"{'=' * 50}")
    total_metrics = {'hr10': 0, 'hr50': 0, 'r10@50': 0, 'r1@1': 0, 'r1@10': 0, 'r1@50': 0}
    valid_count = 0
    for r in results:
        if r.get('metrics') is not None:
            m = r['metrics']
            logging.info(f"Client {r['client_id']:2d}: HR10={m['hr10']:.4f}, HR50={m['hr50']:.4f}, "
                          f"R10@50={m['r10@50']:.4f}, R1@1={m['r1@1']:.4f}, "
                          f"R1@10={m['r1@10']:.4f}, R1@50={m['r1@50']:.4f}, Best Epoch={r['epoch']}")
            total_metrics['hr10'] += m['hr10']
            total_metrics['hr50'] += m['hr50']
            total_metrics['r10@50'] += m['r10@50']
            total_metrics['r1@1'] += m['r1@1']
            total_metrics['r1@10'] += m['r1@10']
            total_metrics['r1@50'] += m['r1@50']
            valid_count += 1
        else:
            logging.info(f"Client {r['client_id']:2d}: FAILED - {r.get('error', 'Unknown error')}")
    if valid_count > 0:
        avg_metrics = {k: v / valid_count for k, v in total_metrics.items()}
        logging.info(f"Average HR10: {avg_metrics['hr10']:.4f}")
        logging.info(f"Average HR50: {avg_metrics['hr50']:.4f}")
        logging.info(f"Average R10@50: {avg_metrics['r10@50']:.4f}")
        logging.info(f"Average R1@1: {avg_metrics['r1@1']:.4f}")
        logging.info(f"Average R1@10: {avg_metrics['r1@10']:.4f}")
        logging.info(f"Average R1@50: {avg_metrics['r1@50']:.4f}")
    avg_hr10 = total_hr10 / len([r for r in results if 'error' not in r])
    logging.info(f"Average HR10: {avg_hr10:.4f}")
    return results
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Federated Learning Trainer')
    parser.add_argument('--client_id', type=int, default=0, help='Client ID to train')
    args = parser.parse_args()
    config = SetParameter()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    trainer = FederatedTrainer(config, args.client_id)
    best_metrics, best_epoch, extra_metrics = trainer.Spa_train()
    logging.info(f"Extra metrics: {extra_metrics}")
