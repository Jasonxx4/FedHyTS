import os
import random
import shutil
import numpy as np
import os.path as osp
import torch
import torch.nn.utils.rnn as rnn_utils
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
from setting import SetParameter
config = SetParameter()
random.seed(1933)
np.random.seed(1933)
class TrainData(Dataset):
    def __init__(self, data, coor, label, neg_label, dis, neg_dis):
        self.data = data
        self.coor = coor
        self.label = label
        self.neg_label = neg_label
        self.dis = dis
        self.neg_dis = neg_dis
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return (self.data[idx], self.coor[idx], self.label[idx],
                self.neg_label[idx], self.dis[idx], self.neg_dis[idx], idx)
class ValiData(Dataset):
    def __init__(self, data, coor, label):
        self.data = data
        self.coor = coor
        self.label = label
        self.index = list(range(len(data)))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return (self.data[idx], self.coor[idx], self.label[idx], self.index[idx])
def load_client_traindata(client_id):
    dataset = str(config.dataset)
    client_dir = osp.join('data', dataset, f'client_{client_id}', 'st_traj')
    similarity_dir = osp.join('data', dataset, f'client_{client_id}', config.distance_type)
    train_node = np.load(osp.join(client_dir, 'train_node_list.npy'), allow_pickle=True)
    train_coor = np.load(osp.join(client_dir, 'train_coor_list.npy'), allow_pickle=True)
    train_dis = np.load(osp.join(similarity_dir, 'train_spatial_distance.npy'))
    pos_num = config.pos_num
    valid_idx = []
    for i in range(len(train_dis)):
        if np.sum(train_dis[i] >= 0) > pos_num * 2:
            valid_idx.append(i)
    valid_idx = np.array(valid_idx)
    if len(valid_idx) == 0:
        raise ValueError(f"Client {client_id}: No valid trajectories found!")
    train_node = train_node[valid_idx]
    train_coor = train_coor[valid_idx]
    train_dis = train_dis[valid_idx[:, None], valid_idx]
    x_list, y_list = [], []
    for traj in train_coor:
        for r in traj:
            x_list.append(r[0])
            y_list.append(r[1])
    meanx, meany, stdx, stdy = np.mean(x_list), np.mean(y_list), np.std(x_list), np.std(y_list)
    train_coor = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy]
                   for r in t] for t in train_coor]
    norm_num = np.max(train_dis)
    train_dis = train_dis / norm_num * config.coe
    train_dis[train_dis < 0] = -1
    train_y, train_neg_y, train_dis_list, train_neg_dis = get_train_label(train_dis, pos_num)
    logging.info(f"Client {client_id} train size: {len(train_node)}, valid: {len(valid_idx)}")
    return train_node, train_coor, train_y, train_neg_y, train_dis_list, train_neg_dis
def get_train_label(input_dis_matrix, count):
    label = []
    neg_label = []
    label_dis = []
    neg_label_dis = []
    for i in range(len(input_dis_matrix)):
        input_r = np.array(input_dis_matrix[i])
        idx = np.argsort(input_r)
        val = input_r[idx]
        re_idx = idx[val != -1]
        re_val = val[val != -1]
        label.append(re_idx[1:count+1])
        label_dis.append(re_val[1:count+1])
        neg_label.append(re_idx[count+1:])
        neg_label_dis.append(re_val[count+1:])
    label = np.array(label, dtype=object)
    neg_label = np.array(neg_label, dtype=object)
    label_dis = np.array(label_dis, dtype=object)
    neg_label_dis = np.array(neg_label_dis, dtype=object)
    return label, neg_label, label_dis, neg_label_dis
def load_client_validata(client_id):
    dataset = str(config.dataset)
    client_dir = osp.join('data', dataset, f'client_{client_id}', 'st_traj')
    similarity_dir = osp.join('data', dataset, f'client_{client_id}', config.distance_type)
    vali_node = np.load(osp.join(client_dir, 'test_node_list.npy'), allow_pickle=True)
    vali_coor = np.load(osp.join(client_dir, 'test_coor_list.npy'), allow_pickle=True)
    vali_dis = np.load(osp.join(similarity_dir, 'test_spatial_distance.npy'))
    valid_idx = []
    for i in range(len(vali_dis)):
        if np.sum(vali_dis[i] >= 0) > 50:
            valid_idx.append(i)
    valid_idx = np.array(valid_idx)
    vali_node = vali_node[valid_idx]
    vali_coor = vali_coor[valid_idx]
    vali_dis = vali_dis[valid_idx[:, None], valid_idx]
    x_list, y_list = [], []
    for traj in vali_coor:
        for r in traj:
            x_list.append(r[0])
            y_list.append(r[1])
    meanx, meany, stdx, stdy = np.mean(x_list), np.mean(y_list), np.std(x_list), np.std(y_list)
    vali_coor = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy]
                  for r in t] for t in vali_coor]
    vali_y = get_label(vali_dis, 50)
    logging.info(f"Client {client_id} vali size: {len(vali_node)}")
    return vali_node, vali_coor, vali_y
def get_label(input_dis_matrix, count):
    label = []
    for i in range(len(input_dis_matrix)):
        input_r = np.array(input_dis_matrix[i])
        idx = np.argsort(input_r)
        val = input_r[idx]
        idx = idx[val != -1]
        label.append(idx[1:count+1])
    return np.array(label)
def federated_train_data_loader(client_id, batchsize):
    def collate_fn(data_tuple):
        data_aco = []
        coor_aco = []
        data_pos = []
        coor_pos = []
        data_neg = []
        coor_neg = []
        data_pos_dis = []
        data_neg_dis = []
        for i, (data, coor, label, neg_label, dis, neg_dis, idx) in enumerate(data_tuple):
            for j in range(len(label)):
                data_aco.append(torch.LongTensor(data))
                coor_aco.append(torch.tensor(coor, dtype=torch.float32))
                data_pos.append(torch.LongTensor(train_x[label[j]]))
                coor_pos.append(torch.tensor(coor_x[label[j]], dtype=torch.float32))
                data_pos_dis.append(dis[j])
                neg_idx_random = np.random.randint(len(neg_label))
                neg_idx = neg_label[neg_idx_random]
                data_neg.append(torch.LongTensor(train_x[neg_idx]))
                coor_neg.append(torch.tensor(coor_x[neg_idx], dtype=torch.float32))
                data_neg_dis.append(neg_dis[neg_idx_random])
        data_pos_dis = torch.tensor(data_pos_dis)
        data_neg_dis = torch.tensor(data_neg_dis)
        aco_length = torch.tensor(list(map(len, data_aco)))
        pos_length = torch.tensor(list(map(len, data_pos)))
        neg_length = torch.tensor(list(map(len, data_neg)))
        data_aco = rnn_utils.pad_sequence(data_aco, batch_first=True, padding_value=0)
        data_pos = rnn_utils.pad_sequence(data_pos, batch_first=True, padding_value=0)
        data_neg = rnn_utils.pad_sequence(data_neg, batch_first=True, padding_value=0)
        coor_aco = rnn_utils.pad_sequence(coor_aco, batch_first=True, padding_value=0)
        coor_pos = rnn_utils.pad_sequence(coor_pos, batch_first=True, padding_value=0)
        coor_neg = rnn_utils.pad_sequence(coor_neg, batch_first=True, padding_value=0)
        return (data_aco, coor_aco, data_pos, coor_pos, data_neg, coor_neg,
                data_pos_dis, data_neg_dis,
                aco_length, pos_length, neg_length)
    train_x, coor_x, train_y, train_neg_y, train_dis, train_neg_dis = load_client_traindata(client_id)
    data_ = TrainData(train_x, coor_x, train_y, train_neg_y, train_dis, train_neg_dis)
    dataset = DataLoader(data_, batch_size=batchsize, shuffle=True, collate_fn=collate_fn, drop_last=True)
    return dataset
def federated_vali_data_loader(client_id, batchsize):
    def collate_fn(data_tuple):
        data = [torch.LongTensor(sq[0]) for sq in data_tuple]
        coor = [torch.tensor(sq[1], dtype=torch.float32) for sq in data_tuple]
        label = [sq[2] for sq in data_tuple]
        idx = [sq[3] for sq in data_tuple]
        data_length = torch.tensor(list(map(len, data)))
        label = torch.tensor(np.array(label))
        data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
        coor = rnn_utils.pad_sequence(coor, batch_first=True, padding_value=0)
        return data, coor, label, data_length, idx
    val_x, coor_x, val_y = load_client_validata(client_id)
    data_ = ValiData(val_x, coor_x, val_y)
    dataset = DataLoader(
        data_,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return dataset, len(val_x)
def load_region_node(client_id, dataset):
    client_dir = osp.join('data', dataset, f'client_{client_id}')
    anchor_file = osp.join(client_dir, 'distance_to_anchor_node.pt')
    if osp.exists(anchor_file):
        region_node = torch.load(anchor_file)
        return region_node
    print(f"Generating anchor node distance matrix for client {client_id}...")
    node_file = str(config.node_file)
    df_node = pd.read_csv(node_file, sep=',')
    all_node = np.array(df_node['node'])
    all_lng = np.array(df_node['lng'])
    all_lat = np.array(df_node['lat'])
    point_dis = np.load('./ground_truth/{}/Point_dis_matrix.npy'.format(dataset))
    st_traj_dir = osp.join('data', dataset, f'client_{client_id}', 'st_traj')
    train_node = np.load(osp.join(st_traj_dir, 'train_node_list.npy'), allow_pickle=True)
    vali_node = np.load(osp.join(st_traj_dir, 'vali_node_list.npy'), allow_pickle=True)
    test_node = np.load(osp.join(st_traj_dir, 'test_node_list.npy'), allow_pickle=True)
    all_traj_nodes = np.concatenate([train_node, vali_node, test_node])
    unique_nodes = np.unique(np.concatenate([n for n in all_traj_nodes if len(n) > 0]))
    node_to_idx = {node: idx for idx, node in enumerate(all_node)}
    client_node_indices = [node_to_idx[n] for n in unique_nodes if n in node_to_idx]
    if len(client_node_indices) == 0:
        raise ValueError(f"Client {client_id}: No valid nodes found in trajectories")
    min_lng = all_lng[client_node_indices].min()
    max_lng = all_lng[client_node_indices].max()
    min_lat = all_lat[client_node_indices].min()
    max_lat = all_lat[client_node_indices].max()
    lng_margin = (max_lng - min_lng) * 0.1 if max_lng > min_lng else 0.1
    lat_margin = (max_lat - min_lat) * 0.1 if max_lat > min_lat else 0.1
    min_lng -= lng_margin
    max_lng += lng_margin
    min_lat -= lat_margin
    max_lat += lat_margin
    print(f"Client {client_id} - Node range: lng[{min_lng:.4f}, {max_lng:.4f}], lat[{min_lat:.4f}, {max_lat:.4f}]")
    if dataset == 'beijing':
        n_rows, n_cols = 7, 8
        lng_step = (max_lng - min_lng) / n_cols if max_lng > min_lng else 0.25
        lat_step = (max_lat - min_lat) / n_rows if max_lat > min_lat else 0.25
        total_nodes = 112557
        exp_scale = 10000.0
        MAX_ANCHORS = 98  
    elif dataset == 'porto':
        n_rows, n_cols = 6, 6
        lng_step = (max_lng - min_lng) / n_cols if max_lng > min_lng else 0.1
        lat_step = (max_lat - min_lat) / n_rows if max_lat > min_lat else 0.1
        total_nodes = 128466
        exp_scale = 10000.0
        MAX_ANCHORS = 162  
    elif dataset == 'tdrive':
        n_rows, n_cols = 8, 7
        lng_step = (max_lng - min_lng) / n_cols if max_lng > min_lng else 0.1
        lat_step = (max_lat - min_lat) / n_rows if max_lat > min_lat else 0.1
        total_nodes = 74671
        exp_scale = 100.0
        MAX_ANCHORS = 112  
    region_node = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
    for i in range(total_nodes):
        if min_lng <= all_lng[i] <= max_lng and min_lat <= all_lat[i] <= max_lat:
            node_id = all_node[i]
            node_lng = int((all_lng[i] - min_lng) / lng_step)
            node_lat = int((all_lat[i] - min_lat) / lat_step)
            node_lng = min(node_lng, n_cols - 1)
            node_lat = min(node_lat, n_rows - 1)
            region_node[node_lat][node_lng].append(node_id)
    selected_node_set = []
    for i in range(n_rows):
        for j in range(n_cols):
            node_list = region_node[i][j]
            if len(node_list) >= 1:
                selected_node_set.append(node_list[0])
            if len(node_list) >= 2:
                selected_node_set.append(node_list[1])
    num_anchors = len(selected_node_set)
    print(f"Client {client_id} - Selected {num_anchors} anchor nodes, padding to MAX_ANCHORS={MAX_ANCHORS}")
    all_distance_to_node = []
    for item in selected_node_set:
        tmp = point_dis[:total_nodes, item]
        ids = np.where(tmp != -1)
        for idx in ids:
            tmp[idx] = np.exp(-(tmp[idx] / exp_scale))
        all_distance_to_node.append(tmp)
    all_distance_to_node = np.array(all_distance_to_node).T  
    if num_anchors < MAX_ANCHORS:
        padding = np.zeros((total_nodes, MAX_ANCHORS - num_anchors), dtype=np.float32)
        all_distance_to_node = np.concatenate([all_distance_to_node, padding], axis=1)
    elif num_anchors > MAX_ANCHORS:
        all_distance_to_node = all_distance_to_node[:, :MAX_ANCHORS]
    distance_to_anchor_node = torch.tensor(all_distance_to_node, dtype=torch.float)
    print(f"Client {client_id} - Final anchor matrix shape: {distance_to_anchor_node.shape}")
    os.makedirs(client_dir, exist_ok=True)
    torch.save(distance_to_anchor_node, anchor_file)
    print(f"Client {client_id} - anchor node matrix saved to {anchor_file}")
    return distance_to_anchor_node
def load_network(client_id, dataset):
    edge_path = str(config.edge_file)
    client_dir = osp.join('data', dataset, f'client_{client_id}')
    node_embedding_path = osp.join(client_dir, 'node_features.npy')
    if not osp.exists(node_embedding_path):
        raise FileNotFoundError(f"Client {client_id} node_features.npy not found! Run federated_spatial_preprocess.py first.")
    node_embeddings = np.load(node_embedding_path)
    df_edge = pd.read_csv(edge_path, sep=',')
    edge_index = df_edge[["s_node", "e_node"]].to_numpy()
    edge_attr = df_edge["length"].to_numpy()
    if dataset == "beijing" or dataset == "porto":
        edge_attr = edge_attr / 100.0
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    node_embeddings = torch.tensor(node_embeddings, dtype=torch.float)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    from torch_geometric.data import Data
    road_network = Data(x=node_embeddings, edge_index=edge_index, edge_attr=edge_attr)
    return road_network
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    client_id = 0
    print(f"Testing data loading for client {client_id}...")
    train_loader = federated_train_data_loader(client_id, batchsize=16)
    print(f"Train loader created: {len(train_loader)} batches")
    vali_loader, vali_len = federated_vali_data_loader(client_id, batchsize=16)
    print(f"Vali loader created: {len(vali_loader)} batches, {vali_len} samples")
def generate_hierarchical_region_graph(client_id, dataset, num_levels=4, virtuals_per_region=None):
    if virtuals_per_region is None:
        virtuals_per_region = [4, 3, 2]  
    client_dir = osp.join('data', dataset, f'client_{client_id}')
    graph_file = osp.join(client_dir, 'hierarchical_region_graph.pt')
    if osp.exists(graph_file):
        print(f"Loading existing hierarchical graph for client {client_id}")
        return torch.load(graph_file)
    print(f"Generating hierarchical region graph for client {client_id}...")
    node_file = str(config.node_file)
    df_node = pd.read_csv(node_file, sep=',')
    all_node = np.array(df_node['node'])
    all_lng = np.array(df_node['lng'])
    all_lat = np.array(df_node['lat'])
    st_traj_dir = osp.join('data', dataset, f'client_{client_id}', 'st_traj')
    train_node = np.load(osp.join(st_traj_dir, 'train_node_list.npy'), allow_pickle=True)
    vali_node = np.load(osp.join(st_traj_dir, 'vali_node_list.npy'), allow_pickle=True)
    test_node = np.load(osp.join(st_traj_dir, 'test_node_list.npy'), allow_pickle=True)
    all_traj_nodes = np.concatenate([train_node, vali_node, test_node])
    unique_nodes = np.unique(np.concatenate([n for n in all_traj_nodes if len(n) > 0]))
    node_to_idx = {node: idx for idx, node in enumerate(all_node)}
    client_node_indices = [node_to_idx[n] for n in unique_nodes if n in node_to_idx]
    if len(client_node_indices) == 0:
        raise ValueError(f"Client {client_id}: No valid nodes found in trajectories")
    min_lng = all_lng[client_node_indices].min()
    max_lng = all_lng[client_node_indices].max()
    min_lat = all_lat[client_node_indices].min()
    max_lat = all_lat[client_node_indices].max()
    lng_margin = (max_lng - min_lng) * 0.1 if max_lng > min_lng else 0.1
    lat_margin = (max_lat - min_lat) * 0.1 if max_lat > min_lat else 0.1
    min_lng -= lng_margin
    max_lng += lng_margin
    min_lat -= lat_margin
    max_lat += lat_margin
    base_rows, base_cols = 6, 6
    lng_step = (max_lng - min_lng) / base_cols if max_lng > min_lng else 0.1
    lat_step = (max_lat - min_lat) / base_rows if max_lat > min_lat else 0.1
    grids = {}
    for i in range(base_rows):
        for j in range(base_cols):
            grid_id = i * base_cols + j
            grids[grid_id] = {
                'nodes': [],
                'center_lng': min_lng + (j + 0.5) * lng_step,
                'center_lat': min_lat + (i + 0.5) * lat_step,
                'row': i,
                'col': j
            }
    total_nodes = len(all_node)
    for idx in range(total_nodes):
        node_lng = all_lng[idx]
        node_lat = all_lat[idx]
        if min_lng <= node_lng <= max_lng and min_lat <= node_lat <= max_lat:
            g_col = min(int((node_lng - min_lng) / lng_step), base_cols - 1)
            g_row = min(int((node_lat - min_lat) / lat_step), base_rows - 1)
            grid_id = g_row * base_cols + g_col
            grids[grid_id]['nodes'].append(idx)
    layer1_nodes = []
    layer1_coords = []
    for grid_id, grid in grids.items():
        if len(grid['nodes']) >= 1:
            layer1_nodes.append(grid['nodes'][0])
            layer1_coords.append([grid['center_lng'], grid['center_lat']])
        if len(grid['nodes']) >= 2 and len(layer1_nodes) < 162:  
            layer1_nodes.append(grid['nodes'][1])
            layer1_coords.append([grid['center_lng'] + 0.01, grid['center_lat'] + 0.01])
    layer1 = {
        'node_ids': layer1_nodes,
        'coords': np.array(layer1_coords),
        'is_virtual': [False] * len(layer1_nodes),
        'parent_grid': list(range(len(layer1_nodes)))  
    }
    hierarchical_graph = {
        'layers': [layer1],
        'grid_structure': grids,
        'geo_bounds': {
            'min_lng': min_lng, 'max_lng': max_lng,
            'min_lat': min_lat, 'max_lat': max_lat
        }
    }
    grid_aggregation = {
        1: [(i, j) for i in range(base_rows) for j in range(base_cols)],  
        2: [(i, j) for i in range(3) for j in range(3)],  
        3: [(i, j) for i in range(2) for j in range(2)],  
        4: [(0, 0)]  
    }
    current_layer = layer1.copy()
    node_offset = len(layer1_nodes)
    for level in range(2, num_levels + 1):
        num_virtuals = virtuals_per_region[level - 2] if level - 2 < len(virtuals_per_region) else 2
        prev_layer = current_layer
        prev_coords = prev_layer['coords']
        prev_is_virtual = prev_layer['is_virtual']
        num_prev_regions = len(prev_coords)
        num_new_regions = num_prev_regions // 4  
        if num_new_regions < 1:
            num_new_regions = 1
        region_groups = {}
        for i in range(num_prev_regions):
            group_id = i % num_new_regions
            if group_id not in region_groups:
                region_groups[group_id] = []
            region_groups[group_id].append(i)
        layer_nodes = list(prev_layer['node_ids'])
        layer_coords = list(prev_layer['coords'])
        layer_is_virtual = list(prev_layer['is_virtual'])
        layer_parents = []
        for region_id, node_indices in region_groups.items():
            region_coords = prev_coords[node_indices]
            center_lng = np.mean(region_coords[:, 0])
            center_lat = np.mean(region_coords[:, 1])
            for v_id in range(num_virtuals):
                offset = 0.02 * (v_id - num_virtuals // 2)
                virtual_lng = center_lng + offset
                virtual_lat = center_lat + offset
                layer_nodes.append(node_offset)
                layer_coords.append([virtual_lng, virtual_lat])
                layer_is_virtual.append(True)
                layer_parents.append(node_indices)  
                node_offset += 1
        current_layer = {
            'node_ids': layer_nodes,
            'coords': np.array(layer_coords),
            'is_virtual': layer_is_virtual,
            'parent_grid': layer_parents,
            'level': level
        }
        hierarchical_graph['layers'].append(current_layer)
    hierarchical_graph['inter_layer_edges'] = build_inter_layer_edges(hierarchical_graph)
    os.makedirs(client_dir, exist_ok=True)
    torch.save(hierarchical_graph, graph_file)
    print(f"Client {client_id} - Hierarchical graph saved to {graph_file}")
    print(f"  Layers: {[len(l['node_ids']) for l in hierarchical_graph['layers']]}")
    print(f"  Virtual nodes per layer: {[sum(l['is_virtual']) for l in hierarchical_graph['layers']]}")
    return hierarchical_graph
def build_inter_layer_edges(hierarchical_graph):
    layers = hierarchical_graph['layers']
    inter_edges = []
    for level in range(len(layers) - 1):
        upper_layer = layers[level + 1]  
        lower_layer = layers[level]     
        upper_is_virtual = upper_layer['is_virtual']
        lower_parents = lower_layer.get('parent_grid', list(range(len(lower_layer['node_ids']))))
        for up_idx, up_parents in enumerate(lower_parents):
            if isinstance(up_parents, int):
                up_parents = [up_parents]
            for low_idx in up_parents:
                if low_idx < len(lower_layer['node_ids']):
                    inter_edges.append((low_idx, len(lower_layer['node_ids']) + up_idx))
        upper_coords = upper_layer['coords']
        for i in range(len(upper_coords)):
            for j in range(i + 1, len(upper_coords)):
                if upper_is_virtual[i] and upper_is_virtual[j]:
                    dist = np.sqrt((upper_coords[i, 0] - upper_coords[j, 0])**2 +
                                   (upper_coords[i, 1] - upper_coords[j, 1])**2)
                    weight = 1.0 / (1.0 + dist * 1000)
                    inter_edges.append((i, j, weight))
    return inter_edges
def load_hierarchical_region_graph(client_id, dataset):
    return generate_hierarchical_region_graph(client_id, dataset)
