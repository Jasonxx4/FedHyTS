import numpy as np
import os.path as osp
import os
import json
from setting import SetParameter
config = SetParameter()
NUM_CLIENTS = 20
SPLIT_MODE = 'region'  
def random_split(all_node_list, all_coor_list, num_clients=NUM_CLIENTS):
    n = len(all_node_list)
    client_ids = np.arange(n) % num_clients
    return client_ids
def region_split(all_coor_list, num_clients=NUM_CLIENTS):
    start_coords = np.array([traj[0] for traj in all_coor_list])
    lng_min, lat_min = start_coords.min(axis=0)
    lng_max, lat_max = start_coords.max(axis=0)
    n_rows, n_cols = 40, 40  
    lng_step = (lng_max - lng_min) / n_cols if lng_max > lng_min else 1
    lat_step = (lat_max - lat_min) / n_rows if lat_max > lat_min else 1
    grid_counts = {}
    grid_traj_indices = {}
    for idx, coord in enumerate(start_coords):
        lng, lat = coord[0], coord[1]
        g_lng = min(int((lng - lng_min) / lng_step), n_cols - 1) if lng_step > 0 else 0
        g_lat = min(int((lat - lat_min) / lat_step), n_rows - 1) if lat_step > 0 else 0
        grid_id = g_lat * n_cols + g_lng
        if grid_id not in grid_counts:
            grid_counts[grid_id] = 0
            grid_traj_indices[grid_id] = []
        grid_counts[grid_id] += 1
        grid_traj_indices[grid_id].append(idx)
    client_counts = [0] * num_clients
    client_grids = {i: [] for i in range(num_clients)}
    sorted_grids = sorted(grid_counts.items(), key=lambda x: -x[1])
    for grid_id, count in sorted_grids:
        if count == 0:
            continue
        min_client = min(range(num_clients), key=lambda c: client_counts[c])
        client_grids[min_client].append(grid_id)
        client_counts[min_client] += count
    print(f"\nGrid-based balanced split ({n_rows}x{n_cols} grids -> {num_clients} clients):")
    for c in range(num_clients):
        print(f"  Client {c:2d}: {client_counts[c]:5d} trajectories")
    client_ids = np.zeros(len(all_coor_list), dtype=int)
    for client_id, grids in client_grids.items():
        for grid_id in grids:
            for traj_idx in grid_traj_indices[grid_id]:
                client_ids[traj_idx] = client_id
    return client_ids
def split_and_save(client_ids, all_node_list, all_coor_list, num_clients=NUM_CLIENTS):
    dataset = config.dataset
    for client_id in range(num_clients):
        client_dir = osp.join('data', dataset, f'client_{client_id}')
        os.makedirs(client_dir, exist_ok=True)
        os.makedirs(osp.join(client_dir, 'st_traj'), exist_ok=True)
    train_size = config.train_set_size
    train_node_list = all_node_list[:train_size]
    train_coor_list = all_coor_list[:train_size]
    train_client_ids = client_ids[:train_size]
    for client_id in range(num_clients):
        mask = train_client_ids == client_id
        client_node = train_node_list[mask]
        client_coor = train_coor_list[mask]
        client_dir = osp.join('data', dataset, f'client_{client_id}', 'st_traj')
        np.save(osp.join(client_dir, 'train_node_list.npy'), client_node)
        np.save(osp.join(client_dir, 'train_coor_list.npy'), client_coor)
    vali_size = config.vali_set_size
    vali_node_list = all_node_list[train_size:vali_size]
    vali_coor_list = all_coor_list[train_size:vali_size]
    vali_client_ids = client_ids[train_size:vali_size]
    for client_id in range(num_clients):
        mask = vali_client_ids == client_id
        client_node = vali_node_list[mask]
        client_coor = vali_coor_list[mask]
        client_dir = osp.join('data', dataset, f'client_{client_id}', 'st_traj')
        np.save(osp.join(client_dir, 'vali_node_list.npy'), client_node)
        np.save(osp.join(client_dir, 'vali_coor_list.npy'), client_coor)
    test_node_list = all_node_list[vali_size:]
    test_coor_list = all_coor_list[vali_size:]
    test_client_ids = client_ids[vali_size:]
    for client_id in range(num_clients):
        mask = test_client_ids == client_id
        client_node = test_node_list[mask]
        client_coor = test_coor_list[mask]
        client_dir = osp.join('data', dataset, f'client_{client_id}', 'st_traj')
        np.save(osp.join(client_dir, 'test_node_list.npy'), client_node)
        np.save(osp.join(client_dir, 'test_coor_list.npy'), client_coor)
    meta = {
        'num_clients': num_clients,
        'split_mode': SPLIT_MODE,
        'train_set_size': int(train_size),
        'vali_set_size': int(vali_size - train_size),
        'test_set_size': int(len(all_node_list) - vali_size),
    }
    for client_id in range(num_clients):
        train_count = np.sum(train_client_ids == client_id)
        vali_count = np.sum(vali_client_ids == client_id)
        test_count = np.sum(test_client_ids == client_id)
        meta[f'client_{client_id}'] = {
            'train_count': int(train_count),
            'vali_count': int(vali_count),
            'test_count': int(test_count),
        }
    meta_path = osp.join('data', dataset, 'federated_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    return meta
def main():
    dataset = str(config.dataset)
    print(f"Dataset: {dataset}")
    print(f"Split mode: {SPLIT_MODE}")
    print(f"Number of clients: {NUM_CLIENTS}")
    all_node_list = np.load(
        osp.join('data', dataset, 'st_traj', 'shuffle_node_list.npy'),
        allow_pickle=True
    )
    all_coor_list = np.load(
        osp.join('data', dataset, 'st_traj', 'shuffle_coor_list.npy'),
        allow_pickle=True
    )
    print(f"Total trajectories: {len(all_node_list)}")
    print(f"Train: {config.train_set_size}, Vali: {config.vali_set_size - config.train_set_size}, Test: {config.test_set_size - config.vali_set_size}")
    if SPLIT_MODE == 'random':
        client_ids = random_split(all_node_list, all_coor_list, NUM_CLIENTS)
    elif SPLIT_MODE == 'region':
        client_ids = region_split(all_coor_list, NUM_CLIENTS)
    else:
        raise ValueError(f"Unknown split mode: {SPLIT_MODE}")
    meta = split_and_save(client_ids, all_node_list, all_coor_list, NUM_CLIENTS)
    print("\n=== Split Statistics ===")
    for client_id in range(NUM_CLIENTS):
        info = meta[f'client_{client_id}']
        print(f"Client {client_id:2d}: train={info['train_count']:5d}, vali={info['vali_count']:5d}, test={info['test_count']:5d}")
    print(f"\nMetadata saved to: data/{dataset}/federated_meta.json")
    print("Split completed!")
if __name__ == '__main__':
    main()
