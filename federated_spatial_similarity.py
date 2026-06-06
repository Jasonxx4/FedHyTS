from STmatching_distribution_ver import network_data
from multiprocessing import Pool, cpu_count
import numpy as np
import networkx as nx
import numba
import random
import pandas as pd
import collections
import os
import os.path as osp
from tqdm import tqdm
from setting import SetParameter
config = SetParameter()
random.seed(1998)
NUM_CLIENTS = 20
NUM_WORKERS = 20  
distance_matrix = None
node_edge_dict = None
longest_traj_len = None
hot_node_id = None
distance_type = None
def generate_node_edge_interation(count):
    matrix = np.zeros([count, count])
    np.fill_diagonal(matrix, 1)
    edge = pd.read_csv(str(config.edge_file))
    node_s, node_e = edge.s_node, edge.e_node
    for idx, (n_s, n_e) in enumerate(zip(node_s, node_e)):
        matrix[n_s, n_e] = 1
        matrix[n_e, n_s] = 1
    return matrix
def find_longest_trajectory(node_list):
    longest = 0
    for node_list_item in node_list:
        if len(node_list_item) > longest:
            longest = len(node_list_item)
    return longest
def load_client_data(client_id):
    client_dir = osp.join('data', dataset, f'client_{client_id}', 'st_traj')
    train_node = np.load(osp.join(client_dir, 'train_node_list.npy'), allow_pickle=True)
    vali_node = np.load(osp.join(client_dir, 'vali_node_list.npy'), allow_pickle=True)
    test_node = np.load(osp.join(client_dir, 'test_node_list.npy'), allow_pickle=True)
    return train_node, vali_node, test_node
def compute_batch_distances(args):
    batch_start, batch_end, query_indices, ref_indices, traj_list, ref_list = args
    global distance_matrix, node_edge_dict, longest_traj_len, hot_node_id, distance_type
    batch_results = []
    for i in query_indices:
        for j in ref_indices:
            if distance_type == 'TP':
                dis = TP_dis(traj_list[i], ref_list[j])
            elif distance_type == 'DITA':
                dis = DITA_dis(traj_list[i], ref_list[j])
            elif distance_type == 'discret_frechet':
                dis = frechet_dis(traj_list[i], ref_list[j])
            elif distance_type == 'LCRS':
                dis = LCRS_dis(traj_list[i], ref_list[j])
            elif distance_type == 'NetERP':
                dis = NetERP_dis(traj_list[i], ref_list[j])
            elif distance_type == 'NetDTW':
                dis = NetDTW_dis(traj_list[i], ref_list[j])
            else:
                dis = TP_dis(traj_list[i], ref_list[j])
            batch_results.append((i, j, dis))
    return batch_results
def compute_traj_distance_matrix_parallel(traj_list, ref_list, batch_size=50):
    global distance_matrix, node_edge_dict, longest_traj_len, hot_node_id, distance_type
    n_query = len(traj_list)
    n_ref = len(ref_list)
    dis_matrix = np.zeros((n_query, n_ref), dtype=np.float32)
    tasks = []
    idx = 0
    for batch_start in range(0, n_query, batch_size):
        batch_end = min(batch_start + batch_size, n_query)
        query_indices = list(range(batch_start, batch_end))
        tasks.append((batch_start, batch_end, query_indices, list(range(n_ref)), traj_list, ref_list))
    print(f"  Total batches: {len(tasks)}, Processing with {NUM_WORKERS} workers...")
    if NUM_WORKERS <= 1:
        print("  Using single-process mode to save memory...")
        for task in tqdm(tasks, desc="Computing distances"):
            batch_result = compute_batch_distances(task)
            for i, j, dis in batch_result:
                dis_matrix[i, j] = dis
    else:
        with Pool(processes=NUM_WORKERS) as pool:
            results = list(tqdm(
                pool.imap(compute_batch_distances, tasks),
                total=len(tasks),
                desc="Computing distances"
            ))
        for batch_result in results:
            for i, j, dis in batch_result:
                dis_matrix[i, j] = dis
    return dis_matrix
@numba.jit(nopython=True, fastmath=True)
def TP_dis(list_a, list_b):
    tr1 = np.array(list_a)
    tr2 = np.array(list_b)
    M, N = len(tr1), len(tr2)
    max1 = -1
    for i in range(M):
        mindis = np.inf
        for j in range(N):
            if distance_matrix[tr1[i]][tr2[j]] != -1:
                temp = distance_matrix[tr1[i]][tr2[j]]
                if temp < mindis:
                    mindis = temp
            else:
                return -1
        if mindis != np.inf and mindis > max1:
            max1 = mindis
    max2 = -1
    for i in range(N):
        mindis = np.inf
        for j in range(M):
            if distance_matrix[tr2[i]][tr1[j]] != -1:
                temp = distance_matrix[tr2[i]][tr1[j]]
                if temp < mindis:
                    mindis = temp
            else:
                return -1
        if mindis != np.inf and mindis > max2:
            max2 = mindis
    return int(max(max1, max2))
@numba.jit(nopython=True, fastmath=True)
def DITA_dis(list_a, list_b):
    tr1, tr2 = np.array(list_a), np.array(list_b)
    M, N = len(tr1), len(tr2)
    cost = np.zeros((M, N))
    tp = distance_matrix[tr1[0]][tr2[0]]
    if tp == -1:
        return -1
    cost[0, 0] = tp
    for i in range(1, M):
        tp = distance_matrix[tr1[i]][tr2[0]]
        if tp == -1:
            return -1
        cost[i, 0] = cost[i - 1, 0] + tp
    for i in range(1, N):
        tp = distance_matrix[tr1[0]][tr2[i]]
        if tp == -1:
            return -1
        cost[0, i] = cost[0, i - 1] + tp
    for i in range(1, M):
        for j in range(1, N):
            small = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            tp = distance_matrix[tr1[i]][tr2[j]]
            if tp == -1:
                return -1
            cost[i, j] = min(small) + tp
    return int(cost[M - 1, N - 1])
@numba.jit(nopython=True, fastmath=True)
def frechet_dis(list_a, list_b):
    tr1, tr2 = np.array(list_a), np.array(list_b)
    M, N = len(tr1), len(tr2)
    c = np.zeros((M + 1, N + 1))
    tp = distance_matrix[tr1[0]][tr2[0]]
    if tp == -1:
        return -1
    c[0, 0] = tp
    for i in range(1, M):
        tp = distance_matrix[tr1[i]][tr2[0]]
        if tp == -1:
            return -1
        temp = tp
        if temp > c[i - 1][0]:
            c[i][0] = temp
        else:
            c[i][0] = c[i - 1][0]
    for i in range(1, N):
        tp = distance_matrix[tr1[0]][tr2[i]]
        if tp == -1:
            return -1
        temp = tp
        if temp > c[0][i - 1]:
            c[0][i] = temp
        else:
            c[0][i] = c[0][i - 1]
    for i in range(1, M):
        for j in range(1, N):
            tp = distance_matrix[tr1[i]][tr2[j]]
            if tp == -1:
                return -1
            c[i, j] = max(tp, min(c[i - 1][j - 1], c[i - 1][j], c[i][j - 1]))
    return int(c[M - 1, N - 1])
@numba.jit(nopython=True, fastmath=True)
def LCRS_dis(list_a, list_b):
    lena = len(list_a)
    lenb = len(list_b)
    c = np.zeros((lena + 1, lenb + 1))
    for i in range(lena):
        for j in range(lenb):
            if node_edge_dict[list_a[i], list_b[j]] >= 1:
                c[i + 1][j + 1] = c[i][j] + 1
            elif c[i + 1][j] > c[i][j + 1]:
                c[i + 1][j + 1] = c[i + 1][j]
            else:
                c[i + 1][j + 1] = c[i][j + 1]
    if c[-1][-1] == 0:
        return longest_traj_len * 2
    else:
        return (lena + lenb - c[-1][-1]) / float(c[-1][-1])
def hot_node():
    global distance_matrix
    max_num = 0
    max_idx = 0
    for idx, nodes_interaction in enumerate(distance_matrix):
        nodes_interaction = np.array(nodes_interaction)
        x = len(nodes_interaction[nodes_interaction != -1])
        if x > max_num:
            max_num = x
            max_idx = idx
    print(f"Hot node: {max_idx}, connections: {max_num}")
    return max_idx
@numba.jit(nopython=True, fastmath=True)
def NetERP_dis(list_a, list_b):
    global distance_matrix, hot_node_id
    lena = len(list_a)
    lenb = len(list_b)
    edit = np.zeros((lena + 1, lenb + 1))
    for i in range(1, lena + 1):
        tp = distance_matrix[hot_node_id][list_a[i - 1]]
        if tp == -1:
            return -1
        edit[i][0] = edit[i - 1][0] + tp
    for i in range(1, lenb + 1):
        tp = distance_matrix[hot_node_id][list_b[i - 1]]
        if tp == -1:
            return -1
        edit[0][i] = edit[0][i - 1] + tp
    for i in range(1, lena + 1):
        for j in range(1, lenb + 1):
            tp1 = distance_matrix[hot_node_id][list_a[i - 1]]
            tp2 = distance_matrix[hot_node_id][list_b[j - 1]]
            tp3 = distance_matrix[list_a[i - 1]][list_b[j - 1]]
            if tp1 == -1 or tp2 == -1 or tp3 == -1:
                return -1
            edit[i][j] = min(edit[i - 1][j] + tp1, edit[i][j - 1] + tp2, edit[i - 1][j - 1] + tp3)
    return edit[-1][-1]
@numba.jit(nopython=True, fastmath=True)
def NetDTW_dis(list_a, list_b):
    global distance_matrix
    lena = len(list_a)
    lenb = len(list_b)
    if lena == 0 or lenb == 0:
        return -1
    cost = np.zeros((lena, lenb))
    for i in range(lena):
        for j in range(lenb):
            d = distance_matrix[list_a[i]][list_b[j]]
            if d == -1:
                return -1
            cost[i, j] = d
    dtw = np.zeros((lena, lenb))
    dtw[0, 0] = cost[0, 0]
    for i in range(1, lena):
        d = distance_matrix[list_a[i]][list_b[0]]
        if d == -1:
            return -1
        dtw[i, 0] = dtw[i - 1, 0] + d
    for j in range(1, lenb):
        d = distance_matrix[list_a[0]][list_b[j]]
        if d == -1:
            return -1
        dtw[0, j] = dtw[0, j - 1] + d
    window = min(lena, lenb)
    for i in range(1, lena):
        j_start = max(0, i - window)
        j_end = min(lenb, i + window + 1)
        for j in range(j_start, j_end):
            d = distance_matrix[list_a[i]][list_b[j]]
            if d == -1:
                return -1
            dtw[i, j] = d + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return int(dtw[lena - 1, lenb - 1])
def compute_client_similarity(client_id):
    global distance_matrix, node_edge_dict, longest_traj_len, hot_node_id, distance_type
    print(f"\n=== Processing Client {client_id} ===")
    client_dir = osp.join('data', dataset, f'client_{client_id}')
    train_node, vali_node, test_node = load_client_data(client_id)
    print(f"Train: {len(train_node)}, Vali: {len(vali_node)}, Test: {len(test_node)}")
    output_dir = osp.join(client_dir, distance_type)
    os.makedirs(output_dir, exist_ok=True)
    if len(train_node) > 0:
        print(f"Computing train similarity for client {client_id}...")
        train_dis = compute_traj_distance_matrix_parallel(train_node, train_node, batch_size=50)
        np.save(osp.join(output_dir, 'train_spatial_distance.npy'), train_dis)
        print(f"Train similarity shape: {train_dis.shape}")
    if len(vali_node) > 0:
        print(f"Computing vali similarity for client {client_id}...")
        vali_dis = compute_traj_distance_matrix_parallel(vali_node, vali_node, batch_size=50)
        np.save(osp.join(output_dir, 'vali_spatial_distance.npy'), vali_dis)
        print(f"Vali similarity shape: {vali_dis.shape}")
    if len(test_node) > 0:
        print(f"Computing test similarity for client {client_id}...")
        test_dis = compute_traj_distance_matrix_parallel(test_node, test_node, batch_size=50)
        np.save(osp.join(output_dir, 'test_spatial_distance.npy'), test_dis)
        print(f"Test similarity shape: {test_dis.shape}")
    print(f"Client {client_id} similarity computation completed!")
def main():
    global distance_matrix, node_edge_dict, longest_traj_len, hot_node_id, distance_type, NUM_WORKERS
    dataset = str(config.dataset)
    distance_type = str(config.distance_type)
    print(f"Dataset: {dataset}")
    print(f"Distance type: {distance_type}")
    print(f"Number of clients: {NUM_CLIENTS}")
    print(f"Number of workers: {NUM_WORKERS}")
    print(f"Available CPU cores: {cpu_count()}")
    if not osp.exists('./ground_truth/{}/Point_dis_matrix.npy'.format(dataset)):
        print("Point_dis_matrix.npy not found! Run spatial_similarity_computation.py first.")
        return
    distance_matrix = np.load('./ground_truth/{}/Point_dis_matrix.npy'.format(dataset))
    print(f"Point distance matrix loaded: {distance_matrix.shape}")
    node_edge_dict = generate_node_edge_interation(true_point)
    sample_train, _, _ = load_client_data(0)
    longest_traj_len = find_longest_trajectory(sample_train)
    if distance_type == 'NetERP':
        hot_node_id = hot_node()
    else:
        hot_node_id = None
    NUM_WORKERS = min(NUM_WORKERS, NUM_CLIENTS)
    for client_id in range(NUM_CLIENTS):
        compute_client_similarity(client_id)
    print("\n=== All clients similarity computation completed! ===")
if __name__ == '__main__':
    dataset = str(config.dataset)
    dataset_point = config.pointnum[str(config.dataset)]
    true_point = config.truenum[str(config.dataset)]
    main()
