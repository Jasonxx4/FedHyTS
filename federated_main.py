import argparse
import logging
import os
import os.path as osp
import datetime
from setting import SetParameter
from federated_trainer import FederatedTrainer
def setup_logger(log_path):
    os.makedirs(osp.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
def main():
    parser = argparse.ArgumentParser(description='Federated Learning Main')
    parser.add_argument('--client_id', type=int, default=0, help='Client ID to train (0-19)')
    parser.add_argument('--num_clients', type=int, default=20, help='Total number of clients')
    args = parser.parse_args()
    config = SetParameter()
    dataset = str(config.dataset)
    distance_type = str(config.distance_type)
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = osp.join('log', dataset, 'federated', f'client_{args.client_id}')
    os.makedirs(log_dir, exist_ok=True)
    log_path = osp.join(log_dir, f'{distance_type}_{current_time}.log')
    setup_logger(log_path)
    logging.info("=" * 60)
    logging.info("Federated Learning - Client Training")
    logging.info("=" * 60)
    logging.info(f"Dataset: {dataset}")
    logging.info(f"Distance Type: {distance_type}")
    logging.info(f"Client ID: {args.client_id}")
    logging.info(f"Total Clients: {args.num_clients}")
    logging.info(f"Log Path: {log_path}")
    if args.client_id < 0 or args.client_id >= args.num_clients:
        logging.error(f"Invalid client_id: {args.client_id}. Must be in range [0, {args.num_clients - 1}]")
        return
    client_dir = osp.join('data', dataset, f'client_{args.client_id}')
    if not osp.exists(client_dir):
        logging.error(f"Client {args.client_id} data not found at {client_dir}")
        logging.error("Please run federated_split.py first to create client data splits.")
        return
    st_traj_dir = osp.join(client_dir, 'st_traj')
    if not osp.exists(st_traj_dir):
        logging.error(f"Client {args.client_id} st_traj not found. Please run federated_split.py first.")
        return
    similarity_dir = osp.join(client_dir, distance_type)
    if not osp.exists(similarity_dir):
        logging.error(f"Client {args.client_id} similarity data not found at {similarity_dir}")
        logging.error("Please run federated_spatial_similarity.py first to compute client similarities.")
        return
    logging.info("=" * 60)
    logging.info(f"Starting Training for Client {args.client_id}")
    logging.info("=" * 60)
    try:
        trainer = FederatedTrainer(config, args.client_id)
        trainer.Spa_train()  
        logging.info(f"Client {args.client_id} training completed successfully!")
    except Exception as e:
        logging.error(f"Client {args.client_id} training failed: {e}")
        raise
if __name__ == '__main__':
    main()
