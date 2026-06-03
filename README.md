# FedHyTS: Hyperbolic Federated Learning for Trajectory Similarity with Personalized Adaptation

Source codes for FedHyTS

## Reproducibility & Training：

1. Data Preparation (First-time only).
   ```bash
   python federated_split.py
   ```
2. Compute similarity matrices for all clients.
   ```bash
   python federated_spatial_similarity.py
   ```
3. Run Federated Learning
   ```bash
   python federated_aggregation.py
   ```
4. Test 
   ```bash
   python federated_aggregation.py --test_only 'specify round number'
   ```
