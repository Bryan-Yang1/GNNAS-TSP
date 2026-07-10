import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GlobalAttention
import json
import argparse
import math
from scipy.stats import rankdata

parser = argparse.ArgumentParser()
parser.add_argument('--time_limit', type=float, default=60)
parser.add_argument('--num_neighbors', type=int, default=-1) # -1 for fully connected
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--node_limit', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--decay_rate', type=float, default=1.2)
parser.add_argument('--max_epochs', type=int, default=30)
parser.add_argument('--decay_every', type=int, default=5)
parser.add_argument('--rank_method', type=str, default='dense', choices=['dense', 'max', 'average'])
parser.add_argument('--cost_loss', type=str, default='MSE', choices=['MSE', 'MAE', 'Huber'])
parser.add_argument('--rank_loss', type=str, default='ListNet', choices=['RankNet', 'ListNet', 'LambdaRank'])
parser.add_argument('--loss_weight', type=float, default=0.5) # weight of loss_cost
parser.add_argument('--train_ids_json', type=str, default=None)
parser.add_argument('--test_ids_json', type=str, default=None)
parser.add_argument('--data_root', type=str, default="data")
parser.add_argument('--instance_training_dir', type=str, default=None)
parser.add_argument('--instance_testing_dir', type=str, default=None)
parser.add_argument('--execution_dir', type=str, default=None)
parser.add_argument('--results_root', type=str, default="outputs")
parser.add_argument('--results_suffix', type=str, default="")
parser.add_argument('--valid_ratio', type=float, default=None)
parser.add_argument('--auto_train_ratio', type=float, default=0.7)
parser.add_argument('--split_seed', type=int, default=0)
parser.add_argument('--device', type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
args = parser.parse_args()

time_limit = args.time_limit
num_neighbors = args.num_neighbors 
hidden_dim = args.hidden_dim
num_layers = args.num_layers
node_limit = args.node_limit
learning_rate = args.learning_rate
decay_rate = args.decay_rate
num_algorithms = 5
mlp_layers = 2
max_epochs = args.max_epochs
decay_every = args.decay_every
test_every = 5
accumulation_steps = 1
rank_method = args.rank_method
cost_loss_type = args.cost_loss
rank_loss_type = args.rank_loss
loss_weight = args.loss_weight
train_ids_json = args.train_ids_json
test_ids_json = args.test_ids_json
data_root = args.data_root
instance_training_dir = args.instance_training_dir or os.path.join(data_root, "tsp_instances_training")
instance_testing_dir = args.instance_testing_dir or os.path.join(data_root, "tsp_instances_testing")
execution_dir = args.execution_dir or os.path.join(data_root, "tsp_executions")
results_root = args.results_root
results_suffix = args.results_suffix
valid_ratio = args.valid_ratio
auto_train_ratio = args.auto_train_ratio
split_seed = args.split_seed

def resolve_device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device_name == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(device_name)

device = resolve_device(args.device)
print(f"Using device: {device}")

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False

def compute_rank(costs, runtimes):
    # costs, runtimes: 1D numpy arrays, lower cost better, if tie then lower runtime better
    # combine cost and runtime for tie-breaking
    # lexsort: first by cost, then by runtime
    idx = np.lexsort((runtimes, costs))
    ranks = np.empty_like(idx)
    ranks[idx] = np.arange(1, len(costs)+1)  # rank starts from 1
    return ranks.astype(np.float32)

# --- Cost Loss Functions ---
def mse_loss(y_pred, y_true):
    return F.mse_loss(y_pred, y_true)

def mae_loss(y_pred, y_true):
    return F.l1_loss(y_pred, y_true)

def huber_loss(y_pred, y_true, delta=1.0):
    return F.huber_loss(y_pred, y_true, delta=delta)

# --- Ranking Loss Functions ---
def ranknet_loss(y_pred, y_true_rank):
    """
    RankNet loss: pairwise ranking loss using 0/1 labels
    - y_pred: [B, num_algorithms] predicted scores (lower = better, like cost)
    - y_true_rank: [B, num_algorithms] true ranks (1 = best, lower is better)
    - S_ij = 1 if rank_i < rank_j (i is better), 0 otherwise
    - P_ij = sigmoid(s_i - s_j) = probability that i ranks higher than j
    """
    y_pred = y_pred.squeeze()
    y_true_rank = y_true_rank.squeeze()
    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_true_rank = y_true_rank.unsqueeze(0)
    
    B, n = y_pred.shape
    
    # Compute pairwise score differences (negate since lower is better)
    # When rank_i < rank_j, we want y_pred_i to be smaller, so s_i - s_j should be negative
    # Therefore we use -(y_pred_i - y_pred_j) = y_pred_j - y_pred_i
    pred_diff = -(y_pred.unsqueeze(-1) - y_pred.unsqueeze(-2))  # [B, n, n]
    rank_diff = y_true_rank.unsqueeze(-1) - y_true_rank.unsqueeze(-2)  # [B, n, n]
    
    # S_ij = 1 if rank_i < rank_j (i is better), 0 otherwise
    S_ij = (rank_diff < 0).float()  # [B, n, n]
    
    # P_ij = sigmoid(s_i - s_j) = probability that i ranks higher than j
    P_ij = torch.sigmoid(pred_diff)
    
    # RankNet cross-entropy loss
    loss = -S_ij * torch.log(P_ij + 1e-8) - (1 - S_ij) * torch.log(1 - P_ij + 1e-8)
    
    # Only consider valid pairs (where ranks differ)
    mask = (rank_diff != 0).float()
    loss = (loss * mask).sum() / (mask.sum() + 1e-8)

    return loss

def listnet_loss(y_pred, y_true_cost):
    """
    ListNet loss: learns to match the ranking distribution
    - y_pred: [B, num_algorithms] predicted costs (lower = better)
    - y_true_cost: [B, num_algorithms] true costs (lower = better)
    - Convert both to probability distributions using softmax and compute KL divergence
    - Using cost directly preserves relative differences better than using ranks
    """
    y_pred = y_pred.squeeze()
    y_true_cost = y_true_cost.squeeze()
    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_true_cost = y_true_cost.unsqueeze(0)
    
    # Convert both cost values to probability distributions
    # Use negative values so that lower costs get higher probabilities
    pred_prob = F.softmax(-y_pred, dim=-1)      # lower pred -> higher prob
    true_prob = F.softmax(-y_true_cost, dim=-1) # lower cost -> higher prob
    
    # KL divergence: -sum(true_prob * log(pred_prob))
    return -torch.sum(true_prob * torch.log(pred_prob + 1e-8), dim=-1).mean()

def lambdarank_loss(y_pred, y_true_rank, k=3, sigma=1.0):
    """
    LambdaRank loss (approximation): RankNet loss weighted by ΔNDCG
    - y_pred: [B, num_algorithms] predicted scores (lower = better, like cost)
    - y_true_rank: [B, num_algorithms] true ranks (1 = best, lower is better)
    - k: compute gradients based on NDCG@k (only pairs affecting top-k)
    - sigma: temperature parameter for pairwise sigmoid
    """
    y_pred = y_pred.squeeze()
    y_true_rank = y_true_rank.squeeze()
    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_true_rank = y_true_rank.unsqueeze(0)
    
    B, n = y_pred.shape
    device = y_pred.device
    
    # Convert ranks to relevance scores (higher = better)
    rel = (n - y_true_rank + 1.0).to(torch.float32)  # [B, n]
    gains = torch.pow(2.0, rel) - 1.0  # [B, n]
    
    # Compute ideal DCG@k
    gains_sorted, _ = torch.sort(gains, dim=1, descending=True)
    positions = torch.arange(1, n + 1, device=device).float()
    discounts = torch.log2(positions + 1.0)  # [n]
    
    topk = min(k, n)
    idcg = (gains_sorted[:, :topk] / discounts[:topk]).sum(dim=1, keepdim=True)  # [B, 1]
    
    # Compute pairwise differences (negate since lower pred score is better)
    pred_diff = -(y_pred.unsqueeze(-1) - y_pred.unsqueeze(-2))  # [B, n, n]
    rank_diff = y_true_rank.unsqueeze(-1) - y_true_rank.unsqueeze(-2)  # [B, n, n]
    
    # Compute ΔNDCG_ij: the change in NDCG when swapping positions i and j
    # Strict swap-based formulation: ΔDCG = (g_i/d_j + g_j/d_i) - (g_i/d_i + g_j/d_j)
    # Approximate positions by current ranks
    pos_i = y_true_rank.unsqueeze(-1)  # [B, n, 1]
    pos_j = y_true_rank.unsqueeze(-2)  # [B, 1, n]
    discount_i = torch.log2(pos_i + 1.0)  # [B, n, 1]
    discount_j = torch.log2(pos_j + 1.0)  # [B, 1, n]
    
    gains_i = gains.unsqueeze(-1)  # [B, n, 1]
    gains_j = gains.unsqueeze(-2)  # [B, 1, n]
    
    # ΔNDCG = |ΔDCG| / IDCG
    swap_delta = (gains_i / discount_j + gains_j / discount_i) - (gains_i / discount_i + gains_j / discount_j)
    delta_ndcg = torch.abs(swap_delta) / (idcg.unsqueeze(-1) + 1e-8)
    
    # Top-k mask: only consider pairs where at least one is in top-k
    topk_mask_i = (pos_i <= topk).float()  # [B, n, 1]
    topk_mask_j = (pos_j <= topk).float()  # [B, 1, n]
    topk_mask = torch.maximum(topk_mask_i, topk_mask_j)  # [B, n, n]: 1 if i or j in top-k
    
    # S_ij: 1 if i is better than j (rank_i < rank_j)
    S_ij = (rank_diff < 0).float()  # [B, n, n]
    
    # RankNet-style pairwise loss weighted by ΔNDCG
    P_ij = torch.sigmoid(sigma * pred_diff)
    pairwise_loss = -S_ij * torch.log(P_ij + 1e-8) - (1 - S_ij) * torch.log(1 - P_ij + 1e-8)
    
    # Apply ΔNDCG weights, top-k mask, and valid pairs mask
    valid_mask = (rank_diff != 0).float()
    weighted_loss = pairwise_loss * delta_ndcg * topk_mask * valid_mask
    
    loss = weighted_loss.sum() / (valid_mask.sum() + 1e-8)
    return loss

class TSPDataset(Dataset):
    def __init__(self, instance_dirs, execution_dirs, algorithms, time_limit, num_neighbors, filtered_ids_json=None):
        if isinstance(instance_dirs, str):
            instance_dirs = [instance_dirs] # instance_dirscan be a single directory or a list of directories

        self.allowed_ids = None
        if filtered_ids_json is not None:
            with open(filtered_ids_json, 'r') as f:
                self.allowed_ids = set(json.load(f))
      
        self.algorithms = algorithms
        self.instances = []  # [(instance_id, tsp_file_path)]
        self.labels_cost = {}  # {instance_id: [cost_algo1, ...]}
        self.labels_runtime = {}  # {instance_id: [runtime_algo1, ...]}
        self.time_limit = time_limit
        self.num_neighbors = num_neighbors

        for inst_dir in instance_dirs:
            for root, _, files in os.walk(inst_dir):
                for file in files:
                    if file.endswith('.tsp'):
                        iid = file.replace('.tsp', '')
                        if self.allowed_ids is not None and iid not in self.allowed_ids:
                            continue
                        self.instances.append((iid, os.path.join(root, file)))

        # Load cost label of each algorithm at the given time limit
        for algo in algorithms:
            algo_dir = execution_dirs[algo]
            for group in os.listdir(algo_dir):  # sub-directories for different groups
                group_path = os.path.join(algo_dir, group)
                if not os.path.isdir(group_path):
                    continue
                for file in os.listdir(group_path):
                    if not file.endswith('.out'):
                        continue
                    iid = file.replace('.tsp.out', '')
                    out_path = os.path.join(group_path, file)
                    try:
                        # load cost and runtime
                        costs, times = [], []
                        with open(out_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 2 and is_number(parts[0]) and is_number(parts[1]): # skip non-number lines
                                    cost, runtime = float(parts[0]), float(parts[1])
                                    costs.append(cost)
                                    times.append(runtime)
                        if not costs:
                            print(f"Skipped, failed to load data: {algo}/{iid}")
                            continue
                        # Find cost at time_limit
                        times_np = np.array(times)
                        costs_np = np.array(costs)
                        valid_idx = np.where(times_np <= self.time_limit)[0]
                        if len(valid_idx) == 0:
                            cost_at_limit = float('inf')
                            runtime_at_limit = float('inf')
                        else:
                            idx = valid_idx[-1]
                            cost_at_limit = costs_np[idx]
                            runtime_at_limit = times_np[idx]
                    except Exception as e: # actually not needed now
                        print(f"Skipped, failed to load data: {algo}/{iid}: {e}")
                        continue
                    if iid not in self.labels_cost:
                        self.labels_cost[iid] = [None] * len(algorithms)
                        self.labels_runtime[iid] = [None] * len(algorithms)
                    self.labels_cost[iid][self.algorithms.index(algo)] = cost_at_limit
                    self.labels_runtime[iid][self.algorithms.index(algo)] = runtime_at_limit
        # only keep instances with complete cost labels
        self.instances = [
            (iid, path) for iid, path in self.instances
            if iid in self.labels_cost
            and all(c is not None for c in self.labels_cost[iid])
            and any(not math.isinf(c) for c in self.labels_cost[iid])
        ]
        # only keep instances with less than 1000 nodes
        filtered_instances = []
        for iid, path in self.instances:
            coords = self._read_coords(path)
            if len(coords) <= node_limit:
                filtered_instances.append((iid, path))
        self.instances = filtered_instances
        # if any label contains inf, replace it with max cost * 1.5
        for iid in self.labels_cost:
            if iid in [iid for iid, _ in self.instances]:
                label = self.labels_cost[iid]
                if any(math.isinf(c) for c in label):
                    valid_costs = [c for c in label if not math.isinf(c)]
                    max_cost = max(valid_costs)
                    self.labels_cost[iid] = [c if not math.isinf(c) else max_cost * 1.5 for c in label]
                    idx_max = label.index(max_cost)
                    max_runtime = self.labels_runtime[iid][idx_max]
                    self.labels_runtime[iid] = [r if not math.isinf(c) else max_runtime for r, c in zip(self.labels_runtime[iid], label)]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance_id, tsp_path = self.instances[idx]
        coords = self._read_coords(tsp_path)  # V x 2

        # node coords z-score standardization
        coords = np.array(coords, dtype=np.float32)
        coords_mean = coords.mean(axis=0, keepdims=True)
        coords_std = coords.std(axis=0, keepdims=True) + 1e-8
        coords_norm = (coords - coords_mean) / coords_std

        # distance matrix z-score standardization
        diff = coords_norm[:, None, :] - coords_norm[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        dist_mean = dist_matrix.mean()
        dist_std = dist_matrix.std() + 1e-8
        dist_matrix_norm = (dist_matrix - dist_mean) / dist_std

        label_cost = np.array(self.labels_cost[instance_id], dtype=np.float32)
        label_runtime = np.array(self.labels_runtime[instance_id], dtype=np.float32)
        # label_cost z-score standardization
        label_cost_mean = label_cost.mean()
        label_cost_std = label_cost.std() + 1e-8
        label_cost_norm = (label_cost - label_cost_mean) / label_cost_std
        # label_rank: ranking with tie-breaking by runtime
        label_rank = compute_rank(label_cost, label_runtime)
        label_rank = label_rank.astype(np.float32)

        V = len(coords)
        # adjacency matrix: fully connected or kNN
        if self.num_neighbors == -1:
            adj = np.ones((V, V), dtype=np.int64)
        else:
            adj = np.zeros((V, V), dtype=np.int64)
            dist = dist_matrix.copy()
            np.fill_diagonal(dist, np.inf)
            knn_idx = np.argpartition(dist, self.num_neighbors, axis=-1)[:, :self.num_neighbors]
            for i in range(V):
                adj[i, knn_idx[i]] = 1
            # undirected graph
            adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 2)  # self-loop as 2
        return {
            'node_coords': torch.tensor(coords_norm, dtype=torch.float32),
            'edge_matrix': torch.tensor(adj, dtype=torch.long),  # adjacency matrix
            'edge_values': torch.tensor(dist_matrix_norm, dtype=torch.float32),  # standardized distance matrix
            'label_cost': torch.tensor(label_cost_norm, dtype=torch.float32),  # [num_algorithms]
            'label_rank': torch.tensor(label_rank, dtype=torch.float32),       # [num_algorithms]
            'instance_id': instance_id
        }

    def _read_coords(self, tsp_path):
        coords = []
        in_section = False
        with open(tsp_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == 'NODE_COORD_SECTION':
                    in_section = True
                    continue
                if line == 'EOF':
                    break
                if in_section:
                    parts = line.split()
                    if len(parts) >= 3:
                        coords.append([float(parts[1]), float(parts[2])])
        return coords

    def _compute_dist_matrix(self, coords):
        coords = np.array(coords)
        diff = coords[:, None, :] - coords[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        return dists

class BatchNormNode(nn.Module):
    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        x_trans = x.transpose(1, 2).contiguous() 
        x_trans_bn = self.batch_norm(x_trans)
        x_bn = x_trans_bn.transpose(1, 2).contiguous() 
        return x_bn

class BatchNormEdge(nn.Module):
    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, e):
        e_trans = e.transpose(1, 3).contiguous() 
        e_trans_bn = self.batch_norm(e_trans)
        e_bn = e_trans_bn.transpose(1, 3).contiguous() 
        return e_bn
    
class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        U = []
        for layer in range(self.L - 1):
            U.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.U = nn.ModuleList(U)
        self.V = nn.Linear(hidden_dim, output_dim, True)

    def forward(self, x):
        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)  # B x H
            Ux = F.relu(Ux)  # B x H
        y = self.V(Ux)  # B x O
        return y

class NodeFeatures(nn.Module):
    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)
    def forward(self, x, edge_gate):
        Ux = self.U(x)  # B x V x H
        Vx = self.V(x)  # B x V x H
        Vx = Vx.unsqueeze(1)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        gateVx = edge_gate * Vx  # B x V x V x H
        if self.aggregation=="mean":
            x_new = Ux + torch.sum(gateVx, dim=2) / (1e-20 + torch.sum(edge_gate, dim=2))  # B x V x H
        elif self.aggregation=="sum":
            x_new = Ux + torch.sum(gateVx, dim=2)  # B x V x H
        return x_new

class EdgeFeatures(nn.Module):
    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, x, e):
        Ue = self.U(e)
        Vx = self.V(x)
        Wx = Vx.unsqueeze(1)  # extend Vx from "B x V x H" to "B x V x 1 x H"
        Vx = Vx.unsqueeze(2)  # extend Vx from "B x V x H" to "B x 1 x V x H"
        e_new = Ue + Vx + Wx
        return e_new

class ResidualGatedGCNLayer(nn.Module):
    def __init__(self, hidden_dim, aggregation="sum"):
        super(ResidualGatedGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, x, e):
        e_in = e
        x_in = x
        # Edge convolution
        e_tmp = self.edge_feat(x_in, e_in)  # B x V x V x H
        # Compute edge gates
        edge_gate = torch.sigmoid(e_tmp)
        # Node convolution
        x_tmp = self.node_feat(x_in, edge_gate)
        # Batch normalization
        e_tmp = self.bn_edge(e_tmp)
        x_tmp = self.bn_node(x_tmp)
        # ReLU Activation
        e = F.relu(e_tmp)
        x = F.relu(x_tmp)
        # Residual connection
        x_new = x_in + x
        e_new = e_in + e
        return x_new, e_new

class ResidualGatedGCNModel(nn.Module):
    def __init__(self, num_algorithms, hidden_dim, num_layers, mlp_layers, num_neighbors):
        super(ResidualGatedGCNModel, self).__init__()
        self.num_algorithms = num_algorithms
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mlp_layers = mlp_layers
        self.num_neighbors = num_neighbors
        self.nodes_coord_embedding = TransformerConv(2, self.hidden_dim)
        self.edges_values_embedding = nn.Linear(1, self.hidden_dim // 2, bias=False)
        self.edges_embedding = nn.Embedding(3, self.hidden_dim // 2)
        gcn_layers = []
        for layer in range(self.num_layers):
            gcn_layers.append(ResidualGatedGCNLayer(self.hidden_dim, 'mean'))
        self.gcn_layers = nn.ModuleList(gcn_layers)
        # Global attention pooling for node and edge features
        self.node_pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(self.hidden_dim, 1)
        ))
        self.edge_pool = GlobalAttention(gate_nn=nn.Sequential(
            nn.Linear(self.hidden_dim, 1)
        ))
        self.mlp = MLP(self.hidden_dim * 2, self.num_algorithms, self.mlp_layers) 

    def forward(self, x_edges, x_edges_values, x_nodes_coord, label_cost, label_rank):
        # x_edges: (B, V, V) adjacency, x_edges_values: (B, V, V), x_nodes_coord: (B, V, 2)
        node_coords = x_nodes_coord.squeeze(0)  # (V, 2)
        adj = x_edges.squeeze(0)  # (V, V)
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()  # [2, num_edges]
        x = self.nodes_coord_embedding(node_coords, edge_index)
        x = x.unsqueeze(0)  # (1, V, H)
        # edge embedding
        e_vals = self.edges_values_embedding(x_edges_values.unsqueeze(3))  # (B, V, V, H//2)
        e_tags = self.edges_embedding(x_edges)  # (B, V, V, H//2)
        e = torch.cat((e_vals, e_tags), dim=3)  # (B, V, V, H)
        # GCN layers
        for layer in range(self.num_layers):
            x, e = self.gcn_layers[layer](x, e)  # B x V x H, B x V x V x H

        # flatten edge features for pooling: [B, V, V, H] -> [B, V*V, H]
        B, V, _, H = e.shape
        e_flat = e.view(B, V*V, H)
        graph_feature_node = self.node_pool(x)  # [B, H]
        graph_feature_edge = self.edge_pool(e_flat)  # [B, H]
        graph_feature = torch.cat([graph_feature_node, graph_feature_edge], dim=-1)  # [B, 2H]
        y_pred = self.mlp(graph_feature)  # B x num_algorithms
        y_pred = y_pred.view(-1, num_algorithms)


        loss_cost = self.loss_costs(y_pred, label_cost)
        loss_rank = self.loss_rank(y_pred, label_cost, label_rank)
        loss = loss_weight * loss_cost + (1 - loss_weight) * loss_rank
        return y_pred, loss

    def loss_costs(self, y_pred, y_true):
        if cost_loss_type == 'MSE':
            return mse_loss(y_pred, y_true)
        elif cost_loss_type == 'MAE':
            return mae_loss(y_pred, y_true)
        elif cost_loss_type == 'Huber':
            return huber_loss(y_pred, y_true)
        else:
            raise ValueError(f"Unknown cost loss type: {cost_loss_type}")

    def loss_rank(self, y_pred, y_true_cost, y_true_rank):
        if rank_loss_type == 'RankNet':
            return ranknet_loss(y_pred, y_true_rank)
        elif rank_loss_type == 'ListNet':
            return listnet_loss(y_pred, y_true_cost)
        elif rank_loss_type == 'LambdaRank':
            return lambdarank_loss(y_pred, y_true_rank, k=3, sigma=1.0)
        else:
            raise ValueError(f"Unknown rank loss type: {rank_loss_type}")

net = ResidualGatedGCNModel(
    num_algorithms=num_algorithms,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    mlp_layers=mlp_layers,
    num_neighbors=num_neighbors
)
if device.type == "cuda" and torch.cuda.device_count() > 1:
    net = nn.DataParallel(net)
net = net.to(device)

execution_dirs = {
    'CLK': os.path.join(execution_dir, "CLK"),
    'EAX': os.path.join(execution_dir, "EAX"),
    'LKH': os.path.join(execution_dir, "LKH"),
    'MAOS': os.path.join(execution_dir, "MAOS"),
    'CONCORDE': os.path.join(execution_dir, "CONCORDE")}
algorithms = ['CLK','EAX', 'LKH', 'MAOS', 'CONCORDE']
all_instance_dirs = [instance_training_dir, instance_testing_dir]
train_instance_dirs = all_instance_dirs if train_ids_json else instance_training_dir
test_instance_dirs = all_instance_dirs if test_ids_json else instance_testing_dir

# --- Dataset and DataLoader split ---
if train_ids_json or test_ids_json:
    dataset = TSPDataset(
            instance_dirs=train_instance_dirs,
            execution_dirs=execution_dirs,
            algorithms=algorithms,
            time_limit=time_limit,
            num_neighbors=num_neighbors,
            filtered_ids_json=train_ids_json)
    total_len = len(dataset)
    if valid_ratio is None:
        valid_ratio = 0.0

    if valid_ratio > 0:
        train_len = int(total_len * (1 - valid_ratio))
        valid_len = total_len - train_len
        train_dataset, valid_dataset = random_split(dataset, [train_len, valid_len], generator=torch.Generator().manual_seed(split_seed))
    else:
        train_dataset = dataset
        valid_dataset = None
else:
    if not 0 < auto_train_ratio < 1:
        raise ValueError("--auto_train_ratio must be between 0 and 1")

    dataset = TSPDataset(
            instance_dirs=all_instance_dirs,
            execution_dirs=execution_dirs,
            algorithms=algorithms,
            time_limit=time_limit,
            num_neighbors=num_neighbors)
    total_len = len(dataset)
    train_len = int(total_len * auto_train_ratio)
    test_len = total_len - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(split_seed))
    valid_dataset = None

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True) if valid_dataset is not None else None
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(valid_dataset) if valid_dataset is not None else 0}")

def train_one_epoch(net, optimizer):
    # Set training mode
    net.train()

    # Initialize running data
    running_loss = 0.0
    running_nb_data = 0

    start_epoch = time.time()
    for batch in train_loader:
        # Convert batch to torch Variables
        x_edges = batch['edge_matrix'].to(device)
        x_edges_values = batch['edge_values'].to(device)
        x_nodes_coord = batch['node_coords'].to(device)
        label_cost = batch['label_cost'].to(device)
        label_rank = batch['label_rank'].to(device)
        
        # Forward pass
        y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes_coord, label_cost, label_rank)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update running data
        running_nb_data += 1
        running_loss += loss.data.item()  # Re-scale loss

    # Compute statistics for full epoch
    loss = running_loss / running_nb_data

    return time.time() - start_epoch, loss

def validate_one_epoch(net):
    if valid_loader is None:
        return None

    net.eval()
    running_loss = 0.0
    running_nb_data = 0
    with torch.no_grad():
        for batch in valid_loader:
            x_edges = batch['edge_matrix'].to(device)
            x_edges_values = batch['edge_values'].to(device)
            x_nodes_coord = batch['node_coords'].to(device)
            label_cost = batch['label_cost'].to(device)
            label_rank = batch['label_rank'].to(device)
            y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes_coord, label_cost, label_rank)
            loss = loss.mean()
            running_nb_data += 1
            running_loss += loss.data.item()
    loss = running_loss / running_nb_data
    return loss

results_dir = os.path.join(
    results_root,
    f"results_{time_limit}s_{hidden_dim}hd_{learning_rate}lr_{decay_rate}dr_{cost_loss_type}_{rank_loss_type}_{rank_method}_{loss_weight}lw{results_suffix}"
)
os.makedirs(results_dir, exist_ok=True)
loss_json_path = os.path.join(results_dir, f"loss.json")

def save_loss(train_losses, test_losses):
    loss_dict = {"train": train_losses, "test": test_losses}
    with open(loss_json_path, "w") as f:
        json.dump(loss_dict, f)

def save_predictions(predictions, epoch):
    pred_path = os.path.join(results_dir, f"predictions_epoch{epoch+1}.json")
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)

def save_model(model, path):
    torch.save(model.state_dict(), path)


if train_ids_json or test_ids_json:
    dataset = TSPDataset(
            instance_dirs=test_instance_dirs,
            execution_dirs=execution_dirs,
            algorithms=algorithms,
            time_limit=time_limit,
            num_neighbors=num_neighbors,
            filtered_ids_json=test_ids_json)
    test_dataset = dataset
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
print(f"Testing dataset size: {len(test_dataset)}")

def test(net, save_pred=False, epoch=None):
    # Set evaluation mode
    net.eval()

    # Initialize running data
    running_loss = 0.0
    running_nb_data = 0
    predictions = []

    with torch.no_grad():
        start_test = time.time()
        for batch in test_loader:
            # Convert batch to torch Variables
            x_edges = batch['edge_matrix'].to(device)
            x_edges_values = batch['edge_values'].to(device)
            x_nodes_coord = batch['node_coords'].to(device)
            label_cost = batch['label_cost'].to(device)
            label_rank = batch['label_rank'].to(device)
            # Forward pass
            y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes_coord, label_cost, label_rank)
            loss = loss.mean()

            # Update running data
            running_nb_data += 1
            running_loss += loss.data.item()

            if save_pred:
                # detach and move to cpu for saving
                pred_np = y_preds.cpu().numpy()
                label_cost_np = label_cost.cpu().numpy()
                label_rank_np = label_rank.cpu().numpy()
                for i in range(pred_np.shape[0]):
                    predictions.append({
                        "instance_id": batch["instance_id"][i],
                        "pred_costs": pred_np[i].tolist(),
                        "label_costs": label_cost_np[i].tolist(),
                        "label_ranks": label_rank_np[i].tolist()
                    })

    loss = running_loss/ running_nb_data

    if save_pred and epoch is not None:
        save_predictions(predictions, epoch)

    return time.time() - start_test, loss

def update_learning_rate(optimizer, lr):
  for param_group in optimizer.param_groups:
      param_group['lr'] = lr
  return optimizer

# Define optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

train_losses = []
valid_losses = []
test_losses = []

for epoch in range(max_epochs):
    train_time, train_loss = train_one_epoch(net, optimizer)
    train_losses.append(train_loss)
    print(f"Epoch: {epoch}, Train Loss: {train_loss}")

    if epoch % decay_every == 0 and epoch > 0:
        learning_rate /= decay_rate
        optimizer = update_learning_rate(optimizer, learning_rate)

    if epoch % test_every == 0 or epoch == max_epochs-1:
        valid_loss = validate_one_epoch(net)
        if valid_loss is not None:
            valid_losses.append(valid_loss)
            print(f"Epoch: {epoch}, Valid Loss: {valid_loss}\n")
        else:
            print(f"Epoch: {epoch}, Validation skipped\n")
        save_loss(train_losses, valid_losses)

save_model(net, os.path.join(results_dir, f"final_model.pth"))

def save_loss_curve(save_path):
    with open(loss_json_path, "r") as f:
        loss_dict = json.load(f)
    train_loss = loss_dict["train"]
    valid_loss = loss_dict["test"]
    fig, ax = plt.subplots()
    ax.plot(train_loss, color='green', label='Train Loss')
    ax.plot([i * test_every for i in range(len(valid_loss))], valid_loss, color='orange', label='Valid Loss')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)

save_loss_curve(save_path=os.path.join(results_dir, f"loss_curve.png"))

# Only test once at the end
test_time, test_loss = test(net, save_pred=True, epoch=max_epochs-1)
print(f"Final Test Loss: {test_loss}")
