import os
import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
import lightning as L
from sklearn.decomposition import PCA
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.graphproppred import Evaluator
from torch_geometric.nn import BatchNorm, global_mean_pool
from sklearn.decomposition import PCA as SKLearnPCA
from torch_geometric.nn import GCNConv, GraphConv, GINEConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse

def check_for_nans_and_infs(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaNs")
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Infs")
    
class PCA(L.LightningModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        mean = torch.mean(x, dim=0)
        std = torch.std(x, dim=0)
        data_normalized = (x - mean) / std
        print(f"Data normalized: {data_normalized.shape}")

        # Compute SVD
        U, S, V = torch.linalg.svd(data_normalized, full_matrices=False)
        print(f"U: {U.shape}, S: {S.shape}, V: {V.shape}")
        # U, S, V = torch.pca_lowrank(data_normalized, q=1, center=True, niter=2)
        
        # Extract the first principal component
        n_components = 1
        first_component = V[:, :n_components]  # (feature_dim, n_components)
        print(f"First component: {first_component.shape}")
        transformed_data = torch.mm(data_normalized, first_component)  # (batch_size, feature_dim) * (feature_dim, n_components)
        print(f"Transformed data: {transformed_data.shape}")
        check_for_nans_and_infs(transformed_data, "Transformed data")
        
        return transformed_data
    
class GNN(L.LightningModule):
    def __init__(self, num_layers, hidden_dim, output_dim, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(GINEConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        self.mlp = torch.nn.Linear(hidden_dim, output_dim)
        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)
        self.embedding_b = BondEncoder(emb_dim=hidden_dim)
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.embedding_h(x)
        e = self.embedding_b(edge_attr)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index, e)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class Model(L.LightningModule):
    def __init__(self, num_layers, hidden_dim, output_dim, dropout):
        super().__init__()
        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)
        self.embedding_b = BondEncoder(emb_dim=hidden_dim)
        self.mlp1 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm_1 = BatchNorm(hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm_2 = BatchNorm(hidden_dim)
        self.mlp3 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm_3 = BatchNorm(hidden_dim)
        self.mlp4 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm_4 = BatchNorm(hidden_dim)
        self.predict = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.embedding_h(x)
        x = self.mlp1(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)
        x = self.mlp2(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.mlp3(x)
        x = self.batch_norm_3(x)
        x = F.relu(x)
        x = self.mlp4(x)
        x = self.batch_norm_4(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.predict(x)
        return x

class GNNEncoder(L.LightningModule):
    def __init__(self, data_name, num_layers, hidden_dim, output_dim, dropout):
        super().__init__()
        self.evaluator = Evaluator(name=data_name)
        # self.encoder = GNN(num_layers, hidden_dim, output_dim, dropout)
        self.encoder = Model(num_layers, hidden_dim, output_dim, dropout)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.y_true = []
        self.y_pred = []
        self.y_true_train = []
        self.y_pred_train = []
        self.y_true_train2 = []
        self.y_pred_train2 = []
        self.loss = 0
    
    def edge_index_to_adjacency_matrix(self, edge_index, num_nodes, edge_weight=None):
        # If no edge weights are provided, use a tensor of ones
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float32).to(self.device)
        
        # Create a sparse adjacency matrix
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([num_nodes, num_nodes]))
        
        return adj
    
    def split_data(self, data):
        # Convert edge_index to dense adjacency matrix
        adj_matrices = to_dense_adj(data.edge_index, batch=data.batch)
        return adj_matrices
    
    def add_noise_and_update_graphs(self, adj_matrices, data):
        num_graphs = adj_matrices.size(0)
        new_edge_indices = []
        new_edge_attrs = []

        for i in range(num_graphs):
            adj = adj_matrices[i]
            mean = 0
            std = 0.5
            noise = torch.randn(adj.shape).to(self.device) * std + mean
            # noise = torch.abs(noise)
            noise = F.relu(noise)
            noise = torch.tril(noise) + torch.tril(noise, diagonal=-1).T

            noisy_adj = adj + noise

            # Convert back to edge_index
            edge_index, _ = dense_to_sparse(noisy_adj)
            new_edge_indices.append(edge_index)
            # Assume new edge attributes are zeros or initialized appropriately
            new_edge_attrs.append(torch.zeros(edge_index.size(1), data.edge_attr.size(1)))
        
        # Update data object
        data.edge_index = torch.cat(new_edge_indices, dim=1)
        data.edge_attr = torch.cat(new_edge_attrs, dim=0)

        return data
        
    def training_step(self, batch):
        x, edge_index, edge_attr, y, batch = batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch

        pred = self.encoder(x, edge_index, edge_attr, batch)
        is_labeled = y == y
        loss1 = self.criterion(pred[is_labeled], y[is_labeled].float())
        self.y_true_train.append(y)
        self.y_pred_train.append(pred)
        self.loss += loss1

        return loss1

        # Convert to dense
        adj = self.edge_index_to_adjacency_matrix(edge_index, x.size(0))
        dense_adj = adj.to_dense()

        # Generate noise of the same shape
        mean = 0
        std = 0.5
        noise = torch.randn(dense_adj.shape).to(self.device) * std + mean
        # noise = torch.abs(noise)
        noise = F.relu(noise)
        noise = torch.tril(noise) + torch.tril(noise, diagonal=-1).T

        # Add noise to all elements
        noise_adj = dense_adj + noise

        # Identify new edges
        new_edges = (noise_adj != 0) & (dense_adj == 0)

        # Update edge_index and edge_attr
        new_edge_indices = new_edges.nonzero(as_tuple=False).t().int().to(self.device)
        new_edge_attrs = torch.zeros((new_edge_indices.size(1), edge_attr.size(1)), dtype=torch.int16).to(self.device)
        new_edge_attrs[:, 0] = 4
        
        # Combine old and new edge indices and attributes
        new_edge_index = torch.cat((edge_index, new_edge_indices), dim=1)
        new_edge_attr = torch.cat((edge_attr, new_edge_attrs), dim=0)
        
        pred_2 = self.encoder(x, new_edge_index, new_edge_attr, batch)
        is_labeled = y == y
        loss2 = self.criterion(pred_2[is_labeled], y[is_labeled].float())
        self.y_true_train2.append(y)
        self.y_pred_train2.append(pred_2)
        self.loss += loss2
        # return loss2

        loss = 0.5*loss1 + 0.5*loss2
        return loss
    
    def validation_step(self, batch):
        # this is the validation loop
        x, edge_index, edge_attr, y, batch = batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch

        pred = self.encoder(x, edge_index, edge_attr, batch)
        self.y_true.append(y)
        self.y_pred.append(pred)

        # adj = self.edge_index_to_adjacency_matrix(edge_index, x.size(0))
        # dense_adj = adj.to_dense()
        # mean = 0
        # std = 0.5
        # noise = torch.randn(dense_adj.shape).to(self.device) * std + mean
        # # noise = torch.abs(noise)
        # noise = F.relu(noise)
        # noise = torch.tril(noise) + torch.tril(noise, diagonal=-1).T
        # noise_adj = dense_adj + noise
        # new_edges = (noise_adj != 0) & (dense_adj == 0)
        # new_edge_indices = new_edges.nonzero(as_tuple=False).t().int().to(self.device)
        # new_edge_attrs = torch.zeros((new_edge_indices.size(1), edge_attr.size(1)), dtype=torch.int16).to(self.device)
        # new_edge_attrs[:, 0] = 4

        # edge_index = torch.cat((edge_index, new_edge_indices), dim=1)
        # edge_attr = torch.cat((edge_attr, new_edge_attrs), dim=0)

        # pred = self.encoder(x, edge_index, edge_attr, batch)
        # self.y_true.append(y)
        # self.y_pred.append(pred)
        return pred

    def test_step(self, batch):
        # this is the test loop
        x, edge_index, edge_attr, y, batch = batch.x, batch.edge_index, batch.edge_attr, batch.y, batch.batch

        pred = self.encoder(x, edge_index, edge_attr, batch)
        self.y_true.append(y)
        self.y_pred.append(pred)

        # adj = self.edge_index_to_adjacency_matrix(edge_index, x.size(0))
        # dense_adj = adj.to_dense()
        # mean = 0
        # std = 0.5
        # noise = torch.randn(dense_adj.shape).to(self.device) * std + mean
        # # noise = torch.abs(noise)
        # noise = F.relu(noise)
        # noise = torch.tril(noise) + torch.tril(noise, diagonal=-1).T
        # noise_adj = dense_adj + noise
        # new_edges = (noise_adj != 0) & (dense_adj == 0)
        # new_edge_indices = new_edges.nonzero(as_tuple=False).t().int().to(self.device)
        # new_edge_attrs = torch.zeros((new_edge_indices.size(1), edge_attr.size(1)), dtype=torch.int16).to(self.device)
        # new_edge_attrs[:, 0] = 4

        # edge_index = torch.cat((edge_index, new_edge_indices), dim=1)
        # edge_attr = torch.cat((edge_attr, new_edge_attrs), dim=0)

        # pred = self.encoder(x, edge_index, edge_attr, batch)
        # self.y_true.append(y)
        # self.y_pred.append(pred)
        return pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    def handle_epoch_end(self, target, pred, phase):
        """
        Handle common logic for training, validation, and testing epoch ends.

        Args:
            target: The target from the epoch.
            pred: The prediction from the epoch.
            phase (str): One of 'train', 'val', or 'test' to indicate the phase.
        """
        # Example of handling metric calculation
        y_true = torch.cat(target, dim=0).cpu().numpy()
        y_pred = torch.cat(pred, dim=0).cpu().detach().numpy()
        
        input_dict = {'y_true': y_true, 'y_pred': y_pred}
        results = self.evaluator.eval(input_dict)
        # print(f"{phase} results: {results}")
        
        # if phase == 'train':
        #     self.log(f'{phase}_loss', self.loss, on_epoch=True, prog_bar=True)
        self.log(f'{phase}_rocauc_score', results['rocauc'], on_epoch=True, prog_bar=False)
        # print(f"{phase} results: {results['rocauc']}")
    
    def on_train_epoch_end(self) -> None:
        self.handle_epoch_end(self.y_true_train, self.y_pred_train, 'train')
        # self.handle_epoch_end(self.y_true_train2, self.y_pred_train2, 'train')
        self.y_true_train.clear()
        self.y_pred_train.clear()
    
    def on_validation_epoch_end(self) -> None:
        self.handle_epoch_end(self.y_true, self.y_pred, 'val')
        self.y_true.clear()
        self.y_pred.clear()

    def on_test_epoch_end(self) -> None:
        self.handle_epoch_end(self.y_true, self.y_pred, 'test')
        self.y_true.clear()
        self.y_pred.clear()

