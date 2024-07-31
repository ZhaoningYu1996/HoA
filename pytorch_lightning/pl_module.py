import os
import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
import lightning as L
from sklearn.decomposition import PCA
from ogb.graphproppred.mol_encoder import AtomEncoder
from ogb.graphproppred import Evaluator
from torch_geometric.nn import BatchNorm, global_mean_pool
from sklearn.decomposition import PCA as SKLearnPCA
from torch_geometric.nn import GCNConv, GraphConv, GINEConv

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
    
class MODEL(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # self.n_embedding = nn.Linear(input_dim, hidden_dim)
        self.mlp = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm_1 = BatchNorm(hidden_dim)
        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)
        self.predict = nn.Linear(hidden_dim, output_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm_2 = BatchNorm(hidden_dim)
        self.mlp3 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm_3 = BatchNorm(hidden_dim)
        self.mlp4 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm_4 = BatchNorm(hidden_dim)
        self.mlp5 = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm_5 = BatchNorm(hidden_dim)
    
    def forward(self, x, adj, batch):
        x = self.embedding_h(x)
        if adj.is_sparse:
            x = x + torch.sparse.mm(adj, x)
        else:
            x = x + torch.mm(adj, x)
        x = self.mlp(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)
        x = self.mlp4(x)
        x = self.batch_norm_4(x)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.mlp2(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)
        x = self.mlp5(x)
        x = self.batch_norm_5(x)
        x = self.predict(x)
        return x

class Encoder(L.LightningModule):
    def __init__(self, data_name, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.evaluator = Evaluator(name=data_name)
        self.encoder = MODEL(input_dim, hidden_dim, output_dim)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.y_true = []
        self.y_pred = []
        self.y_true_train = []
        self.y_pred_train = []
        self.loss = 0
    
    def edge_index_to_adjacency_matrix(self, edge_index, num_nodes, edge_weight=None):
        # If no edge weights are provided, use a tensor of ones
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float32).to(self.device)
        
        # Create a sparse adjacency matrix
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([num_nodes, num_nodes]))
        
        return adj
        
    def training_step(self, batch):
        x, edge_index, y, batch = batch.x, batch.edge_index, batch.y, batch.batch
        adj = self.edge_index_to_adjacency_matrix(edge_index, x.size(0))
        pred_1 = self.encoder(x, adj, batch)
        is_labeled = y == y
        loss = self.criterion(pred_1[is_labeled], y[is_labeled].float())
        self.y_true_train.append(y)
        self.y_pred_train.append(pred_1)
        self.loss += loss

        # Convert to dense
        dense_adj = adj.to_dense()

        # Generate noise of the same shape
        noise = torch.randn(dense_adj.shape).to(self.device)

        # Add noise to all elements
        noise_adj = dense_adj + noise

        pred_2 = self.encoder(x, noise_adj, batch)
        is_labeled = y == y
        loss += self.criterion(pred_2[is_labeled], y[is_labeled].float())
        self.y_true_train.append(y)
        self.y_pred_train.append(pred_2)
        self.loss += loss
        return loss
    
    def validation_step(self, batch):
        # this is the validation loop
        x, edge_index, y, batch = batch.x, batch.edge_index, batch.y, batch.batch
        adj = self.edge_index_to_adjacency_matrix(edge_index, x.size(0))
        x = self.encoder(x, adj, batch)
        self.y_true.append(y)
        self.y_pred.append(x)
        return x

    def test_step(self, batch):
        # this is the test loop
        x, edge_index, y, batch = batch.x, batch.edge_index, batch.y, batch.batch
        adj = self.edge_index_to_adjacency_matrix(edge_index, x.size(0))
        x = self.encoder(x, adj, batch)
        self.y_true.append(y)
        self.y_pred.append(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
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
        self.log(f'{phase}_rocauc_score', results['rocauc'], on_epoch=True, prog_bar=True)
    
    def on_train_epoch_end(self) -> None:
        self.handle_epoch_end(self.y_true_train, self.y_pred_train, 'train')
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

