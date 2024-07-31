import torch
import torch.nn as nn
from torch_geometric.nn import PNAConv, GCNConv, GraphConv, GINConv, GINEConv, GATConv, GATv2Conv, TransformerConv, ARMAConv, TAGConv, SAGEConv
from torch_geometric.nn import global_mean_pool, BatchNorm

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch.nn.functional as F

model_mapping: {
    'pna': PNAConv,
    'gcnv2': GraphConv,
    'gatv2': GATv2Conv,
    'arma': ARMAConv,
    'tag': TAGConv,
    'gine': GINEConv,
}

class PNA_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, deg, dropout):
        super(PNA_Net, self).__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(PNAConv(in_channels=hid_channels, out_channels=hid_channels,aggregators=aggregators,scalers=scalers,deg=deg, post_layers=1))
            self.batch_norms.append(BatchNorm(hid_channels))

        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, return_embedding=False):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
        
        if return_embedding:
            return x
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class GCN_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, dropout):
        super(GCN_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(GCNConv(in_channels=hid_channels, out_channels=hid_channels, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class GCNv2_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, dropout):
        super(GCNv2_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(GraphConv(in_channels=hid_channels, out_channels=hid_channels, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch, return_embedding=False):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
        
        if return_embedding:
            return x
        
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class SAGE_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, dropout):
        super(SAGE_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(SAGEConv(in_channels=hid_channels, out_channels=hid_channels, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class GIN_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, dropout):
        super(GIN_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(GINConv(nn.Sequential(nn.Linear(hid_channels, hid_channels), nn.ReLU(), nn.Linear(hid_channels, hid_channels))))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class GINE_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, dropout):
        super(GINE_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(GINEConv(nn.Sequential(nn.Linear(hid_channels, hid_channels), nn.ReLU(), nn.Linear(hid_channels, hid_channels))))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        self.embedding_b = BondEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, edge_attr, batch, return_embedding=False):
        x = self.embedding_h(x)
        e = self.embedding_b(edge_attr)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index, e)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
        
        if return_embedding:
            return x
        
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class GAT_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, heads, num_layers, dropout):
        super(GAT_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(GATConv(in_channels=hid_channels, out_channels=hid_channels, heads=heads, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class GATv2_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, heads, num_layers, dropout):
        super(GATv2_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(GATv2Conv(in_channels=hid_channels, out_channels=hid_channels, heads=heads, concat=False, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch, return_embedding=False):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
        
        if return_embedding:
            return x
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class Transformer_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, heads, num_layers, dropout):
        super(Transformer_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(TransformerConv(in_channels=hid_channels, out_channels=hid_channels, heads=heads, concat=False, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
            
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class ARMA_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, dropout):
        super(ARMA_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(ARMAConv(in_channels=hid_channels, out_channels=hid_channels, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch, return_embedding=False):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
        
        if return_embedding:
            return x
        
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
class TAG_Net(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, num_layers, dropout):
        super(TAG_Net, self).__init__()
        
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropout = dropout
        for _ in range(self.num_layers):
            self.layers.append(TAGConv(in_channels=hid_channels, out_channels=hid_channels, add_self_loops=True))
            self.batch_norms.append(BatchNorm(hid_channels))
        
        self.mlp = torch.nn.Linear(hid_channels, out_channels)
        self.embedding_h = AtomEncoder(emb_dim=hid_channels)
        
    def forward(self, x, edge_index, batch, return_embedding=False):
        x = self.embedding_h(x)
        
        for i in range(len(self.layers)):
            x_h = x
            x = self.layers[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = x_h + x
            x = F.dropout(x, self.dropout, training=self.training)
        
        if return_embedding:
            return x
        x = global_mean_pool(x, batch)
        x = self.mlp(x)

        return x
    
# class SubModel(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super(SubModel, self).__init__()
        
#         self.models = torch.nn.ModuleList()

class MoGNNs(torch.nn.Module):
    def __init__(self, hid_channels, out_channels, model_names, model_list, norm_list, num_layers_list, dropout, atten_dropout):
        super(MoGNNs, self).__init__()

        self.hid_channels = hid_channels
        self.sorted_dim = sorted(list(set(hid_channels)))
        # self.models = model_list
        self.models = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for model in model_list:
            self.models.append(model)
        for model in norm_list:
            self.batch_norms.append(model)
        self.models_names = model_names
        # self.batch_norms = norm_list
        self.Q_weight = torch.nn.ModuleList()
        self.K_weight = torch.nn.ModuleList()
        self.num_layers_list = num_layers_list
        self.dropout = dropout
        
        self.atten_dropout = nn.Dropout(p=atten_dropout)
        
        self.mlp = torch.nn.Linear(max(hid_channels), out_channels)
        
        # for _ in num_layers_list:
        #     self.Q_weight.append(nn.Linear(max(hid_channels), max(hid_channels)))
        #     self.K_weight.append(nn.Linear(max(hid_channels), max(hid_channels)))
        
    def pad(self, x):

        # Find the maximum size of the second dimension
        max_size = max(tensor.size(1) for tensor in x)

        # Pad the tensors and collect them in a new list
        new_x = []
        
        tensor_size = []
        for tensor in x:
            # Calculate how much padding is needed
            tensor_size.append(tensor.size(1))
            padding = max_size - tensor.size(1)
            # Pad the tensor and add it to the list
            padded_tensor = F.pad(tensor, (0, padding))
            new_x.append(padded_tensor)

        return new_x, tensor_size
    
    def unpad(self, x, original_sizes):

        # Remove the padding from each tensor
        unpadded_tensors = [tensor[:, :size] for tensor, size in zip(x, original_sizes)]

        return unpadded_tensors
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # x = [x for _ in range(len(self.num_layers_list))]
        for i in range(max(self.num_layers_list)):
            indices = []
            for j in range(len(self.num_layers_list)):
                if self.num_layers_list[j] >= i+1:
                    indices.append(j)
                    if self.models_names[j] == 'GINE_Net':
                        x_h = x[j]
                        x[j] = self.models[j][i](x[j], edge_index, edge_attr)
                        x[j] = self.batch_norms[j][i](x[j])
                        x[j] = F.relu(x[j])
                        x[j] = x_h + x[j]
                        x[j] = F.dropout(x[j], self.dropout[j], training=self.training)
                    else:
                        x_h = x[j]
                        x[j] = self.models[j][i](x[j], edge_index)
                        x[j] = self.batch_norms[j][i](x[j])
                        x[j] = F.relu(x[j])
                        x[j] = x_h + x[j]
                        x[j] = F.dropout(x[j], self.dropout[j], training=self.training)

            if len(indices) > 1:
                x, size_list = self.pad(x)
                V = torch.stack(x, dim=0)[indices]
                V = V.reshape(V.size(1), V.size(0), V.size(2))
                prev_V = V
                Q = V
                K = V
                # Q = self.Q_weight[i](V)
                # K = self.K_weight[i](V)
                if not torch.equal(V, prev_V):
                    print(V)
                    print(prev_V)
                    print(stop)
                scores = torch.matmul(Q, K.transpose(-1, -2))
                atten = self.atten_dropout(nn.Softmax(dim=-1)(scores))
                atten_out = torch.matmul(atten, V)
                atten_out = atten_out.reshape(atten_out.size(1), atten_out.size(0), atten_out.size(2))
                atten_out = torch.unbind(atten_out, dim=0)
                atten_out = self.unpad(atten_out, size_list)
                for index, value in zip(indices, atten_out):
                    x[index] = value
                    # print(value)

        x, _ = self.pad(x)
        x = torch.stack(x, dim=0)
        x = torch.sum(x, dim=0)
        x = global_mean_pool(x, batch)
        out = self.mlp(x)
        
        return out
            
            
            
            
        

            
            
            
        