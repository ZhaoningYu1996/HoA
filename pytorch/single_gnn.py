import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred import Evaluator
from model import PNA_Net, GCN_Net, GIN_Net, GINE_Net, GATv2_Net, ARMA_Net, GCNv2_Net, TAG_Net, Transformer_Net
from tqdm import tqdm

def train():
    model.train()
    for data in train_loader:
        data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        # out = model(data.x, data.edge_index, data.batch)
        is_labeled = data.y == data.y
        loss = criterion(out[is_labeled], data.y.float()[is_labeled])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(loder):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loder:
            data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            # out = model(data.x, data.edge_index, data.batch)
            y_true.append(data.y)
            y_pred.append(out)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().detach().numpy()
    input_dict = {'y_true': y_true, 'y_pred': y_pred}

    return evaluator.eval(input_dict)

data_name = 'ogbg-molbbbp'
dataset = PygGraphPropPredDataset(name=data_name, root='./tmp/dataset')

evaluator = Evaluator(name=data_name)

loop_n = 10
list_train_rocauc = [[] for i in range(loop_n)]
list_valid_rocauc = [[] for i in range(loop_n)]
list_test_rocauc = [[] for i in range(loop_n)]
max_train_rocauc = []
max_valid_rocauc = []
max_test_rocauc = []

for i in range(loop_n):

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in dataset[split_idx["train"]]:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset[split_idx["train"]]:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    # model = PNA_Net(hid_channels=70, out_channels=1, num_layers=4, dropout=0.3, deg=deg).to(device)
    model = GINE_Net(hid_channels=300, out_channels=1, num_layers=5, dropout=0.5).to(device)
    # model = GATv2_Net(hid_channels=75, out_channels=1, heads=4, num_layers=5, dropout=0.5).to(device)
    # model = ARMA_Net(hid_channels=300, out_channels=1, num_layers=5, dropout=0.3).to(device)
    # model = GCNv2_Net(hid_channels=300, out_channels=1, num_layers=5, dropout=0.5).to(device)
    # model = TAG_Net(hid_channels=300, out_channels=1, num_layers=4, dropout=0.5).to(device)
    # model = Transformer_Net(hid_channels=75, out_channels=1, heads=4, num_layers=5, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    max_valid = 0.0
    for epoch in tqdm(range(1,101)):
        train()
        train_acc = test(train_loader)
        valid_acc = test(valid_loader)
        test_acc = test(test_loader)
        list_train_rocauc[i].append(list(train_acc.values()))
        list_valid_rocauc[i].append(list(valid_acc.values()))
        list_test_rocauc[i].append(list(test_acc.values()))
        if list(valid_acc.values())[0] >= max_valid:
            max_valid = list(valid_acc.values())[0]
            torch.save(model.state_dict(), 'models/PNA.pth')
        print(
            f'Epoch: {epoch:03d}, Train AUC: {train_acc}, Valid AUC :{valid_acc}, Test AUC: {test_acc}')

    max_valid_rocauc_result = max(list_valid_rocauc[i])
    index_test = list_valid_rocauc[i].index(max_valid_rocauc_result)
    max_train_rocauc.append(list_train_rocauc[i][index_test])
    max_valid_rocauc.append(list_valid_rocauc[i][index_test])
    max_test_rocauc.append(list_test_rocauc[i][index_test])

for i in range(loop_n):
    print('max_train_rocauc -> max_valid_rocauc -> max_test_rocauc:', max_train_rocauc[i], ' -> ', max_valid_rocauc[i],
          ' -> ', max_test_rocauc[i])


