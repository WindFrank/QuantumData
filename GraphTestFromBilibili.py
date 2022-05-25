import torch
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

dataset = Planetoid(root='./TestData', name="Cora")
print('success!')

print(dataset)
print('number of graph:', len(dataset))
print('number of classes:', dataset.num_classes)
print('number of features:', dataset.num_features)
print('number of node features:', dataset.num_node_features)
print('number of edge features:', dataset.num_edge_features)

print(dataset.data)

print("edge_index:\t\t", dataset.data.edge_index.shape)
print(dataset.data.edge_index)
print("train mask:\t\t", dataset.data.train_mask.shape)

data = dataset[0]


class MyOwnDataset(Dataset):
    def get(self, idx: int) -> Data:
        return self.data[idx]

    def len(self) -> int:
        return len(data)

    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv = SAGEConv(dataset.num_features,
                             dataset.num_classes,
                             aggr="max")  # max, mean, add...)

    def forward(self):
        x = self.conv(data.x, data.edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    out = model()[data.train_mask]
    label = data.y[data.train_mask]
    loss = F.nll_loss(out, label)
    loss.backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        temp = logits[mask]
        temp2 = logits[mask].max(1)
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

best_val_acc = test_acc = 0
for epoch in range(1, 100):
    train()
    _, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Val: {:.4f}, Test{:.4f}'

    if epoch % 10 == 0:
        print(log.format(epoch, best_val_acc, test_acc))