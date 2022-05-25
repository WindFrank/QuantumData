# import torch

'''
So there are 4 nodes in the graph, v1 … v4, each of which is associated with a 2-dimensional feature vector,
and a label y indicating its class. 
These two can be represented as FloatTensors:
'''
# x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
# y = torch.tensor([0, 1, 0, 1], dtype=torch.float)
from torch_geometric.loader import DataLoader

'''
The graph connectivity (edge index) should be confined with the COO format, 
i.e. the first list contains the index of the source nodes, 
while the index of target nodes is specified in the second list.
'''

# edge_index = torch.tensor([[0, 1, 2, 0, 3], [1, 0, 1, 3, 2]], dtype=torch.long)

'''
Note that the order of the edge index is irrelevant to the Data object you create since such information 
is only for computing the adjacency matrix. 
Therefore, the above edge_index express the same information as the following one.
edge_index = torch.tensor([[0, 2, 1, 0, 3],
                           [3, 1, 0, 1, 2]], dtype=torch.long)
'''

'''
Putting them together, we can create a Data object as shown below:
'''
import torch
from torch_geometric.data import Data


'''x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)
y = torch.tensor([0, 1, 0, 1], dtype=torch.float)

edge_index = torch.tensor([[0, 2, 1, 0, 3],
                           [3, 1, 0, 1, 2]], dtype=torch.long)

data = Data(x=x, y=y, edge_index=edge_index)'''
'''Data(edge_index=[2, 5], x=[4, 2], y=[4])'''

'''
Dataset

The dataset creation procedure is not very straightforward, 
but it may seem familiar to those who’ve used torchvision, 
as PyG is following its convention. 
PyG provides two different types of dataset classes, InMemoryDataset and Dataset. 
As they indicate literally, the former one is for data that fit in your RAM,
while the second one is for much larger data. 
Since their implementations are quite similar, I will only cover InMemoryDataset.

To create an InMemoryDataset object, there are 4 functions you need to implement:

    raw_file_name()

It returns a list that shows a list of raw, unprocessed file names. 
If you only have a file then the returned list should only contain 1 element. 
In fact, you can simply return an empty list and specify your file later in process().


    processed_file_names()
    
Similar to the last function, it also returns a list containing the file names of all the processed data. 
After process() is called, Usually, the returned list should only have one element, 
storing the only processed data file name.


    download()
    
This function should download the data you are working on to the directory as specified in self.raw_dir. 
If you don’t need to download data, simply drop in "pass" in the function.


    process()
    
This is the most important method of Dataset. 
You need to gather your data into a list of Data objects. 
Then, call self.collate() to compute the slices that will be used by the DataLoader object. 
The following shows an example of the custom dataset from PyG official website.
'''

'''
import torch
from torch_geometric.data import InMemoryDataset


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
'''

'''
I will show you how I create a custom dataset from the data provided in RecSys Challenge 2015 later in this article.


DataLoader

The DataLoader class allows you to feed data by batch into the model effortlessly. 
To create a DataLoader object, you simply specify the Dataset and the batch size you want.
'''

# loader = DataLoader(dataset, batch_size=512, shuffle=True)

'''
Every iteration of a DataLoader object yields a Batch object, 
which is very much like a Data object but with an attribute, 
“batch”. It indicates which graph each node is associated with. 
Since a DataLoader aggregates x, y, and edge_index from different samples/ graphs into Batches, 
the GNN model needs this “batch” information to know which nodes belong to the same graph within a batch 
to perform computation.
'''

# for batch in loader:
#    batch
#    >>> Batch(x=[1024, 21], edge_index=[2, 1568], y=[512], batch=[1024])

embed_dim = 128
from torch_geometric.nn import TopKPooling, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
y = torch.tensor([0, 1, 0, 1], dtype=torch.float)

edge_index = torch.tensor([[0, 2, 1, 0, 3],
                           [3, 1, 0, 1, 2]], dtype=torch.long)


'''这里表示意思为，各个数据的行与列。如在本例中，x有四个表示向量，每个向量是二维，那么x=[4, 2]'''
data1 = Data(x=x, y=y, edge_index=edge_index)
data2 = Data(x=x, y=y, edge_index=edge_index)

df = [data1, data2]
train_loader = DataLoader(df, 1)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = SAGEConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.item_embedding = torch.nn.Embedding(num_embeddings=len(df), embedding_dim=embed_dim)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)
        x = x.squeeze(1)

        x = F.relu(self.conv1(x, edge_index))

        x, edge_index, _, batch, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))

        x, edge_index, _, batch, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        x, edge_index, _, batch, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x


def train():
    model.train()

    loss_all = 0
    for data in df:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(df)


device = torch.device('cuda')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
crit = torch.nn.BCELoss()
for epoch in range(1):
    train()

