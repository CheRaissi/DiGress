import os

import torch
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset
import torch_geometric.utils

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class DivideTreeDataset(Dataset):
    def __init__(self, data_file):
        """ This class is used to load the divide tree datasets. """
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        filename = os.path.join(base_path, data_file)
        self.adjs, self.coords, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max = torch.load(
            filename)
        print(f'Dataset {filename} loaded from file')

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        adj = self.adjs[idx]
        coords = self.coords[idx]
        n = adj.shape[-1]
        # X = torch.ones(n, 1, dtype=torch.float)

        s_num = n // 2
        p_num = n - s_num
        d_nodes = torch.LongTensor(p_num * [1] + s_num *[0])
        X = F.one_hot(d_nodes,num_classes=2).to(torch.float) # Peaks and saddles type

        y = torch.zeros([1, 0]).float()
        edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        num_nodes = n * torch.ones(1, dtype=torch.long)
        data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr, pos=coords,
                                         y=y, idx=idx, n_nodes=num_nodes)
        return data


class TreeDataset11(DivideTreeDataset):
    def __init__(self):
        super().__init__('Alaska_11.pt')

class TreeDataset51(DivideTreeDataset):
    def __init__(self):
        super().__init__('Alaska_51.pt')

class DivideTreeDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=201):
        super().__init__(cfg)
        self.n_graphs = n_graphs
        self.prepare_data()
        self.inner = self.train_dataloader()

    def __getitem__(self, item):
        return self.inner[item]

    def prepare_data(self, graphs):
        test_len = int(round(len(graphs) * 0.2))
        train_len = int(round((len(graphs) - test_len) * 0.8))
        val_len = len(graphs) - train_len - test_len
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        splits = random_split(graphs, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(1234))

        datasets = {'train': splits[0], 'val': splits[1], 'test': splits[2]}
        super().prepare_data(datasets)

class DivideTreeDataModule(DivideTreeDataModule):
    def prepare_data(self):
        graphs = TreeDataset11()
        return super().prepare_data(graphs)



class DivideTreeDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'DivideTree'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types() #torch.Tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

