import random

import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from torch_geometric.transforms import RadiusGraph, ToDense
from torch_scatter import scatter_mean


class Encode:
    def __init__(self, encoder, batch_size, grad_enabled=False):
        self.encoder = encoder
        self.batch_size = batch_size
        self.grad_enabled = grad_enabled

    def __call__(self, data):
        prev_mode = self.encoder.training
        self.encoder.eval()
        with torch.set_grad_enabled(self.grad_enabled):
            data.x = torch.cat([self.encoder(x).flatten(1) for x in data.x.split(self.batch_size)])
        self.encoder.train(prev_mode)
        return data


class AdaptiveRadiusGraph:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, data):
        r = self.calculate_radius(data.area, len(data.x))
        return RadiusGraph(r)(data)

    def calculate_radius(self, area, num_nodes):
        factor = np.sqrt(area / (num_nodes * 18000))
        if factor < 1.3:
            factor = 1
        r = int(factor * self.epsilon)
        return r


class NodeDropout:
    def __init__(self, p, min_nodes=None, max_nodes=None):
        self.p = p
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes

    def __call__(self, data):
        node_mask = torch.bernoulli(torch.full((len(data.x),), 1-self.p)).to(torch.bool)

        if self.min_nodes is not None:
            while torch.sum(node_mask) < self.min_nodes:
                node_mask[random.randrange(len(node_mask))] = True

        if self.max_nodes is not None:
            while torch.sum(node_mask) > self.max_nodes:
                node_mask[random.randrange(len(node_mask))] = False

        if data.x is not None:
            data.x = data.x[node_mask]

        if data.pos is not None:
            data.pos = data.pos[node_mask]

        if data.norm is not None:
            data.norm = data.norm[node_mask]

        if data.edge_index is not None:
            data.edge_index = None

        if data.edge_attr is not None:
            data.edge_attr = None

        return data


class Cluster:
    def __init__(self, min_nodes, max_nodes, distance_threshold):
        self.min_nodes = min_nodes
        self.distance_threshold = distance_threshold
        self.to_dense = ToDense(max_nodes)

    def __call__(self, data):
        if len(data.x) < self.min_nodes:
            return data

        connectivity = self.to_dense(data).adj
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=self.distance_threshold,
                                             connectivity=connectivity)
        labels = clustering.fit_predict(data.x)
        labels = torch.from_numpy(labels)

        data.x = scatter_mean(data.x, labels, dim=0)
        data.pos = scatter_mean(data.pos, labels, dim=0) if data.pos is not None else None

        if data.edge_index is not None:
            new_edges = []
            edges = data.edge_index.T
            for edge in edges:
                if labels[edge[0]] != labels[edge[1]]:
                    new_edges.append((labels[edge[0]], labels[edge[1]]))
            data.edge_index = torch.from_numpy(np.unique(edges, axis=0)).T

        return data