"""This file contains all the functions to infer the information from the user interface.
It is relies on Pandas Dataframes and the Cluster Class defined below."""
import pandas as pd
import numpy as np


class Cluster(object):
    def __init__(self, name, points):
        self.points = points
        self.name = name
        self.center = self.get_center()

    def get_center(self):
        return np.mean(self.points, axis=0)

    def insert(self, points):
        self.points = np.concatenate([self.points, points])
        self.center = self.get_center()

    def remove(self, indices):
        self.points = np.delete(self.points, indices, axis=0)
        self.center = self.get_center()

    def distance(self, cluster):
        return np.linalg.norm(self.center - cluster.center)

    def __repr__(self):
        return '{}:\t center: {}'.format(self.name, self.center)

    def __len__(self):
        return len(self.points)

class SphericalCluster(Cluster):
    def __init__(self, name, points):
        super(SphericalCluster, self).__init__(name=name, points=points)
        self.radius = self.get_radius()

    def get_radius(self):
        return np.linalg.norm(self.points - self.center, axis=1).max()

    def insert(self, points):
        Cluster.insert(self, points)
        self.radius = self.get_radius()

    def remove(self, indices):
        Cluster.remove(self, indices)
        self.radius = self.get_radius()

    def has_overlap(self, cluster, buffer=1.0):
        dist = np.linalg.norm(self.center - cluster.center)
        return dist <= buffer * (self.radius + cluster.radius)

    def __repr__(self):
        return '{}:\tcenter: {}\tradius: {}'.format(self.name, self.center, self.radius)


class Node(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return '({}, {})'.format(self.name, self.value)


nodes = [Node(i, np.random.rand(2)) for i in range(100)]


def nodes_to_df(nodes):
    data_dict = {n.name: n.value for n in nodes}
    columns = sorted(nodes[0].value.keys()) if isinstance(nodes[0].value, dict) else None       # ensure columns are sorted
    return pd.DataFrame.from_dict(data_dict, orient='index', columns=columns)


def df_to_nodes(df):
    return df.to_dict(orient='index')


class NodeCluster(SphericalCluster):
    def __init__(self, name, nodes):
        self.df = nodes_to_df(nodes)
        super(NodeCluster, self).__init__(name=name, points=self.df.values)

    def insert(self, nodes):
        df = nodes_to_df(nodes)
        if np.any(np.isin(df.index, self.df.index)):
            raise AttributeError('Ids of nodes are already used in Cluster. If this is desired use "update".')
        self.df = self.df.append(df)
        SphericalCluster.insert(self, df.values)

    def remove(self, indices):
        self.df = self.df.drop(index=indices)
        SphericalCluster.remove(self, indices)

    def update(self, nodes):
        """overwrite values of existent nodes and append new ones"""
        df = nodes_to_df(nodes)
        existent = np.isin(df.index, self.df.index)
        # overwrite
        self.df.iloc[existent] = df.iloc[existent]
        self.points[existent] = df.iloc[existent]

        self.df = self.df.append(df.iloc[existent.__invert__()])        # append
        SphericalCluster.insert(self, df.iloc[existent.__invert__()].values)        # recompute

    def get_members(self):
        return self.df.index

    def is_member(self, node):
        if isinstance(node, Node):
            return node.name in self.df.index
        else:       # iterable of nodes
            ids = [n.name for n in nodes]
        return np.isin(ids, self.df.index)


c = NodeCluster('c', nodes)