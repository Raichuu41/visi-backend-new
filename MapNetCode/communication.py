import numpy as np
import pandas as pd
import json
import requests
from sklearn.cluster import KMeans
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def make_nodes(position, name=None, label=None, index=False):
    """Args:
        names (list or np.ndarray): names of nodes, e.g. image ID
        position (2 x N np.ndarray): x,y coordinates of datapoints
        label (list or np.ndarray), optional: labels for datapoints (to assign various labels to one point choose list
            of lists or np.ndarray of shape N x L, where L is the number of labels)"""
    # format label
    if label is None:
        label = [None] * len(position)
    label = np.array(label)
    assert label.ndim <= 2, 'a node can only have one dimensional labels'
    if label.ndim == 1:
        label = label.reshape(len(label), -1)
    label = label.tolist()

    # make nodes
    nodes = pd.DataFrame({'x': position[:, 0], 'y': position[:, 1], 'labels': list(label)})
    if name is not None:
        nodes['name'] = name
    if index is True:
        nodes['index'] = np.arange(0, len(position))
    return nodes.to_dict(orient='index')


def read_nodes(json_nodes):
    df = pd.DataFrame.from_dict(json_nodes, orient='index')
    df.index = df['index']
    return df.sort_index()


def send_payload(nodes, socket_id, categories=None):
    headers = {'content-type': 'application/json'}
    payload = {'nodes': nodes, 'socket_id': socket_id}
    if categories is not None:
        payload['categories'] = categories
    #print(payload)
    requests.post("http://localhost:3000/api/v1/updateEmbedding", data=json.dumps(payload), headers=headers)


def toyClustering(n_cluster, pos_max, var_max):
    center = []
    for i in range(n_cluster):
        while True:
            c = pos_max * np.random.random(2)
            if np.all([np.linalg.norm(c - cc) > 3 * var_max for cc in center]):
                center.append(c)
                break

    position = []
    for c in center:
        n = np.random.randint(10, 30)
        var_x, var_y = var_max * np.random.random(2)
        x = np.random.normal(loc=c[0], scale=var_x, size=n)
        y = np.random.normal(loc=c[1], scale=var_y, size=n)
        position.append(np.stack([x, y]).transpose())

    return np.concatenate(position)


def get_clustering(position, k_max=10, plot=False):
    # n_cluster = np.random.randint(1, 5)
    # position = toyClustering(n_cluster, 20, 1)
    km = [KMeans(n_clusters=int(i)) for i in range(1, k_max)]
    score = np.array([km[i].fit(position).score(position) for i in range(len(km))])
    # predict the elbow point / optimal number of clusters
    if np.abs(score).max() < 100:
        prediction = 1
    else:
        rel_change = np.array([(score[i+1] - score[i]) / np.abs(score[i]) for i in range(0, len(score) - 1)])
        prediction = np.where(rel_change > 0.75)[0]
        if len(prediction) == 0:
            prediction = np.argmax(rel_change) + 2
        else:
            prediction = prediction[-1] + 2
    # print(prediction, n_cluster)

    # get the cluster center
    km = km[prediction-1]

    if plot:
        plt.figure()
        plt.scatter(position[:, 0], position[:, 1], c=km.labels_, cmap=plt.cm.jet)
        plt.show(block=False)

    return km.labels_, km.cluster_centers_


def infer_groups(position, labels, buffer=1.0, merge='closest'):
    """merge can be closest or all"""

    class Cluster(object):
        def __init__(self, points, label):
            self.points = points
            self.label = label
            self.center = self.get_center()
            self.radius = self.get_radius()

        def get_center(self):
            return np.mean(self.points, axis=0)

        def get_radius(self):
            return np.linalg.norm(self.points - self.center, axis=1).max()

        def insert(self, points):
            self.points = np.concatenate([self.points, points])
            self.center = self.get_center()
            self.radius = self.get_radius()

        def has_overlap(self, cluster, buffer=1.0):
            dist = np.linalg.norm(self.center - cluster.center)
            return dist <= buffer * (self.radius + cluster.radius)

        def distance(self, cluster):
            return np.linalg.norm(self.center - cluster.center)

    if not merge in ['closest', 'all']:
        raise AttributeError('merge has to be "closest" or "all"')

    new_samples = np.where(labels == -1)[0]
    old_samples = np.setdiff1d(range(len(labels)), new_samples)

    new_labels, new_centers = get_clustering(position[new_samples])
    if len(old_samples) == 0:
        return new_labels

    # indicate new groups by adding 0.5
    new_labels = new_labels + 0.5
    labels[new_samples] = new_labels
    new_groups = np.unique(new_labels)

    # infer old groups
    old_groups = np.unique(labels[old_samples])
    
    # make clusters
    clusters = np.array([Cluster(points=position[labels == g], label=g) for g in old_groups])
    new_clusters = np.array([Cluster(points=position[labels == g], label=g) for g in new_groups])

    # update clusters / merge new clusters into old
    for n in range(len(new_clusters)):
        new_clstr = new_clusters[n]

        # find overlap with the old clusters
        overlap = np.full(len(clusters), fill_value=float('inf'))
        for o in range(len(clusters)):
            old_clstr = clusters[o]
            if new_clstr.has_overlap(old_clstr, buffer=buffer):
                overlap[o] = new_clstr.distance(old_clstr)

        # merge
        if np.isfinite(overlap).any():
            if merge == 'closest':          # choose the closest cluster
                clusters[np.argmin(overlap)].insert(new_clstr.points)
            else:       # merge all with overlap
                idcs = np.where(np.isfinite(overlap))[0]
                points = np.concatenate([oc.points for oc in clusters[idcs]])
                label = np.min([oc.label for oc in clusters[idcs]])         # choose smallest group label
                points = np.concatenate([points, new_clstr.points])
                cluster = Cluster(points, label)
                clusters = np.delete(clusters, idcs)
                clusters = np.append(clusters, cluster)

        else:
            new_clstr.label = np.max([oc.label for oc in clusters]) + 1         # give cluster a valid
            clusters = np.append(clusters, new_clstr)

    # renew the labels
    labels = -1 * np.ones(len(position))
    for c in clusters:
        mask = np.all(np.isin(position, c.points), axis=1)
        labels[mask] = c.label

    assert np.all(labels != -1), 'labelling went wrong'

    return labels


if __name__ == '__main__':
    pass

