"""This file contains all functions and classes related to clustering data points.
Clustering is performed with kmeans of faiss or KMeans of sklearn.cluster."""
import faiss
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


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


class KeyCluster(SphericalCluster):
    """Spherical cluster where individual points can be edited via their key."""
    def __init__(self, name, keys, points):
        self.keys = np.array(keys)
        super(KeyCluster, self).__init__(name=name, points=points)

    def insert(self, keys, points):
        if np.any(np.isin(keys, self.keys)):
            raise AttributeError('Ids of nodes are already used in Cluster. If this is desired use "update".')
        self.keys = np.append(self.keys, keys)
        SphericalCluster.insert(self, points)

    def remove(self, indices):
        self.keys = np.delete(self.keys, indices)
        SphericalCluster.remove(self, indices)

    def update(self, keys, points):
        """overwrite values of existent nodes and append new ones"""
        existent = np.isin(keys, self.keys)
        self.points[np.where(self.keys == keys)[0]] = points[existent]         # overwrite
        self.keys = np.append(self.keys, keys(existent.__invert__()))          # append
        SphericalCluster.insert(self, points[existent.__invert__()])        # recompute

    def get_members(self):
        return self.keys


def distortion(vectors, centroids):
    """The k-means algorithm tries to minimize distortion,
    which is defined as the sum of the squared distances between each observation vector and its dominating centroid.
    """
    dist_sq = cdist(vectors, centroids, metric='sqeuclidean')       # squared distance from all observations to all centroids
    dist_sq = np.min(dist_sq, axis=1)           # only take distance for the dominating centroid for each sample
    return sum(dist_sq)


def kmeans(vectors, n_centroids, use_faiss=True, gpu=False, niter=20):
    if use_faiss:
        km = faiss.Kmeans(vectors.shape[1], n_centroids, niter, gpu=gpu)
        km.train(vectors.astype(np.float32))
        km.cluster_centers_ = km.centroids
        _, labels = km.assign(vectors.astype(np.float32))
        km.labels_ = labels
    else:
        km = KMeans(n_centroids)
        km.fit(vectors)

    return km


def auto_kmeans(vectors, k_max=10, use_faiss=True, gpu=False, niter=20, verbose=False):
    """Predicts optimal clustering (number of clusters) using the elbow method on the distortion."""
    # compute clusters for each k up to k_max
    kmns = [kmeans(vectors, k, use_faiss, gpu, niter) for k in np.arange(1, k_max+1, dtype=int)]
    if use_faiss:
        scores = np.array([distortion(vectors, km.centroids) for km in kmns])
    else:
        scores = np.array([abs(km.score(vectors)) for km in kmns])
    # predict the elbow point / optimal number of clusters
    if scores.max() < 100:
        best = 1
    else:
        rel_change = np.array([-(scores[i + 1] - scores[i]) / scores[i] for i in range(0, len(scores) - 1)])
        if np.all(rel_change <= 0.75):
            best = np.argmax(rel_change)
        else:
            best = np.where(rel_change > 0.75)[0][-1]
        best += 1       # compensate for shift due to relative change
    if verbose:
        print('Predict {} cluster.'.format(best))

    return kmns[best]
