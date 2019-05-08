import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import math
from itertools import combinations
import multiprocessing
from functools import partial           # to be able to pass kwargs to pool.map in multiprocessing
import deepdish as dd
import os



def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def get_cluster_indices(lbl, feature, label, avg_samples_per_cluster, niter=20):
    sample_mask = label == lbl
    samples = torch.nonzero(sample_mask).view(-1).numpy()
    n_centroids = int(math.ceil(len(samples) * 1. / min(avg_samples_per_cluster, len(samples))))
    if len(samples) == n_centroids:
        return [[s] for s in samples]
    kmeans = faiss.Kmeans(feature.shape[1], n_centroids, niter, gpu=False)
    kmeans.train(feature[sample_mask].numpy())
    _, centroids = kmeans.assign(feature[sample_mask].numpy())
    cluster_indices = []
    for c in range(n_centroids):
        cluster_indices.append(torch.as_tensor(samples[(centroids == c)], dtype=torch.long, device=feature.device))
    return cluster_indices


class TripletLoss(nn.Module):
    def __init__(self, triplet_selector,  margin):
        super(TripletLoss, self).__init__()
        self.triplet_selector = triplet_selector
        self.margin = margin

    def forward(self, feature, label, weights=None):
        triplets, feature, label, weights = self.triplet_selector.get_triplets(feature, label, weights)

        if triplets is None:
            return torch.tensor(0.).type_as(feature)

        ap_dist = F.pairwise_distance(feature[triplets[:, 0]], feature[triplets[:, 1]], p=2)
        an_dist = F.pairwise_distance(feature[triplets[:, 0]], feature[triplets[:, 2]], p=2)

        losses = F.relu(ap_dist - an_dist + self.margin.type_as(ap_dist))
        if weights is not None:
            weights = weights[triplets.view(-1)].view(-1, 3).prod(dim=1).pow(1./3)
            losses = losses * weights.type_as(losses)

        return losses.sum()


class TripletSelector(object):
    def __init__(self, margin, negative_selection_fn, negative_labels=()):
        super(TripletSelector, self).__init__()
        self.negative_selection_fn = negative_selection_fn
        self.margin = margin
        self.negative_labels = negative_labels

    def _get_triplets(self, feature, label):
        labelset = torch.unique(label)

        distance_matrix = pdist(feature).cpu()

        triplets = []
        for lbl in labelset:
            if lbl in self.negative_labels:
                continue
            idx_positives = torch.nonzero(label == lbl)
            if len(idx_positives) < 2:
                continue
            idx_negatives = torch.nonzero(label != lbl)

            anchor_positives = torch.LongTensor(list(combinations(idx_positives, 2)))  # All anchor-positive pairs
            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]

            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):        # compute the loss values for all negatives and choose according to selection function
                an_distances = distance_matrix[anchor_positive[0], idx_negatives]
                loss_values = torch.clamp(ap_distance - an_distances + self.margin.type_as(ap_distance), min=0)

                hard_negative = self.negative_selection_fn(loss_values, self.margin)
                if hard_negative is not None:
                    triplets.append(torch.cat([anchor_positive, idx_negatives[hard_negative].view(1)]))

        if len(triplets) == 0:
            return None
        triplets = torch.stack(triplets)
        return triplets

    def get_triplets(self, feature, label, weights=None):
        return self._get_triplets(feature, label), feature, label, weights


class ExemplarTripletSelector(TripletSelector):
    def __init__(self, margin, negative_selection_fn, avg_samples_per_cluster, negative_labels=(), niter=20, gpu=True):
        super(ExemplarTripletSelector, self).__init__(margin, negative_selection_fn, negative_labels)
        self.avg_samples_per_cluster = avg_samples_per_cluster
        self.niter = niter
        self.gpu = gpu if not self.parallel else False

    def get_exemplars(self, feature, label, weights=None):
        labelset = torch.unique(label)
        kwargs = {'feature': feature.detach().cpu(), 'label': label, 'avg_samples_per_cluster': self.avg_samples_per_cluster,
                  'niter': 10}

        cluster = [get_cluster_indices(lbl, **kwargs) for lbl in labelset]

        # idcs_all = torch.cat([torch.cat([c for c in cluster_indices]) for cluster_indices in cluster])
        # assert len(idcs_all) == len(torch.unique(idcs_all)), 'Duplicate indices in clustering.'
        # compute the exemplars
        exemplars = []
        exemplar_labels = []
        for cluster_indices, lbl in zip(cluster, labelset):
            xmplrs = [feature[idcs].mean(dim=0) for idcs in cluster_indices]
            exemplar_labels.extend([lbl] * len(xmplrs))
            exemplars.extend(xmplrs)

        if weights is not None:
            exemplar_weights = []
            for cluster_indices, lbl in zip(cluster, labelset):
                xmplr_wghts = [weights[idcs].mean() for idcs in cluster_indices]
                exemplar_weights.extend(xmplr_wghts)
            exemplar_weights = torch.stack(exemplar_weights)
        else:
            exemplar_weights = None

        return torch.stack(exemplars), torch.stack(exemplar_labels), exemplar_weights

    def get_triplets(self, feature, label, weights=None):
        exemplars, exemplar_labels, exemplar_weights = self.get_exemplars(feature, label, weights=weights)
        return self._get_triplets(exemplars, exemplar_labels), exemplars, exemplar_labels, exemplar_weights


# selection functions

def select_hardest(loss_values, margin=None):
    if loss_values.max() == 0:
        return None
    return torch.argmax(loss_values)


def select_random(loss_values, margin=None):
    if loss_values.max() == 0:
        return None
    return torch.randint(0, len(loss_values), (1,), dtype=torch.long)[0]


def select_semihard(loss_values, margin):
    idcs = torch.nonzero((loss_values.view(-1) < margin) & (loss_values.view(-1) > 0)).view(-1)
    if len(idcs) == 0:
        return None
    choice = torch.randint(0, len(idcs), (1,), dtype=torch.long)[0]
    return idcs[choice]
