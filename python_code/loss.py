import torch.nn as nn
import torch
import torch.nn.functional as F
from itertools import combinations
import numpy as np
import faiss
import warnings
from copy import deepcopy

from .utils import _binary_search_perplexity
from .helpers import BalancedBatchSampler


MACHINE_EPSILON = np.finfo(np.double).eps


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class TSNELoss(nn.Module):
    def __init__(self, N, perplexity=10, early_exaggeration_fac=1, use_gpu=False):
        super(TSNELoss, self).__init__()
        self.N = N
        self.perplexity = perplexity
        self.exaggeration_fac = early_exaggeration_fac
        self.use_cuda = use_gpu
        self.beta = torch.zeros(self.N).type(torch.DoubleTensor)
        if self.use_cuda:
            self.beta = self.beta.cuda()

    @staticmethod
    def low_dim_prob(y, use_cuda=False):
        """Computes Q = {q_ij} the low-dimensional probabiltiy matrix of tSNE using a student-t distribution."""
        # make sure tensors are used
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y)
        # use high precision
        if not y.dtype == torch.float64:
            y = y.type(torch.DoubleTensor)
        if use_cuda:
            y = y.cuda()

        # compute numerator
        dist_sq = pdist(y)          # compute all pairwise distances
        numerator = torch.pow(1 + dist_sq, -1)

        # compute denominator
        idcs_upper = torch.triu(torch.ones(dist_sq.shape, dtype=torch.bool), diagonal=1) # use symmetry and discard diagonal entries
        denominator = 2.0 * torch.sum(numerator[idcs_upper])

        # compute probability and set Q_ii to zero
        Q = numerator / denominator
        Q[torch.eye(Q.shape[0], dtype=torch.bool)] = 0

        # ensure there are no zeros or negative values
        Q = torch.where(Q > 0, Q, torch.tensor(MACHINE_EPSILON).type_as(Q))
        return Q

    @staticmethod                   # rounding errors cause deviation from TSNE!
    def high_dim_prob(x, beta, perplexity, use_cuda=False):
        # make sure tensors are used
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        if not isinstance(beta, torch.Tensor):
            beta = torch.from_numpy(beta)

        # use high precision
        if not x.dtype == torch.float64:
            x = x.type(torch.DoubleTensor)
        if not beta.dtype == torch.float64:
            beta = beta.type(torch.DoubleTensor)

        if use_cuda:
            x = x.cuda()
            beta = beta.cuda()
        beta[beta == 0] = MACHINE_EPSILON     # ensure sigma is not zero

        N = len(x)
        dist_sq = pdist(x)      # compute all pairwise distances (HERE: rounding errors cause deviation from TSNE!)
        # only use k nearest neighbors
        k = 3 * perplexity + 2     # +1 to be coherent with TSNE and +1 for sample itself
        dist_sq, nn = torch.sort(dist_sq, dim=1)
        dist_sq, nn = dist_sq[:, 1:k], nn[:, 1:k]
        numerator = torch.exp(-dist_sq * beta.view(-1, 1))        # divide each row by its sigma

        # compute denominator
        denominator = torch.sum(numerator, dim=1, keepdim=True)

        # compute probability
        P = numerator / denominator

        rows = (torch.arange(0, N, dtype=torch.long).view(-1, 1) * torch.ones((N, k-1), dtype=torch.long)).view(-1)
        cols = nn.cpu().contiguous().view(-1)
        P = torch.sparse.DoubleTensor(torch.stack([rows, cols]), P.view(-1).cpu(), torch.Size([N, N]))
        P = P + P.t()

        # Normalize the joint probability distribution
        max_P = np.maximum(P._values().sum().numpy(), MACHINE_EPSILON)
        P.div_(max_P)

        P = P.to_dense()
        if use_cuda:
            P = P.cuda()
        # ensure there are no zeros or negative values
        P = torch.where(P > 0, P, torch.tensor(MACHINE_EPSILON).type_as(P))
        return P

    def _compute_beta(self, x):
        if isinstance(x, torch.Tensor):     # faiss and perplexity search require np.ndarray
            x = x.cpu().numpy()
        # use 32-bit precision for faiss
        if not x.dtype == np.float32:
            x = x.astype(np.float32)
        # compute nearest neighbors for faster approximation
        index = faiss.IndexFlatL2(x.shape[1])  # build the index
        index.add(x)
        k = 3 * self.perplexity + 2     # +1 to be coherent with TSNE and +1 for sample itself
        distances_sq, neighbors = index.search(x, k)

        _, beta = _binary_search_perplexity(distances_sq[:, 1:], neighbors[:, 1:],         # exclude sample itself
                                            self.perplexity, verbose=0)
        self.beta = torch.from_numpy(beta).type_as(self.beta)       # move to cuda if use_cuda

    def forward(self, x, y, indices=None):
        if indices is None:
            indices = torch.arange(0, self.N, dtype=torch.long)
        elif len(indices) < 3 * self.perplexity + 2:
            raise ValueError('Number of provided samples ({}) is too small for nearest neighbor estimation with given '
                             'perplexity. Please use at least {}.'.format(len(indices), 3*self.perplexity+2))
        # compute the KL divergence between low- and high-dimensional probabilities
        # self._reset_sigma()     # reset the sigma values
        # self._compute_sigma(x)      # compute the sigma values to given perplexity
        P = self.high_dim_prob(x, self.beta[indices], self.perplexity, self.use_cuda)       # upper triangle only
        P = P * self.exaggeration_fac
        Q = self.low_dim_prob(y, self.use_cuda)             # upper triangle only

        assert (torch.isfinite(P)).all(), "All high dimensional probabilities should be finite"
        assert (P >= 0).all(), "All high dimensional probabilities should be non-negative"
        assert (P <= 1).all(), ("All high dimensional probabilities should be less "
                                "or then equal to one")
        assert (torch.isfinite(Q)).all(), "All low dimensional probabilities should be finite"
        assert (Q >= 0).all(), "All low dimensional probabilities should be non-negative"
        assert (Q <= 1).all(), ("All low dimensional probabilities should be less "
                                "or then equal to one - max value: {}".format(Q.max()))

        return torch.sum(P * torch.log(
            torch.where(P > 0, P, torch.tensor(MACHINE_EPSILON).type_as(P)) / Q
            ))


class TSNEWrapper(TSNELoss):
    def __init__(self, N, perplexity=30, early_exaggeration_fac=1, use_gpu=False):
        super(TSNEWrapper, self).__init__(N, perplexity, early_exaggeration_fac, use_gpu)

    def forward(self, data, output):
        x = data[0]
        y = output
        indices = data[1]
        return super(TSNEWrapper, self).forward(x, y, indices)


class TSNEWrapperMapNet(TSNELoss):
    def __init__(self, N, perplexity=30, early_exaggeration_fac=1, use_gpu=False):
        super(TSNEWrapperMapNet, self).__init__(N, perplexity, early_exaggeration_fac, use_gpu)

    def forward(self, data, output):
        x = output[0]
        y = output[1]
        indices = data[1]
        return super(TSNEWrapperMapNet, self).forward(x, y, indices)

class CosineLoss(nn.Module):
    def __init__(self, average=False):
        super(TripletLoss, self).__init__()
        self.average = average

    def forward(self, prediction, labels):
        if prediction.dim() == 2 and labels.dim() == 1 and prediction.shape[0] == labels.shape[0]:
            raise ValueError("CosineLoss: wrong dims for prediction and labels.")
        n_samples = labels.shape[0]
        onehot = torch.zeros_like(prediction)
        onehot[torch.arange(0,n_samples), labels] = 1
        return 1 - torch.nn.functional.cosine_similarity(prediction, onehot, dim=1)

class TripletLoss(nn.Module):
    def __init__(self, triplet_selector, average=False):
        super(TripletLoss, self).__init__()
        self.triplet_selector = triplet_selector
        self.margin = self.triplet_selector.margin
        self.average = average

    def forward(self, feature, label, weights=None):
        N_batch = len(feature)

        # normalize features
        feature = feature / feature.norm(dim=1, keepdim=True)

        triplets, feature, label, weights = self.triplet_selector.get_triplets(feature, label, weights)

        if triplets is None:
            return torch.autograd.Variable(torch.tensor(0.), requires_grad=True).type_as(feature)

        ap_dist = F.pairwise_distance(feature[triplets[:, 0]], feature[triplets[:, 1]], p=2)
        an_dist = F.pairwise_distance(feature[triplets[:, 0]], feature[triplets[:, 2]], p=2)

        losses = F.relu(ap_dist - an_dist + self.margin.type_as(ap_dist))
        if weights is not None:
            weights = weights[triplets.view(-1)].view(-1, 3).prod(dim=1).pow(1./3)
            losses = losses * weights.type_as(losses)

        if self.average:
            return losses.sum() / N_batch          # not the normal mean, so that fewer triplets give better result

        return losses.sum()


class TripletSelector(object):
    """Assumes class wise negative labels are given as negaive of label itself.
    Use 0 to use negative with all classes.
    This means the minimum of valid class labels has to be 1!."""
    def __init__(self, margin, negative_selection_fn):
        super(TripletSelector, self).__init__()
        self.negative_selection_fn = negative_selection_fn
        self.margin = margin

    def _get_triplets(self, feature, label):
        labelset = torch.unique(label[label > 0])

        distance_matrix = pdist(feature).cpu()

        triplets = []
        for lbl in labelset:
            idx_positives = torch.nonzero(label == lbl)
            if len(idx_positives) < 2:
                continue
            idx_negatives = torch.nonzero((label != lbl) * (label >= 0) + (label == -lbl))

            anchor_positives = torch.LongTensor(list(combinations(idx_positives, 2)))  # All anchor-positive pairs
            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]] # distances between a und p per pair

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
    # ^select negatives inside margin
    if len(idcs) == 0:
        return None#torch.argmin(loss_values)
    choice = torch.randint(0, len(idcs), (1,), dtype=torch.long)[0]
    return idcs[choice]


class TripletLossWrapper(TripletLoss):
    def __init__(self, triplet_selector, data_labels,
                 data_weights=None,
                 N_labels_per_minibatch=2, N_samples_per_label=5, N_pure_negatives=5,
                 average=False):
        super(TripletLossWrapper, self).__init__(triplet_selector, average=average)
        self.data_labels = data_labels
        self.data_weights = torch.ones(len(data_labels)) if data_weights is None else data_weights

        self.N_labels_per_minibatch = N_labels_per_minibatch
        self.N_samples_per_label = N_samples_per_label
        self.N_pure_negatives = N_pure_negatives

    def forward(self, data, output):
        features = output[0]
        indices = data[1]
        labels = self.data_labels[indices].clone().detach()
        weights = self.data_weights[indices].clone().detach().to(features.device)

        # compute triplet loss for minibatches
        minibatches = list(BalancedBatchSampler(labels.cpu().numpy(),
                                                n_labels=self.N_labels_per_minibatch,
                                                n_samples=self.N_samples_per_label,
                                                n_pure_negatives=self.N_pure_negatives))

        distance_loss = torch.autograd.Variable(torch.tensor(0., device=features.device), requires_grad=True)
        if len(minibatches) > 0:
            for mb in minibatches:  # compute minibatch wise triplet loss
                distance_loss = distance_loss + \
                                super(TripletLossWrapper, self).forward(features[mb], labels[mb], weights=weights[mb])
            distance_loss = distance_loss / len(minibatches)
        else:
            warnings.warn('Did not find any positives in batch. No triplets formed.')

        return distance_loss


class TripletLossWrapper_pretraining(TripletLoss):
    def __init__(self, triplet_selector, data_labels, average=False):
        super(TripletLossWrapper_pretraining, self).__init__(triplet_selector, average=average)
        self.data_labels = data_labels

    def forward(self, data, output):
        features = output[0]
        indices = data[1]
        labels = torch.tensor(self.data_labels[indices])

        distance_loss = super(TripletLossWrapper_pretraining, self).forward(features, labels)
        return distance_loss


class L1RegWrapper(nn.Module):
    def __init__(self, model):
        super(L1RegWrapper, self).__init__()
        self.model = deepcopy(model)
        self.l1_crit = nn.L1Loss(size_average=False)

    def forward(self, data, output):
        reg_loss = 0
        for param in self.model.parameters():
            target = torch.zeros(param.shape).type_as(param)
            reg_loss += self.l1_crit(param, target)

        return reg_loss

    def __repr__(self):
        return 'L1RegWrapper()'


