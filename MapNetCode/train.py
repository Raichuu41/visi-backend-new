import os
import sys
import time
import numpy as np
import h5py
import pickle
import deepdish as dd
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from faiss_master import faiss
from sklearn.svm import SVC
import sklearn.metrics as metrics
from collections import Counter
import shutil
import torch.nn as nn
import copy
from losses import TripletLoss, ExemplarTripletSelector, TripletSelector, select_semihard, select_hardest, select_random

sys.path.append('../../TSNENet')
from loss import TSNELoss
from dataset import IndexDataset

import matplotlib as mpl
mpl.use('TkAgg')

# sys.path.append('../SmallNets')
# from triplets_loss import TripletLoss
# from triplets_utils import RandomNegativeTripletSelector, KHardestNegativeTripletSelector, \
#     BinaryTripletSelector, SemihardNegativeTripletSelector


sys.path.append('../../FullPipeline')
import matplotlib.pyplot as plt
from aux import AverageMeter, TBPlotter, save_checkpoint, write_config, load_weights


from model import MapNet
from communication import send_payload, make_nodes


cycle = 0
previously_modified = np.array([], dtype=np.long)
if torch.cuda.is_available():
    torch.cuda.set_device(0)


def initialize_embedder(embedder, weight_file=None, feature=None,
                        **kwargs):
    """Compute the initial weights for the embedding network.
    If weight_file is specified, simply load weights from there."""

    if weight_file is None:
        if feature is None:
            raise AttributeError('Either weight file or feature has to be provided.')
        weight_file = train_embedder(embedder, feature, **kwargs)

    if not os.path.isfile(weight_file):
        raise RuntimeError('Weight file not found.')

    # load weights of TSNENet
    # pretrained_dict = load_weights(weight_file, embedder.state_dict())
    pretrained_dict = load_weights(weight_file, embedder.state_dict(), prefix_file='embedder.')
    embedder.load_state_dict(pretrained_dict)


class NormalizedMSE(nn.Module):
    def __init__(self):
        super(NormalizedMSE, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, vec1, vec2):
        abs_dist = self.mse(vec1.data, vec2.data).mean(dim=1)
        norm = torch.max(vec1.norm(p=2, dim=1), vec2.norm(p=2, dim=1))
        dist = abs_dist / norm
        return dist.mean()


# TODO: random split of train test fine? Consider using subset for training only.
def train_embedder(embedder, feature, lr=1e-3, batch_size=100, experiment_id=None, random_state=123):
    # log and saving options
    exp_name = 'MapNet_embedder'

    if experiment_id is not None:
        exp_name = experiment_id + '_' + exp_name

    log = TBPlotter(os.path.join('runs/embedder', 'tensorboard', exp_name))
    log.print_logdir()

    outpath_model = os.path.join('runs/embedder/models')
    if not os.path.isdir(outpath_model):
        os.makedirs(outpath_model)

    # general
    use_cuda = torch.cuda.is_available()
    N = len(feature)

    idx_train, idx_test = train_test_split(range(N), test_size=0.2, random_state=random_state, shuffle=True)
    kwargs = {'num_workers': 4, 'drop_last': True} if use_cuda else {'drop_last': True}
    train_loader = DataLoader(IndexDataset(feature[idx_train]), batch_size=batch_size,              # careful, returned index is now for idx_train selection
                              **kwargs)
    test_loader = DataLoader(IndexDataset(feature[idx_test]), batch_size=batch_size,                # careful, returned index is now for idx_test selection
                             **kwargs)

    if use_cuda:
        embedder = embedder.cuda()
    stop_early_compression = 3
    stop_early_exaggeration = 1
    early_exaggeration_factor = 1

    optimizer = torch.optim.Adam(embedder.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, threshold=1e-3, verbose=True)

    train_criterion = TSNELoss(N=len(idx_train), early_exaggeration_fac=early_exaggeration_factor,
                               use_cuda=use_cuda)
    test_criterion = TSNELoss(N=len(idx_test), early_exaggeration_fac=early_exaggeration_factor,
                              use_cuda=use_cuda)
    noise_criterion = NormalizedMSE()
    print('Compute beta for KL-Loss...')
    train_criterion._compute_beta(torch.from_numpy(feature[idx_train]).cuda())
    test_criterion._compute_beta(torch.from_numpy(feature[idx_test]).cuda())
    print('done...')

    log_interval = 10

    def train(epoch):
        kl_losses = AverageMeter()
        noise_regularization = AverageMeter()
        losses = AverageMeter()
        # if epoch == stop_early_compression:
        #     print('stop early compression')

        # switch to train mode
        embedder.train()
        for batch_idx, (fts, idx) in enumerate(train_loader):
            fts = torch.autograd.Variable(fts.cuda()) if use_cuda else torch.autograd.Variable(fts)
            outputs = embedder(fts)
            noise_outputs = embedder(fts + 0.1 * torch.rand(fts.shape).type_as(fts))

            kl_loss = train_criterion(fts, outputs, idx)
            noise_reg = noise_criterion(outputs, noise_outputs)
            loss = kl_loss + 10 * noise_reg.type_as(kl_loss)

            kl_losses.update(kl_loss.data, len(idx))
            noise_regularization.update(noise_reg.data, len(idx))
            losses.update(loss.data, len(idx))

            # if epoch <= stop_early_compression:
            #     optimizer.weight_decay = 0
            #     loss += 0.01 * outputs.norm(p=2, dim=1).mean().type_as(loss)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{}]\t'
                      'Loss: {:.4f} ({:.4f})\t'
                      'LR: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(idx), len(train_loader.sampler),
                    float(losses.val), float(losses.avg),
                    optimizer.param_groups[-1]['lr']))

        log.write('kl_loss', float(kl_losses.avg), epoch, test=False)
        log.write('noise_reg', float(noise_regularization.avg), epoch, test=False)
        log.write('loss', float(losses.avg), epoch, test=False)

        return losses.avg


    def test(epoch):
        kl_losses = AverageMeter()
        noise_regularization = AverageMeter()
        losses = AverageMeter()

        # switch to evaluation mode
        embedder.eval()
        for batch_idx, (fts, idx) in enumerate(test_loader):
            fts = torch.autograd.Variable(fts.cuda()) if use_cuda else torch.autograd.Variable(fts)
            outputs = embedder(fts)
            noise_outputs = embedder(fts + 0.1 * torch.rand(fts.shape).type_as(fts))

            kl_loss = test_criterion(fts, outputs, idx)
            noise_reg = noise_criterion(outputs, noise_outputs)
            loss = kl_loss + 10 * noise_reg.type_as(kl_loss)

            kl_losses.update(kl_loss.data, len(idx))
            noise_regularization.update(noise_reg.data, len(idx))
            losses.update(loss.data, len(idx))

            if batch_idx % log_interval == 0:
                print('Test Epoch: {} [{}/{}]\t'
                      'Loss: {:.4f} ({:.4f})\n'.format(
                    epoch, (batch_idx + 1) * len(idx), len(test_loader.sampler),
                    float(losses.val), float(losses.avg)))

        log.write('kl_loss', float(kl_losses.avg), epoch, test=True)
        log.write('noise_reg', float(noise_regularization.avg), epoch, test=True)
        log.write('loss', float(losses.avg), epoch, test=True)
        return losses.avg

    # train network until scheduler reduces learning rate to threshold value
    lr_threshold = 1e-5
    epoch = 1

    best_loss = float('inf')
    best_epoch = -1
    while optimizer.param_groups[-1]['lr'] >= lr_threshold:
        train(epoch)
        testloss = test(epoch)
        scheduler.step(testloss)
        if testloss < best_loss:
            best_loss = testloss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'best_loss': best_loss,
                'state_dict': embedder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, os.path.join(outpath_model, exp_name + '.pth.tar'))
        epoch += 1

    print('Finished training embedder with best loss: {} from epoch {}'.format(best_loss, best_epoch))

    return os.path.join(outpath_model, exp_name + '.pth.tar')


def compute_embedding(embedder, feature):
    print('Compute initial embedding...')
    if torch.cuda.is_available():
        embedder = embedder.cuda()
    dataloader = DataLoader(feature, batch_size=64, shuffle=False, num_workers=4, drop_last=False)

    embedder.eval()
    embedding = []
    for batch_idx, features in enumerate(dataloader):
        if (batch_idx + 1) % 10 == 0:
            print('{}/{}'.format((batch_idx + 1) * features.shape[0], len(dataloader.dataset)))
        if torch.cuda.is_available():
            fts = torch.autograd.Variable(features.cuda())
        else:
            fts = torch.autograd.Variable(features)
        embedding.append(embedder(fts).data.cpu())
    embedding = torch.cat(embedding).numpy()
    print('Done.')
    return embedding


def get_feature(mapper, feature):
    print('Compute features...')
    if torch.cuda.is_available():
        mapper = mapper.cuda()
    dataloader = DataLoader(feature, batch_size=64, shuffle=False, num_workers=4, drop_last=False)

    mapper.eval()
    new_feature = []
    for batch_idx, fts in enumerate(dataloader):
        if (batch_idx + 1) % 10 == 0:
            print('{}/{}'.format((batch_idx + 1) * fts.shape[0], len(dataloader.dataset)))
        if torch.cuda.is_available():
            fts = torch.autograd.Variable(fts.cuda())
        else:
            fts = torch.autograd.Variable(fts)
        new_feature.append(mapper(fts).data.cpu())
    new_feature = torch.cat(new_feature).numpy()
    print('Done.')
    return new_feature


def get_modified(old_embedding, new_embedding, tol=None):
    if tol is None:
        return np.where(np.any(old_embedding != new_embedding, axis=1))[0]
    else:
        return np.where(np.any(np.abs(old_embedding - new_embedding) > tol, axis=1))[0]


# TODO: using cluster center is not ideal
def get_neighborhood(position, idx_modified):
    """Use faiss to compute neighborhood of modified samples."""
    if len(idx_modified) == 0:
        return []
    print('Infer neighborhood...')
    start = time.time()

    index = faiss.IndexFlatL2(position.shape[1])  # build the index
    index.add(position.astype(np.float32))  # add vectors to the index
    # define nearest neighbors wrt cluster center
    center = np.mean(position[idx_modified], axis=0).reshape(1, -1)
    dist, idx = index.search(center.astype(np.float32), len(position))
    in_modified = np.isin(idx, idx_modified)
    max_dist = 1.1 * np.max(dist[in_modified])
    print(max_dist)
    neighbors = idx[(dist <= max_dist) * (in_modified.__invert__())]

    stop = time.time()
    print('Done. ({}min {}s)'.format(int((stop-start))/60, (stop-start) % 60))

    return neighbors


def mutual_k_nearest_neighbors(vectors, sample_idcs, negative_idcs=None, k=1):
    print('Find high dimensional neighbors...')
    start = time.time()
    index = faiss.IndexFlatL2(vectors.shape[1])  # build the index
    index.add(vectors.astype(np.float32))  # add vectors to the index
    dist, idx = index.search(vectors[sample_idcs].astype(np.float32), len(vectors))

    # the neighbor score for each sample is computed as the sum of distances to all the samples
    scores = np.zeros(len(vectors))
    for i in range(len(vectors)):
        mask = idx == i
        scores[i] = dist[mask].sum()
    neighbors = np.argsort(scores)
    neighbors = neighbors[np.isin(neighbors, sample_idcs).__invert__()][:k]

    stop = time.time()
    print('Done. ({}min {}s)'.format(int((stop-start))/60, (stop-start) % 60))
    return neighbors, scores[neighbors]


def score_k_nearest_neighbors(vectors, sample_idcs, negative_idcs=None, k=1):
    print('Find high dimensional neighbors...')
    start = time.time()
    index = faiss.IndexFlatL2(vectors.shape[1])  # build the index
    index.add(vectors.astype(np.float32))  # add vectors to the index
    dist, idx = index.search(vectors[sample_idcs].astype(np.float32), len(vectors))

    mask = np.isin(idx, sample_idcs).__invert__()
    idx = idx[mask].reshape(len(idx), -1)

    # get the neighbor score for each sample
    # the neighbor score for each sample is computed as the sum of the column numbers in which it appears
    scores = np.zeros(len(vectors), dtype=np.longlong)
    for col in range(idx.shape[1]):
        scores[idx[:, col]] += col
    neighbors = np.argsort(scores)[len(sample_idcs)+1:len(sample_idcs)+1+k]

    stop = time.time()
    print('Done. ({}min {}s)'.format(int((stop-start))/60, (stop-start) % 60))
    return neighbors, scores[neighbors]


def listed_k_nearest_neighbors(vectors, sample_idcs, negative_idcs=None, k=1):
    print('Find high dimensional neighbors...')
    start = time.time()
    index = faiss.IndexFlatL2(vectors.shape[1])  # build the index
    index.add(vectors.astype(np.float32))  # add vectors to the index
    _, idx = index.search(vectors[sample_idcs].astype(np.float32), len(vectors))

    mask = np.isin(idx, sample_idcs).__invert__()
    idx = idx[mask].reshape(len(idx), -1)
    scores = np.arange(idx.shape[1]).reshape(1, -1) * np.ones(idx.shape)

    idx = idx.transpose().flatten()         # list nearest mutual neighbors and make them unique
    scores = scores.transpose().flatten()
    neighbors = []
    for i in idx:
        if i not in neighbors:
            neighbors.append(i)
        if len(neighbors) == k:
            break

    stop = time.time()
    print('Done. ({}min {}s)'.format(int((stop-start))/60, (stop-start) % 60))
    return neighbors, scores


def svm_k_nearest_neighbors(vectors, sample_idcs, negative_idcs=None, k=1, test=False):
    # train an SVM to separate feature space into classes
    print('Find high dimensional neighbors...')
    start = time.time()

    if negative_idcs is None:
        negative_idcs = []

    N_unlabeled = min(500, len(vectors)-len(sample_idcs)-len(negative_idcs))
    clf = SVC(kernel='rbf', C=1.0, gamma='auto', probability=True)
    print('Train SVM using {} positives, {} random negatives and {} negatives.'.format(len(sample_idcs),
                                                                                       N_unlabeled,
                                                                                       len(negative_idcs)))

    idx_unlabeled = np.setdiff1d(range(len(vectors)), np.concatenate([sample_idcs, negative_idcs]))
    idx_unlabeled = np.random.choice(idx_unlabeled, N_unlabeled, replace=False)
    if len(negative_idcs) == 0:
        train_data = np.concatenate([vectors[sample_idcs], vectors[idx_unlabeled]])
        sample_weights = np.concatenate([10 * np.ones(len(sample_idcs)), np.ones(N_unlabeled)])
    else:
        train_data = np.concatenate([vectors[sample_idcs], vectors[idx_unlabeled], vectors[negative_idcs]])
        sample_weights = np.concatenate([10*np.ones(len(sample_idcs)), np.ones(N_unlabeled), 10*np.ones(len(negative_idcs))])

    labels = np.zeros(len(train_data))
    labels[:len(sample_idcs)] = 1
    clf.fit(X=train_data, y=labels, sample_weight=sample_weights)
    prob = clf.predict_proba(vectors)[:, 1]
    prob[sample_idcs] = -1

    neighbors = np.argsort(prob)[-1::-1][:k]

    stop = time.time()

    if test:
        category = 'genre'
        thresholds = [0.75, 0.85, 0.9, 0.95]
        eval_df_train = pd.DataFrame(columns=thresholds, index=['precision', 'recall', 'f1', 'N', 'positives', 'negatives'])
        eval_df_test = pd.DataFrame(columns=thresholds, index=['precision', 'recall', 'f1', 'N', 'positives', 'negatives'])
        for threshold in thresholds:
            # evaluate on train data
            info_file_train = '/export/home/kschwarz/Documents/Masters/wikiart/datasets/info_artist_49_multilabel_test.hdf5'
            df = dd.io.load(info_file_train)['df']
            label = df[category].values
            count = Counter(label[sample_idcs])
            mainlabel = np.array(count.keys())[np.argmax(count.values())]

            probab = clf.predict_proba(vectors)[:, 1]
            y_pred = probab >= threshold
            y_true = label == mainlabel
            prec, rec, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, pos_label=1, average='binary')
            eval_df_train[threshold] = prec, rec, f1, y_pred.sum(), len(sample_idcs), len(negative_idcs)

            # evaluate on test data
            feature_file_val = 'features/MobileNetV2_info_artist_49_multilabel_val.hdf5'
            info_file_val = '/export/home/kschwarz/Documents/Masters/wikiart/datasets/info_artist_49_multilabel_val.hdf5'
            df = dd.io.load(info_file_val)['df']
            label = df[category].values
            data = dd.io.load(feature_file_val)
            try:
                names, features = data['image_id'], data['feature']
            except KeyError:
                try:
                    names, features = data['image_names'], data['features']
                except KeyError:
                    names, features = data['image_name'], data['features']
            if not (names == df['image_id']).all():
                raise RuntimeError('Image names in info file and feature file do not match.')
            probab = clf.predict_proba(features)[:, 1]
            y_pred = probab >= threshold
            y_true = label == mainlabel
            prec, rec, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, pos_label=1, average='binary')
            eval_df_test[threshold] = prec, rec, f1, y_pred.sum(), len(sample_idcs), len(negative_idcs)

        print(eval_df_train)
        print(eval_df_test)

    print('Done. ({}min {}s)'.format(int((stop-start))/60, (stop-start) % 60))
    return neighbors, prob[neighbors]


def select_neighbors(feature, sample_idcs, negative_idcs=None, k=1,
                     neighbor_fn=mutual_k_nearest_neighbors, test=False):
    neighbors, scores = neighbor_fn(feature, sample_idcs, negative_idcs=negative_idcs, k=k, test=test)

    # scale scores to range (0, 1) and invert them --> range(1, 0)
    scores = -(scores - scores.min()) * 1.0 / (scores.max() - scores.min()) + 1

    return neighbors, scores

class ChangeRateLogger(object):
    def __init__(self, n_track, threshold, order='smaller'):
        self.n_track = n_track
        self.threshold = threshold
        self.data = []
        self.order = order

    def check_threshold(self):
        tuple_data = np.stack([self.data[:-1], self.data[1:]], axis=1)
        diffs = np.array(map(lambda x: x[1] - x[0], tuple_data))

        thresh_stop = np.all(np.abs(diffs) < self.threshold)
        if thresh_stop:
            print('ChangeRateLogger: Change between last {} values was less than {}. Send stop criterion.'
                  .format(self.n_track, self.threshold))

        if self.order == 'smaller':
            order_stop = np.all(diffs > 0)
            if order_stop:
                print('ChangeRateLogger: Last {} values have all increased. Send stop criterion.'
                      .format(self.n_track, self.threshold))
        elif self.order == 'larger':
            order_stop = np.all(diffs < 0)
            if order_stop:
                print('ChangeRateLogger: Last {} values have all decreased. Send stop criterion.'
                      .format(self.n_track, self.threshold))
        else:
            order_stop = False

        stop = thresh_stop or order_stop
        return stop

    def add_value(self, value):
        self.data.append(value)
        if len(self.data) > self.n_track:
            self.data.pop(0)
            stop = self.check_threshold()
        else:
            stop = False
        return stop


class NormalizedDistanceLoss(nn.Module):
    def __init__(self):
        super(NormalizedDistanceLoss, self).__init__()

    def forward(self, input):
        upper_triangle = torch.triu(
            torch.ones((len(input), len(input)), dtype=torch.uint8),
            diagonal=1)
        distances = (-2 * input.mm(torch.t(input)) + \
                     input.pow(2).sum(dim=1).view(1, -1) + \
                     input.pow(2).sum(dim=1).view(-1, 1))[upper_triangle]
        # normalize distances
        norm = torch.max(input.norm(p=2, dim=1).data)
        distances = distances / norm
        loss = torch.mean(distances)
        return loss


class ContrastiveNormalizedDistanceLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(ContrastiveNormalizedDistanceLoss, self).__init__()
        self.margin = margin

    def forward(self, input, target):
        """target:
            0: other class - distance to class should increase
            1: same class - distance within class should decrease"""
        upper_triangle = torch.triu(
            torch.ones((len(input), len(input)), dtype=torch.uint8),
            diagonal=1)
        distances = (-2 * input.mm(torch.t(input)) + \
                     input.pow(2).sum(dim=1).view(1, -1) + \
                     input.pow(2).sum(dim=1).view(-1, 1))[upper_triangle]

        # target matrix:
        # 1: decrease distance --> set target_mat = 1
        # 2: increase distance --> set target_mat = 0
        # 4: do nothing --> ignore distances
        target[target == 0] = 2
        target_mat = torch.matmul(target.view(-1, 1), target.view(1, -1))[upper_triangle]
        distances = distances[target_mat != 4]
        target_mat = target_mat[target_mat != 4]
        target_mat[target_mat == 2] = 0

        # contraction_loss = torch.sum(target_mat.type_as(distances) * distances) / (target_mat == 1).sum().type_as(distances)
        # repulsion_loss = torch.sum((1-target_mat).type_as(distances) * torch.clamp(torch.pow(self.margin - torch.sqrt(distances), 2), min=0.0)) / (target_mat == 0).sum().type_as(distances)
        # loss = 0.5 * contraction_loss + 0.5 * repulsion_loss
        loss = torch.mean(target_mat.type_as(distances) * distances +
                          (1-target_mat).type_as(distances) * torch.pow(
                              torch.clamp(self.margin.type_as(distances) - torch.sqrt(distances), min=0.0), 2))
        return loss


class SoftNormLoss(nn.Module):
    def __init__(self, norm_value, margin):
        super(SoftNormLoss, self).__init__()
        self.value = norm_value
        self.margin = margin

    def forward(self, norm):
        diff = torch.abs(self.value.type_as(norm) - norm)
        loss = torch.clamp(diff - self.margin.type_as(norm), min=0.0)
        return loss


class ExemplarLoss(nn.Module):
    def __init__(self, margin, buffer=0.0):
        super(ExemplarLoss, self).__init__()
        self.margin = margin
        self.buffer = buffer

    def forward(self, positives, negatives):
        center = torch.mean(positives, dim=0, keepdim=True)
        idx_center = torch.argmin(torch.norm(positives - center, p=2, dim=1))
        exemplar = positives[idx_center]

        attractive_loss = torch.clamp(
            torch.norm(positives - exemplar, p=2, dim=1) - (1-self.buffer)*self.margin.type_as(positives), min=0)
        repulsive_loss = torch.clamp((1+self.buffer)*self.margin.type_as(positives) - torch.norm(negatives - exemplar, p=2, dim=1), min=0)

        loss = 0.5 * torch.mean(attractive_loss) + 0.5 * torch.mean(repulsive_loss)
        return loss


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class ContrastiveLoss(nn.Module):
    def __init__(self, margin, buffer=0.0):
        super(ContrastiveLoss, self).__init__()
        if not isinstance(margin, torch.Tensor):
            margin = torch.as_tensor(margin)
        self.margin = torch.nn.Parameter(margin)
        self.buffer = buffer

    def forward(self, positives, negatives):
        # pairwise distances
        dist_sq = pdist(torch.cat([positives, negatives]))
        idcs_upper = torch.triu(torch.ones(dist_sq.shape, dtype=torch.uint8),
                                diagonal=1)  # use symmetry and discard diagonal entries
        idcs_positives = idcs_upper.clone()
        idcs_positives[:, len(positives):] = 0

        idcs_negatives = idcs_upper.clone()
        idcs_negatives[:, :len(positives)] = 0
        idcs_negatives[len(positives):, len(positives):] = 0

        positives_dist = torch.sqrt(dist_sq[idcs_positives])
        negatives_dist = torch.sqrt(dist_sq[idcs_negatives])

        attractive_loss = torch.pow(torch.clamp(positives_dist - (1-self.buffer)*self.margin.type_as(dist_sq), min=0), 2)
        repulsive_loss = torch.pow(torch.clamp((1+self.buffer)*self.margin.type_as(dist_sq) - negatives_dist, min=0), 2)

        loss = 0.5 * torch.mean(attractive_loss) + 0.5 * torch.mean(repulsive_loss)
        return loss


def train(net, feature, image_id, old_embedding, target_embedding,
          idx_modified, idx_old_neighbors, idx_new_neighbors,
          idx_negatives,
          lr=1e-3, experiment_id=None, socket_id=None, scale_func=None,
          categories=None, label=None):
    global cycle, previously_modified
    print(idx_modified)
    cycle += 1
    # log and saving options
    exp_name = 'MapNet'

    if experiment_id is not None:
        exp_name = experiment_id + '_' + exp_name

    log = TBPlotter(os.path.join('runs/mapping', 'tensorboard', exp_name))
    log.print_logdir()

    outpath_config = os.path.join('runs/mapping', exp_name, 'configs')
    if not os.path.isdir(outpath_config):
        os.makedirs(outpath_config)
    outpath_embedding = os.path.join('runs/mapping', exp_name, 'embeddings')
    if not os.path.isdir(outpath_embedding):
        os.makedirs(outpath_embedding)
    outpath_feature = os.path.join('runs/mapping', exp_name, 'features')
    if not os.path.isdir(outpath_feature):
        os.makedirs(outpath_feature)
    outpath_model = os.path.join('runs/mapping', exp_name, 'models')
    if not os.path.isdir(outpath_model):
        os.makedirs(outpath_model)

    # general
    N = len(feature)
    use_cuda = torch.cuda.is_available()
    if not isinstance(old_embedding, torch.Tensor):
        old_embedding = torch.from_numpy(old_embedding.copy())
    if not isinstance(target_embedding, torch.Tensor):
        target_embedding = torch.from_numpy(target_embedding.copy())

    if use_cuda:
        net = net.cuda()
    net.train()

    # Set up differend groups of indices
    # each sample belongs to one group exactly, hierarchy is as follows:
    # 1: samples moved by user in this cycle
    # 2: negatives selected through neighbor method
    # 3: new neighborhood
    # 4: samples moved by user in previous cycles
    # 5: old neighborhood
    # 5: high dimensional neighborhood of moved samples
    # 6: fix points / unrelated (remaining) samples

    # # find high dimensional neighbors
    idx_high_dim_neighbors, _ = svm_k_nearest_neighbors(feature, np.union1d(idx_modified, idx_new_neighbors),
                                                        negative_idcs=idx_negatives,
                                                        k=100)  # use the first 100 nn of modified samples          # TODO: Better rely on distance

    # ensure there is no overlap between different index groups
    idx_modified = np.setdiff1d(idx_modified, idx_negatives)            # just ensure in case negatives have moved accidentially    TODO: BETTER FILTER BEFORE
    idx_new_neighbors = np.setdiff1d(idx_new_neighbors, np.concatenate([idx_modified, idx_negatives]))
    idx_previously_modified = np.setdiff1d(previously_modified, np.concatenate([idx_modified, idx_new_neighbors, idx_negatives]))
    idx_old_neighbors = np.setdiff1d(np.concatenate([idx_old_neighbors, idx_high_dim_neighbors]),
                                     np.concatenate([idx_modified, idx_new_neighbors, idx_previously_modified, idx_negatives]))
    idx_fix_points = np.setdiff1d(range(N),
                                  np.concatenate([idx_modified, idx_new_neighbors,
                                                  idx_previously_modified, idx_old_neighbors, idx_negatives]))

    for i, g1 in enumerate([idx_modified, idx_new_neighbors, idx_previously_modified,
                            idx_old_neighbors, idx_fix_points, idx_negatives]):
        for j, g2 in enumerate([idx_modified, idx_new_neighbors, idx_previously_modified,
                                idx_old_neighbors, idx_fix_points, idx_negatives]):
            if i != j and len(np.intersect1d(g1, g2)) != 0:
                print('groups: {}, {}'.format(i, j))
                print(np.intersect1d(g1, g2))
                raise RuntimeError('Index groups overlap.')

    print('Group Overview:'
          '\n\tModified samples: {}'
          '\n\tNegative samples: {}'    
          '\n\tNew neighbors: {}'
          '\n\tPreviously modified samples: {}'
          '\n\tOld neighbors: {}'
          '\n\tFix points: {}'.format(
        len(idx_modified), len(idx_negatives), len(idx_new_neighbors), len(idx_previously_modified),
        len(idx_old_neighbors), len(idx_fix_points)))

    # modify label
    label[idx_modified, -1] = 'modified'
    label[idx_negatives, -1] = 'negative'
    label[idx_previously_modified, -1] = 'prev_modified'
    label[idx_new_neighbors, -1] = 'new neighbors'
    label[idx_old_neighbors, -1] = 'old neighbors'
    label[idx_high_dim_neighbors, -1] = 'high dim neighbors'
    label[idx_fix_points, -1] = 'other'

    kl_criterion = TSNELoss(N, use_cuda=use_cuda)
    l2_criterion = torch.nn.MSELoss(reduction='none')                  # keep the output fixed
    hl2_criterion = torch.nn.MSELoss(reduction='none')                  # keep the output fixed
    noise_criterion = NormalizedMSE()

    # define the index samplers for data

    batch_size = 500
    max_len = max(len(idx_modified) + len(idx_previously_modified), len(idx_negatives), len(idx_new_neighbors),
                  len(idx_old_neighbors), len(idx_fix_points))
    if max_len == len(idx_fix_points):
        n_batches = max_len / (batch_size * 2) + 1
    else:
        n_batches = max_len / batch_size + 1

    sampler_modified = torch.utils.data.BatchSampler(
        sampler=torch.utils.data.SubsetRandomSampler(idx_modified),
        batch_size=batch_size, drop_last=False)

    sampler_negatives = torch.utils.data.BatchSampler(
        sampler=torch.utils.data.SubsetRandomSampler(idx_negatives),
        batch_size=batch_size, drop_last=False)

    sampler_new_neighbors = torch.utils.data.BatchSampler(
        sampler=torch.utils.data.SubsetRandomSampler(idx_new_neighbors),
        batch_size=batch_size, drop_last=False)

    sampler_prev_modified = torch.utils.data.BatchSampler(
        sampler=torch.utils.data.SubsetRandomSampler(idx_previously_modified),
        batch_size=batch_size, drop_last=False)

    sampler_old_neighbors = torch.utils.data.BatchSampler(
        sampler=torch.utils.data.SubsetRandomSampler(idx_old_neighbors),
        batch_size=batch_size, drop_last=False)

    sampler_high_dim_neighbors = torch.utils.data.BatchSampler(
        sampler=torch.utils.data.SubsetRandomSampler(idx_high_dim_neighbors),
        batch_size=batch_size, drop_last=False)

    sampler_fixed = torch.utils.data.BatchSampler(
        sampler=torch.utils.data.SubsetRandomSampler(idx_fix_points),
        batch_size=2 * batch_size, drop_last=False)

    # train network until scheduler reduces learning rate to threshold value
    lr_threshold = 1e-5
    track_l2_loss = ChangeRateLogger(n_track=5, threshold=5e-2, order='smaller')
    track_noise_reg = ChangeRateLogger(n_track=10, threshold=-1, order='smaller')          # only consider order --> negative threshold
    stop_criterion = False


    epoch = 1
    new_features = feature.copy()
    new_embedding = old_embedding.numpy().copy()

    t_beta = []
    t_train = []
    t_tensorboard = []
    t_save = []
    t_send = []
    t_iter = []

    tensor_feature = torch.from_numpy(feature)
    norms = torch.norm(tensor_feature, p=2, dim=1)
    feature_norm = torch.mean(norms)
    feature_std = torch.std(tensor_feature[np.concatenate([idx_new_neighbors, idx_modified])], dim=0)
    # distance_criterion = ExemplarLoss(margin=feature_norm, buffer=0.5)
    # distance_criterion = ContrastiveLoss(margin=feature_norm, buffer=0.1)
    triplet_selector = ExemplarTripletSelector(margin=feature_norm, negative_selection_fn=select_semihard,
                                               negative_labels=(torch.tensor(0),),
                                               avg_samples_per_cluster=20, gpu=True, parallel=False, niter=10)
    distance_criterion = TripletLoss(triplet_selector=triplet_selector, margin=triplet_selector.margin)
    dist_negatives = []
    # distance_criterion = ContrastiveNormalizedDistanceLoss(margin=0.2 * feature_norm)
    # triplet_margin = feature_norm
    # triplet_selector = SemihardNegativeTripletSelector(margin=triplet_margin, cpu=False, preselect_index_positives=10, preselect_index_negatives=1, selection='random')
    # distance_criterion = TripletLoss(margin=triplet_margin, triplet_selector=triplet_selector)
    # positive_triplet_collector = np.array([], dtype=int)
    # negative_triplet_collector = np.array([], dtype=int)

    del norms

    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad] + [p for p in distance_criterion.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=1e-3,
                                                           verbose=True)

    embeddings = {}
    model_states = {}
    cpu_net = copy.deepcopy(net).cpu() if use_cuda else net
    model_states[0] = {
        'epoch': 0,
        'loss': float('inf'),
        'state_dict': cpu_net.state_dict().copy(),
        'optimizer': optimizer.state_dict().copy(),
        'scheduler': scheduler.state_dict().copy()
    }
    embeddings[0] = old_embedding.numpy().copy()

    while not stop_criterion:
        # if epoch < 30:           # do not use dropout at first
        #     net.eval()
        # else:
        net.train()

        t_iter_start = time.time()

        # compute beta for kl loss
        t_beta_start = time.time()
        kl_criterion._compute_beta(new_features)
        t_beta_end = time.time()
        t_beta.append(t_beta_end - t_beta_start)

        # set up losses
        l2_losses = AverageMeter()
        hl2_losses = AverageMeter()
        kl_losses = AverageMeter()
        distance_losses = AverageMeter()
        noise_regularization = AverageMeter()
        feature_norm = AverageMeter()
        weight_regularization = AverageMeter()
        losses = AverageMeter()

        t_load = []
        t_forward = []
        t_loss = []
        t_backprop = []
        t_update = []
        t_tot = []

        # iterate over fix points (assume N_fixpoints >> N_modified)
        t_train_start = time.time()
        t_load_start = time.time()
        batch_loaders = []
        for smplr in [sampler_modified, sampler_negatives, sampler_new_neighbors, sampler_prev_modified,
                      sampler_old_neighbors, sampler_fixed, sampler_high_dim_neighbors]:
            batches = list(smplr)
            if len(batches) == 0:
                batches = [[] for i in range(n_batches)]
            while len(batches) < n_batches:
                to = min(n_batches - len(batches), len(batches))
                batches.extend(list(smplr)[:to])
            batch_loaders.append(batches)

        for batch_idx in range(n_batches):
            t_tot_start = time.time()

            moved_indices = batch_loaders[0][batch_idx]
            negatives_indices = batch_loaders[1][batch_idx]
            new_neigh_indices = batch_loaders[2][batch_idx]
            prev_moved_indices = batch_loaders[3][batch_idx]
            old_neigh_indices = batch_loaders[4][batch_idx]
            fixed_indices = batch_loaders[5][batch_idx]
            high_neigh_indices = batch_loaders[6][batch_idx]
            n_moved, n_neg, n_new, n_prev, n_old, n_fixed, n_high = (len(moved_indices), len(negatives_indices),
                                                             len(new_neigh_indices), len(prev_moved_indices),
                                                             len(old_neigh_indices),
                                                             len(fixed_indices), len(high_neigh_indices))

            # load data
            indices = np.concatenate([new_neigh_indices, moved_indices, negatives_indices, prev_moved_indices,
                                      fixed_indices, old_neigh_indices, high_neigh_indices]).astype(long)
            if len(indices) < 3 * kl_criterion.perplexity + 2:
                continue
            data = tensor_feature[indices]
            input = torch.autograd.Variable(data.cuda()) if use_cuda else torch.autograd.Variable(data)

            t_load_end = time.time()
            t_load.append(t_load_end - t_load_start)

            # compute forward
            t_forward_start = time.time()

            fts_mod = net.mapping(input)
            noise = torch.randn_like(input) * 0.1 * feature_std.type_as(input)         # scale by dimensionwise std of features
            fts_mod_noise = net.mapping(input + noise)
            emb_mod = net.embedder(torch.nn.functional.relu(fts_mod))

            t_forward_end = time.time()
            t_forward.append(t_forward_end - t_forward_start)

            # compute losses
            # modified --> KL, L2, Dist
            # new neighborhood --> KL, Dist
            # previously modified --> KL, L2
            # old neighborhood + high dimensional neighborhood --> KL
            # fix point samples --> KL, L2

            t_loss_start = time.time()

            # LOW DIMENSIONAL L2 LOSS
            batch_idx_l2_fixed = np.concatenate([np.arange(0, n_new), np.arange(n_new+n_moved, len(indices)-n_high)])
            idx_l2_fixed = indices[batch_idx_l2_fixed]        # on all but moved and high dimensional nn
            # idx_l2_fixed = np.concatenate([new_neigh_indices, negatives_indices, prev_moved_indices,
            #                                fixed_indices, old_neigh_indices, high_neigh_indices]).astype(long)
            # l2_loss_input = torch.cat([emb_mod[:n_new], emb_mod[n_new + n_moved:]])
            l2_loss = torch.mean(
                l2_criterion(emb_mod[batch_idx_l2_fixed], target_embedding[idx_l2_fixed].type_as(emb_mod)))
            l2_losses.update(l2_loss.data, len(idx_l2_fixed))

            kl_loss = kl_criterion(fts_mod, emb_mod, indices)
            kl_losses.update(kl_loss.data, len(data))

            # # TRIPLET LOSS FOR MOVED SAMPLES
            # distance_loss_input = fts_mod[n_new: n_new + n_moved + n_neg]
            # distance_loss_target = torch.cat([torch.ones(n_moved), torch.zeros(n_neg)])
            # selected_negatives = None if n_neg == 0 else {1: np.arange(n_moved, n_moved + n_neg)}
            # distance_loss, positive_triplets, negative_triplets = distance_criterion(distance_loss_input, distance_loss_target, concealed_classes=[0], weights=None, selected_negatives=selected_negatives)
            # if positive_triplets is not None:
            #     positive_triplets = np.unique(positive_triplets.numpy().flatten()).astype(int)
            #     negative_triplets = np.unique(negative_triplets.numpy()).astype(int)
            #     positive_triplets = indices[n_new: n_new + n_moved + n_neg][positive_triplets]
            #     negative_triplets = indices[n_new: n_new + n_moved + n_neg][negative_triplets]
            #     positive_triplet_collector = np.union1d(positive_triplet_collector, positive_triplets)
            #     negative_triplet_collector = np.union1d(negative_triplet_collector, negative_triplets)
            distance_loss_positives = fts_mod[n_new: n_new + n_moved]
            N_rand_negatives = max(2*n_moved, min(500, n_prev+n_fixed+n_old))
            idcs_distance_negatives = np.concatenate([np.arange(n_new + n_moved, n_new + n_moved + n_neg),
                                                      np.random.choice(
                                                          np.arange(n_new + n_moved + n_neg,
                                                                    n_new + n_moved + n_neg + n_prev + n_fixed + n_old),
                                                          N_rand_negatives, replace=False)
                                                      ])
            dist_negatives = np.union1d(dist_negatives, indices[idcs_distance_negatives])
            distance_loss_negatives = fts_mod[idcs_distance_negatives]
            distance_loss_label = torch.cat([torch.ones(n_moved, dtype=torch.long),
                                             torch.zeros(N_rand_negatives + n_neg, dtype=torch.long)])
            distance_loss_weights = torch.cat([torch.ones(n_moved),
                                               torch.ones(n_neg),
                                               0.5 * torch.ones(N_rand_negatives)])
            distance_loss = distance_criterion(torch.cat([distance_loss_positives, distance_loss_negatives]), distance_loss_label, distance_loss_weights)
            # distance_loss = distance_criterion(positives=distance_loss_positives, negatives=distance_loss_negatives)          # for contrastive loss
            distance_losses.update(distance_loss.data, n_moved+n_neg)

            # # HIGH DIMENSIONAL L2 LOSS ON ALL BUT MOVED
            # HIGH DIMENSIONAL L2 LOSS ON NEGATIVES and previously moved ONLY
            # idx_hl2_fixed = np.concatenate([negatives_indices, prev_moved_indices,
            #                                 # fixed_indices, old_neigh_indices,
            #                                 # high_neigh_indices             # LEAVE OUT HIGH DIMENSIONAL NEIGHBORS and see what happens...
            #                                 ]).astype(long)
            idx_hl2_fixed = indices[idcs_distance_negatives]
            hl2_loss_input = distance_loss_negatives
            hl2_loss_target = tensor_feature[idx_hl2_fixed]
            hl2_loss = torch.mean(hl2_criterion(hl2_loss_input, hl2_loss_target.type_as(hl2_loss_input)))

            hl2_losses.update(hl2_loss.data, len(idx_hl2_fixed))


            # REGULARIZATION
            weight_reg = torch.autograd.Variable(torch.tensor(0.)).type_as(l2_loss)
            for param in net.mapping.parameters():
                weight_reg += param.norm(1)
            weight_regularization.update(weight_reg, len(data))

            noise_reg = noise_criterion(fts_mod, fts_mod_noise)
            noise_regularization.update(noise_reg.data, len(data))

            loss = distance_loss.type_as(l2_loss) + 10 * hl2_loss + 0 * l2_loss + 1 * kl_loss.type_as(l2_loss)# + \
                   #10 * noise_reg.type_as(l2_loss)# + 1e-5 * weight_reg.type_as(l2_loss)
            # if epoch >= 10:
            #     loss = loss + 1e-5 * weight_reg.type_as(l2_loss)
            losses.update(loss.data, len(data))

            t_loss_end = time.time()
            t_loss.append(t_loss_end - t_loss_start)

            feature_norm.update(torch.mean(fts_mod.norm(p=2, dim=1)).data, len(data))

            # backprop

            t_backprop_start = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_backprop_end = time.time()
            t_backprop.append(t_backprop_end - t_backprop_start)

            # update

            t_update_start = time.time()

            # update current embedding
            new_embedding[indices] = emb_mod.data.cpu().numpy()

            t_update_end = time.time()
            t_update.append(t_update_end - t_update_start)

            if epoch > 5 and (batch_idx+1) * batch_size >= 2000:
                print('\tend epoch after {} random fix point samples'.format((batch_idx+1) * batch_size))
                break

            t_tot_end = time.time()
            t_tot.append(t_tot_end - t_tot_start)

            t_load_start = time.time()

        # print('Times:'
        #       '\n\tLoader: {})'
        #       '\n\tForward: {})'
        #       '\n\tLoss: {})'
        #       '\n\tBackprop: {})'
        #       '\n\tUpdate: {})'
        #       '\n\tTotal: {})'.format(
        #     np.mean(t_load),
        #     np.mean(t_forward),
        #     np.mean(t_loss),
        #     np.mean(t_backprop),
        #     np.mean(t_update),
        #     np.mean(t_tot),
        # ))

        t_train_end = time.time()
        t_train.append(t_train_end - t_train_start)

        t_tensorboard_start = time.time()
        scheduler.step(losses.avg)
        # label[positive_triplet_collector, -1] = 'positive triplet'
        # label[negative_triplet_collector, -1] = 'negative triplet'
        log.write('l2_loss', float(l2_losses.avg), epoch, test=False)
        log.write('hl2_loss', float(hl2_losses.avg), epoch, test=False)
        log.write('distance_loss', float(distance_losses.avg), epoch, test=False)
        log.write('kl_loss', float(kl_losses.avg), epoch, test=False)
        log.write('noise_regularization', float(noise_regularization.avg), epoch, test=False)
        log.write('feature_norm', float(feature_norm.avg), epoch, test=False)
        # log.write('norm_loss', float(norm_losses.avg), epoch, test=False)
        log.write('weight_reg', float(weight_regularization.avg), epoch, test=False)
        log.write('loss', float(losses.avg), epoch, test=False)
        t_tensorboard_end = time.time()
        t_tensorboard.append(t_tensorboard_end - t_tensorboard_start)

        t_save_start = time.time()

        cpu_net = copy.deepcopy(net).cpu() if use_cuda else net

        model_states[epoch] = {
            'epoch': epoch,
            'loss': losses.avg.cpu(),
            'state_dict': cpu_net.state_dict().copy(),
            'optimizer': optimizer.state_dict().copy(),
            'scheduler': scheduler.state_dict().copy()
        }
        embeddings[epoch] = new_embedding

        t_save_end = time.time()
        t_save.append(t_save_end - t_save_start)

        print('Train Epoch: {}\t'
              'Loss: {:.4f}\t'
              'L2 Loss: {:.4f}\t'
              'HL2 Loss: {:.4f}\t'
              'Distance Loss: {:.4f}\t'
              'KL Loss: {:.4f}\t'
              'Weight Reg: {:.4f}\t'
              'Noise Reg: {:.4f}\t'
              'Margin: {:.4f}\t'
              'LR: {:.6f}'.format(
            epoch,
            float(losses.avg),
            float(0.1 * l2_losses.avg),
            float(10 * hl2_losses.avg),
            float(1 * distance_losses.avg),
            float(10 * kl_losses.avg),
            float(1e-5 * weight_regularization.avg),
            float(10 * noise_regularization.avg),
            distance_criterion.margin,
            optimizer.param_groups[-1]['lr']))

        t_send_start = time.time()

        # send to server
        if socket_id is not None:
            position = new_embedding if scale_func is None else scale_func(new_embedding)
            nodes = make_nodes(position=position, index=True, label=label)
            send_payload(nodes, socket_id, categories=categories)

        t_send_end = time.time()
        t_send.append(t_send_end - t_send_start)

        epoch += 1
        l2_stop_criterion = False#track_l2_loss.add_value(l2_losses.avg)
        epoch_stop_criterion = epoch > 150
        regularization_stop_criterion = False#track_noise_reg.add_value(noise_regularization.avg)
        lr_stop_criterion = optimizer.param_groups[-1]['lr'] < lr_threshold
        stop_criterion = any([l2_stop_criterion, regularization_stop_criterion, lr_stop_criterion, epoch_stop_criterion])

        t_iter_end = time.time()
        t_iter.append(t_iter_end - t_iter_start)

    print('Times:'
          '\n\tBeta: {})'
          '\n\tTrain: {})'
          '\n\tTensorboard: {})'
          '\n\tSave: {})'
          '\n\tSend: {})'
          '\n\tIteration: {})'.format(
        np.mean(t_beta),
        np.mean(t_train),
        np.mean(t_tensorboard),
        np.mean(t_save),
        np.mean(t_send),
        np.mean(t_iter),
    ))

    print('Training details: '
          '\n\tMean: {}'
          '\n\tMax: {} ({})'
          '\n\tMin: {} ({})'.format(
        np.mean(t_train),
        np.max(t_train), np.argmax(t_train),
        np.min(t_train), np.argmin(t_train)
    ))

    previously_modified = np.append(previously_modified, idx_modified)

    # compute new features
    new_features = get_feature(net.mapping, feature)

    # print('Save output files...')
    # write output files for the cycle
    outfile_config = os.path.join(outpath_config, 'cycle_{:03d}_config.pkl'.format(cycle))
    outfile_embedding = os.path.join(outpath_embedding, 'cycle_{:03d}_embeddings.hdf5'.format(cycle))
    outfile_feature = os.path.join(outpath_feature, 'cycle_{:03d}_feature.hdf5'.format(cycle))
    outfile_model_states = os.path.join(outpath_model, 'cycle_{:03d}_models.pth.tar'.format(cycle))

    with h5py.File(outfile_embedding, 'w') as f:
        f.create_dataset(name='image_id', shape=image_id.shape, dtype=image_id.dtype, data=image_id)
        for epoch in embeddings.keys():
            data = embeddings[epoch]
            f.create_dataset(name='epoch_{:04d}'.format(epoch), shape=data.shape, dtype=data.dtype, data=data)
    print('\tSaved {}'.format(os.path.join(os.getcwd(), outfile_embedding)))

    with h5py.File(outfile_feature, 'w') as f:
        f.create_dataset(name='feature', shape=new_features.shape, dtype=new_features.dtype, data=new_features)
        f.create_dataset(name='image_id', shape=image_id.shape, dtype=image_id.dtype, data=image_id)
    print('\tSaved {}'.format(os.path.join(os.getcwd(), outfile_feature)))

    model_states_to_save = {k: v for k, v in model_states.items()[::3]}                # only every 3rd
    torch.save(model_states_to_save, outfile_model_states)
    print('\tSaved {}'.format(os.path.join(os.getcwd(), outfile_model_states)))

    # write config file
    config_dict = {'idx_modified': idx_modified, 'idx_old_neighbors': idx_old_neighbors,
                   'idx_new_neighbors': idx_new_neighbors, 'idx_high_dim_neighbors': idx_high_dim_neighbors,
                   'idx_negatives': idx_negatives, 'dist_negatives': dist_negatives}
    with open(outfile_config, 'w') as f:
        pickle.dump(config_dict, f)
    print('\tSaved {}'.format(os.path.join(os.getcwd(), outfile_config)))

    print('Done.')

    print('Finished training.')
    return new_embedding


def reset(experiment_id=None, dataset='wikiart'):
    global cycle, previously_modified
    cycle = 0
    previously_modified = np.array([], dtype=np.long)

    exp_name = 'MapNet'

    if experiment_id is not None:
        exp_name = experiment_id + '_' + exp_name
    if dataset == 'shape':
        exp_name = 'ShapeDataset_' + exp_name
    elif dataset == 'office':
        exp_name = 'OfficeDataset_' + exp_name
    elif dataset == 'bam':
        exp_name = 'BAMDataset_' + exp_name
    if os.path.isdir(os.path.join('runs/mapping', exp_name)):
        shutil.rmtree(os.path.join('runs/mapping', exp_name), ignore_errors=True)
        print('Reset log directory.')
    if os.path.isdir(os.path.join('runs/mapping/tensorboard', exp_name)):
        shutil.rmtree(os.path.join('runs/mapping/tensorboard', exp_name), ignore_errors=True)
        print('Reset tensorboard log directory.')
    if os.path.isdir(os.path.join('runs/embedder/tensorboard', exp_name + '_embedder')):
        shutil.rmtree(os.path.join('runs/embedder/tensorboard', exp_name + '_embedder'), ignore_errors=True)
        print('Reset tensorboard log directory for embedder.')

if __name__ == '__main__':
    pass
