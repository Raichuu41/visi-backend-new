import os
import sys
import time
import numpy as np
import h5py
import pickle
import deepdish as dd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from faiss_master import faiss
from Queue import Queue

sys.path.append('../TSNENet')
from loss import TSNELoss
from dataset import IndexDataset

sys.path.append('../FullPipeline')
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from aux import AverageMeter, TBPlotter, save_checkpoint, write_config, load_weights


if not os.getcwd().endswith('/MapNetCode'):
    os.chdir('/net/hciserver03/storage/kschwarz/Documents/Masters/WebInterface/MapNetCode')
sys.path.append('.')
from model import MapNet
from communication import send_payload, make_nodes


cycle = 0
previously_modified = np.array([], dtype=np.long)


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
    print('Compute beta for KL-Loss...')
    train_criterion._compute_beta(torch.from_numpy(feature[idx_train]).cuda())
    test_criterion._compute_beta(torch.from_numpy(feature[idx_test]).cuda())
    print('done...')

    log_interval = 10

    def train(epoch):
        losses = AverageMeter()
        # if epoch == stop_early_compression:
        #     print('stop early compression')

        # switch to train mode
        embedder.train()
        for batch_idx, (fts, idx) in enumerate(train_loader):
            fts = torch.autograd.Variable(fts.cuda()) if use_cuda else torch.autograd.Variable(fts)
            outputs = embedder(fts)
            loss = train_criterion(fts, outputs, idx)

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

        log.write('loss', float(losses.avg), epoch, test=False)
        return losses.avg

    def test(epoch):
        losses = AverageMeter()

        # switch to evaluation mode
        embedder.eval()
        for batch_idx, (fts, idx) in enumerate(test_loader):
            fts = torch.autograd.Variable(fts.cuda()) if use_cuda else torch.autograd.Variable(fts)
            outputs = embedder(fts)
            loss = test_criterion(fts, outputs, idx)
            losses.update(loss.data, len(idx))

            if batch_idx % log_interval == 0:
                print('Test Epoch: {} [{}/{}]\t'
                      'Loss: {:.4f} ({:.4f})\n'.format(
                    epoch, (batch_idx + 1) * len(idx), len(test_loader.sampler),
                    float(losses.val), float(losses.avg)))

        log.write('loss', float(losses.avg), epoch, test=True)
        return losses.avg

    # train network until scheduler reduces learning rate to threshold value
    lr_threshold = 1e-5
    epoch = 1

    best_loss = float('inf')
    while optimizer.param_groups[-1]['lr'] >= lr_threshold:
        train(epoch)
        testloss = test(epoch)
        scheduler.step(testloss)
        if testloss < best_loss:
            best_loss = testloss
            torch.save({
                'epoch': epoch,
                'best_loss': best_loss,
                'state_dict': embedder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, os.path.join(outpath_model, exp_name + '.pth.tar'))
        epoch += 1

    print('Finished training embedder with best loss: {}'.format(best_loss))

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
    neighbors = idx[(dist <= max_dist) * (in_modified.__invert__())]

    stop = time.time()
    print('Done. ({}min {}s)'.format(int((stop-start))/60, (stop-start) % 60))

    return neighbors


def mutual_k_nearest_neighbors(vectors, sample_idcs, k=1):
    print('Find high dimensional neighbors...')
    start = time.time()
    index = faiss.IndexFlatL2(vectors.shape[1])  # build the index
    index.add(vectors.astype(np.float32))  # add vectors to the index
    _, idx = index.search(vectors[sample_idcs].astype(np.float32), len(vectors))

    mask = np.isin(idx, sample_idcs).__invert__()
    idx = idx[mask].reshape(len(idx), -1)

    # get the neighbor score for each sample
    # the neighbor score for each sample is computed as the sum of the column numbers in which it appears
    scores = np.zeros(len(vectors), dtype=np.longlong)
    for col in range(idx.shape[1]):
        scores[idx[:, col]] += col
    neighbors = np.argsort(scores)[len(sample_idcs)+1:len(sample_idcs)+1+k]

    # idx = idx.transpose().flatten()         # list nearest mutual neighbors and make them unique
    # neighbors = []
    # for i in idx:
    #     if i not in neighbors:
    #         neighbors.append(i)
    #     if len(neighbors) == k:
    #         break

    stop = time.time()
    print('Done. ({}min {}s)'.format(int((stop-start))/60, (stop-start) % 60))
    return neighbors


class ChangeRateLogger(object):
    def __init__(self, n_track, threshold):
        self.n_track = n_track
        self.threshold = threshold
        self.data = []

    def check_threshold(self):
        tuple_data = np.stack([self.data[:-1], self.data[1:]], axis=1)
        diffs = np.array(map(lambda x: np.abs(x[1] - x[0]), tuple_data))
        stop = np.all(diffs < self.threshold)
        return stop

    def add_value(self, value):
        self.data.append(value)
        if len(self.data) > self.n_track:
            self.data.pop(0)
            stop = self.check_threshold()
        else:
            stop = False
        if stop:
            print('ChangeRateLogger: Change between last {} values was less than {}. Send stop criterion.'
                  .format(self.n_track, self.threshold))
        return stop


def train(net, feature, image_id, label, old_embedding, target_embedding,
          idx_modified, idx_old_neighbors, idx_new_neighbors,
          lr=1e-3, experiment_id=None, socket_id=None, scale_func=None):
    global cycle, previously_modified
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

    # find high dimensional neighbors
    idx_high_dim_neighbors = mutual_k_nearest_neighbors(feature, idx_modified,
                                                        k=100)  # use the first 100 nn of modified samples

    # ensure there is no overlap between different index groups
    previously_modified = np.setdiff1d(previously_modified, idx_modified)           # if sample was modified again, allow change
    neighbors = np.unique(np.concatenate([idx_old_neighbors, idx_new_neighbors, idx_high_dim_neighbors]))
    neighbors = np.setdiff1d(neighbors, previously_modified)
    space_samples = np.setdiff1d(range(N), np.concatenate([idx_modified, neighbors, previously_modified]))

    for i, g1 in enumerate([idx_modified, previously_modified, neighbors, space_samples]):
        for j, g2 in enumerate([idx_modified, previously_modified, neighbors, space_samples]):
            if i != j and len(np.intersect1d(g1, g2)) != 0:
                print('groups: {}, {}'.format(i, j))
                print(np.intersect1d(g1, g2))
                raise RuntimeError('Index groups overlap.')

    print('Group Overview:'
          '\n\tModified samples: {}'
          '\n\tPreviously modified samples: {}'
          '\n\tNeighbors samples: {}'
          '\n\tSpace samples: {}'.format(
        len(idx_modified), len(previously_modified), len(neighbors), len(space_samples)
    ))
    # modify label
    label[idx_modified, -1] = 'modified'
    label[previously_modified, -1] = 'prev_modified'
    label[neighbors, -1] = 'neighbors'
    label[idx_high_dim_neighbors, -1] = 'high_dim_neighbors'
    label[space_samples, -1] = 'other'


    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=1e-3,
                                                           verbose=True)

    kl_criterion = TSNELoss(N, use_cuda=use_cuda)
    l2_criterion = torch.nn.MSELoss(reduction='none')                  # keep the output fixed

    # define the index samplers for data
    #   sample_sampler: modified samples and previously modified samples
    #   neighbor_sampler: high-dimensional and previous and new neighbors (these will be shown more often in training process)
    #   space_sampler: remaining samples (neither modified nor neighbors, only show them once)

    batch_size = 1000

    max_len = max(len(idx_modified) + len(previously_modified), len(neighbors), len(space_samples))

    n = len(idx_modified) + len(previously_modified)
    print('{} modified'.format(n))
    n_repeat = 1 if (n == max_len or n == 0) else max_len / n + 1
    sample_sampler = torch.utils.data.BatchSampler(
        sampler=torch.utils.data.SubsetRandomSampler((list(idx_modified) + list(previously_modified)) * n_repeat),
        batch_size=batch_size, drop_last=False)

    n = len(neighbors)
    print('{} neighbors'.format(n))
    n_repeat = 1 if (n == max_len or n == 0) else max_len / n + 1
    neighbor_sampler = torch.utils.data.BatchSampler(
        sampler=torch.utils.data.SubsetRandomSampler(list(neighbors) * n_repeat),
        batch_size=batch_size, drop_last=False)

    n = len(space_samples)
    print('{} space samples'.format(n))
    n_repeat = 1 if (n == max_len or n == 0) else max_len / n + 1
    space_sampler = torch.utils.data.BatchSampler(
        sampler=torch.utils.data.SubsetRandomSampler(list(space_samples) * n_repeat),
        batch_size=batch_size, drop_last=False)

    print('max_length loader: {}, ({}) ({}) ({})'
          .format(max_len / batch_size + 1, len(sample_sampler), len(neighbor_sampler), len(space_sampler)))

    # train network until scheduler reduces learning rate to threshold value
    track_l2_loss = ChangeRateLogger(n_track=5, threshold=5e-2)
    stop_criterion = False

    embeddings = {}
    model_states = {}
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
    while not stop_criterion:
        t_iter_start = time.time()

        # compute beta for kl loss
        t_beta_start = time.time()
        kl_criterion._compute_beta(new_features)
        t_beta_end = time.time()
        t_beta.append(t_beta_end - t_beta_start)

        # set up losses
        l2_losses = AverageMeter()
        kl_losses = AverageMeter()
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
        loaders = zip(list(space_sampler), list(sample_sampler), list(neighbor_sampler))
        for batch_idx, (space_indices, sample_indices, neighbor_indices) in enumerate(loaders):
            t_tot_start = time.time()

            # load data
            indices = np.concatenate([space_indices, sample_indices, neighbor_indices])
            if len(indices) < 3 * kl_criterion.perplexity + 2:
                continue
            data = tensor_feature[indices]
            input = torch.autograd.Variable(data.cuda()) if use_cuda else torch.autograd.Variable(data)

            t_load_end = time.time()
            t_load.append(t_load_end - t_load_start)

            # compute forward
            t_forward_start = time.time()

            fts_mod = net.mapping(input)
            emb_mod = net.embedder(torch.nn.functional.relu(fts_mod))

            t_forward_end = time.time()
            t_forward.append(t_forward_end - t_forward_start)

            # compute losses

            t_loss_start = time.time()

            kl_loss = kl_criterion(fts_mod, emb_mod, indices)

            kl_losses.update(kl_loss.data, len(data))

            idx_l2_fixed = np.concatenate([space_indices, sample_indices])
            l2_loss = torch.mean(
                l2_criterion(emb_mod[:len(idx_l2_fixed)], target_embedding[idx_l2_fixed].type_as(emb_mod)),
                dim=1)
            # weight loss of space samples equally to all modified samples
            l2_loss = 0.5 * torch.mean(l2_loss[:len(space_indices)]) + 0.5 * torch.mean(l2_loss[len(space_indices):])

            l2_losses.update(l2_loss.data, len(idx_l2_fixed))

            loss = 0.6 * l2_loss + 0.4 * kl_loss.type_as(l2_loss)
            losses.update(loss.data, len(data))

            t_loss_end = time.time()
            t_loss.append(t_loss_end - t_loss_start)
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
        log.write('l2_loss', float(l2_losses.avg), epoch, test=False)
        log.write('kl_loss', float(kl_losses.avg), epoch, test=False)
        log.write('loss', float(losses.avg), epoch, test=False)
        t_tensorboard_end = time.time()
        t_tensorboard.append(t_tensorboard_end - t_tensorboard_start)

        t_save_start = time.time()

        model_states[epoch] = {
            'epoch': epoch,
            'loss': losses.avg,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        embeddings[epoch] = new_embedding

        t_save_end = time.time()
        t_save.append(t_save_end - t_save_start)

        print('Train Epoch: {}\t'
              'Loss: {:.4f}\t'
              'L2 Loss: {:.4f}\t'
              'KL Loss: {:.4f}\t'
              'LR: {:.6f}'.format(
            epoch,
            float(losses.avg),
            float(l2_losses.avg),
            float(kl_losses.avg),
            optimizer.param_groups[-1]['lr']))

        t_send_start = time.time()

        # send to server
        if socket_id is not None:
            position = new_embedding if scale_func is None else scale_func(new_embedding)
            nodes = make_nodes(position=position, index=True, label=label)
            send_payload(nodes, socket_id)

        t_send_end = time.time()
        t_send.append(t_send_end - t_send_start)

        epoch += 1
        stop_criterion = track_l2_loss.add_value(l2_losses.avg)

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

    print('Save output files...')
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

    with h5py.File(outfile_feature, 'w') as f:
        f.create_dataset(name='feature', shape=new_features.shape, dtype=new_features.dtype, data=new_features)
        f.create_dataset(name='image_id', shape=image_id.shape, dtype=image_id.dtype, data=image_id)

    torch.save(model_states, outfile_model_states)

    # write config file
    config_dict = {'idx_modified': idx_modified, 'idx_old_neighbors': idx_old_neighbors,
                   'idx_new_neighbors': idx_new_neighbors, 'idx_high_dim_neighbors': idx_high_dim_neighbors}
    with open(outfile_config, 'w') as f:
        pickle.dump(config_dict, f)
    print('Done.')

    print('Finished training.')
    return new_embedding


if __name__ == '__main__':
    pass
