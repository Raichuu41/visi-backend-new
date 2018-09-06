import os
import sys
import time
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from faiss_master import faiss

from communication import send_payload, make_nodes

sys.path.append('../TSNENet')
from loss import TSNELoss
from dataset import IndexDataset

sys.path.append('../FullPipeline')
import matplotlib as mpl
mpl.use('TkAgg')
from aux import AverageMeter, TBPlotter, save_checkpoint, write_config, load_weights


# if not os.getcwd().endswith('/MapNetCode'):
#     os.chdir(os.path.join(os.getcwd(), 'MapNetCode'))
sys.path.append('./MapNetCode')
from model import MapNet
# from initialization import load_feature


# feature_file = 'features/MobileNetV4_info_artist_49_multilabel_test_full_images_128.hdf5'
# id, feature = load_feature(feature_file)
# net = MapNet(feature_dim=feature.shape[1], output_dim=2)
# weight_file = 'runs/embedder/08-03-12-48_TSNENet_large_model_model_best.pth.tar'


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
def train_embedder(embedder, feature, lr=1e-2, batch_size=100, experiment_id=None, random_state=123):
    # log and saving options
    timestamp = time.strftime('%m-%d-%H-%M')
    exp_name = 'MapNet_embedder_' + timestamp

    if experiment_id is not None:
        exp_name = experiment_id + '_' + exp_name

    log = TBPlotter(os.path.join('runs/embedder', 'tensorboard', exp_name))
    log.print_logdir()

    outpath_model = os.path.join('runs/embedder/models')
    os.makedirs(outpath_model)

    # general
    use_cuda = torch.cuda.is_available()
    N = len(feature)

    idx_train, idx_test = train_test_split(range(N), test_size=0.2, random_state=random_state, shuffle=True)
    kwargs = {'num_workers': 8} if use_cuda else {}
    train_loader = DataLoader(IndexDataset(feature[idx_train]), batch_size=batch_size,              # careful, returned index is now for idx_train selection
                              sampler=torch.utils.data.SubsetRandomSampler(idx_train), **kwargs)
    test_loader = DataLoader(IndexDataset(feature[idx_test]), batch_size=batch_size,                # careful, returned index is now for idx_test selection
                             sampler=torch.utils.data.SubsetRandomSampler(idx_test), **kwargs)

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
    lr_threshold = 1e-6
    epoch = 1

    best_loss = float('inf')
    while optimizer.param_groups[-1]['lr'] > lr_threshold:
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

    return os.path.join(outpath_model, exp_name)


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


def get_modified(old_embedding, new_embedding):
    return np.where(np.any(old_embedding != new_embedding, axis=1))[0]


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


def dummy_func(net, feature, embedding, idx_modified, idx_old_neighbors, idx_new_neighbors,
               lr=1e-4, experiment_id=None, socket_id=None, node_id=None):
    for i in range(10):
        new_embedding = embedding.copy()
        idx_move = np.unique(np.concatenate([idx_old_neighbors, idx_new_neighbors]))
        new_embedding[idx_move] += np.random.rand(len(idx_move), embedding.shape[1]) * 5

        # send to server
        if socket_id is not None:
            nodes = make_nodes(position=new_embedding, name=node_id)
            send_payload(nodes, socket_id)

        time.sleep(2)


def train(net, feature, embedding, idx_modified, idx_old_neighbors, idx_new_neighbors,
          lr=1e-4, experiment_id=None, socket_id=None, node_id=None):
    # log and saving options
    timestamp = time.strftime('%m-%d-%H-%M')
    exp_name = 'MapNet_' + timestamp

    if experiment_id is not None:
        exp_name = experiment_id + '_' + exp_name

    log = TBPlotter(os.path.join('MapNetCode/runs/mapping', 'tensorboard', exp_name))
    log.print_logdir()

    save_interval = 10
    outpath_model = os.path.join('MapNetCode/runs/mapping', exp_name, 'models')
    os.makedirs(outpath_model)
    outfile_embedding = h5py.File(os.path.join('MapNetCode/runs/mapping', 'embeddings.hdf5'), 'w')

    # general
    N = len(feature)
    use_cuda = torch.cuda.is_available()
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.from_numpy(embedding.copy())
    if use_cuda:
        net = net.cuda()
    net.train()

    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, threshold=1e-3,
                                                           verbose=True)

    kl_criterion = TSNELoss(N)
    l2_criterion = torch.nn.MSELoss()                  # keep the output fixed

    # define the data loaders
    #   sample_loader: modified samples
    #   neighbor_loader: previous and new neighbors
    #   fixpoint_loader: remaining samples (neither modified nor neighbors)

    kwargs = {'num_workers': 8} if use_cuda else {}
    sample_loader = DataLoader(IndexDataset(feature), batch_size=100,
                               sampler=torch.utils.data.SubsetRandomSampler(idx_modified), **kwargs)

    neighbor_loader = DataLoader(IndexDataset(feature), batch_size=100,
                                 sampler=torch.utils.data.SubsetRandomSampler(
                                     np.unique(np.concatenate([idx_old_neighbors, idx_new_neighbors]))),
                                 **kwargs)

    # TODO: test performance with random samples instead of whole dataset
    fixpoint_loader = DataLoader(IndexDataset(feature), batch_size=200,
                                 sampler=torch.utils.data.SubsetRandomSampler(
                                     np.setdiff1d(range(N), np.concatenate([idx_modified,
                                                                            idx_old_neighbors,
                                                                            idx_new_neighbors]))),
                                 **kwargs)

    # train network until scheduler reduces learning rate to threshold value
    lr_threshold = 1e-6
    log_interval = 10
    epoch = 1
    new_features = feature.copy()
    new_embedding = embedding.numpy().copy()
    while optimizer.param_groups[-1]['lr'] > lr_threshold:
        # compute beta for kl loss
        kl_criterion._compute_beta(new_features)

        # set up losses
        l2_losses = AverageMeter()
        kl_losses = AverageMeter()
        losses = AverageMeter()

        # iterate over fix points (assume N_fixpoints >> N_modified)
        for batch_idx, (fixpoint_data, fixpoint_indices) in enumerate(fixpoint_loader):
            sample_data, sample_indices = next(iter(sample_loader))
            neighbor_data, neighbor_indices = next(iter(neighbor_loader))
            data = torch.cat([fixpoint_data, sample_data, neighbor_data])
            indices = torch.cat([fixpoint_indices, neighbor_indices, sample_indices])

            input = torch.autograd.Variable(data.cuda()) if use_cuda else torch.autograd.Variable(data)
            fts_mod = net.mapping(input)
            emb_mod = net.embedder(torch.nn.functional.relu(fts_mod))

            kl_loss = kl_criterion(fts_mod, emb_mod, indices)
            kl_losses.update(kl_loss.data, len(data))

            idx_l2_fixed = torch.cat([fixpoint_indices, sample_indices])
            l2_loss = l2_criterion(emb_mod[:len(idx_l2_fixed)], embedding[idx_l2_fixed].type_as(emb_mod))
            l2_losses.update(l2_loss.data, len(idx_l2_fixed))

            loss = 0.6 * l2_loss + 0.4 * kl_loss.type_as(l2_loss)
            losses.update(loss.data, len(data))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update current embedding
            new_embedding[indices] = emb_mod.data.cpu().numpy()

            # if batch_idx % log_interval == 0:
            #     print('Train Epoch: {} [{}/{}]\t'
            #           'Loss: {:.4f} ({:.4f})\t'
            #           'LR: {:.6f}'.format(
            #         epoch, (batch_idx + 1) * len(fixpoint_indices), len(fixpoint_loader.sampler),
            #         float(losses.val), float(losses.avg),
            #         optimizer.param_groups[-1]['lr']))

        scheduler.step(losses.avg)
        log.write('l2_loss', float(l2_losses.avg), epoch, test=False)
        log.write('kl_loss', float(kl_losses.avg), epoch, test=False)
        log.write('loss', float(losses.avg), epoch, test=False)

        if epoch % save_interval == 0:
            torch.save({
                'epoch': epoch,
                'loss': losses.avg,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, os.path.join(outpath_model, 'epoch_{:04d}'.format(epoch) + '.pth.tar'))

        outfile_embedding.create_dataset(name='epoch_{:04d}'.format(epoch),
                                         shape=new_embedding.shape, dtype=new_embedding.dtype,
                                         data=new_embedding)

        print('Train Epoch: {}\t'
              'Loss: {:.4f}\t'
              'LR: {:.6f}'.format(
            epoch,
            float(losses.avg),
            optimizer.param_groups[-1]['lr']))

        # send to server
        if socket_id is not None:
            nodes = make_nodes(position=new_embedding, name=node_id)
            send_payload(nodes, socket_id)

        epoch += 1
    outfile_embedding.close()


