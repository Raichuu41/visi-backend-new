import os
import sys
import time
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from faiss_master import faiss

from communication import send_payload, make_nodes

sys.path.append('../FullPipeline')
import matplotlib as mpl
mpl.use('TkAgg')
from aux import AverageMeter, TBPlotter, save_checkpoint, write_config, plot_embedding_2d, load_weights

sys.path.append('../TSNENet')
from loss import TSNELoss
from dataset import IndexDataset


if not os.getcwd().endswith('/MapNetCode'):
    os.chdir(os.path.join(os.getcwd(), 'MapNetCode'))
sys.path.append(os.getcwd())
from model import MapNet
from initialization import load_feature


feature_file = 'features/MobileNetV4_info_artist_49_multilabel_test_full_images_128.hdf5'
id, feature = load_feature(feature_file)
net = MapNet(feature_dim=feature.shape[1], output_dim=2)

weight_file = 'runs/embedder/08-03-12-48_TSNENet_large_model_model_best.pth.tar'


def initialize_embedder(net, weight_file=None):
    """Compute the initial weights for the embedding network.
    If weight_file is specified, simply load weights from there."""

    if weight_file is None:
        raise NotImplementedError('Training embedder not yet implemented.')

    if not os.path.isfile(weight_file):
        raise RuntimeError('Weight file not found.')

    # load weights of TSNENet
    pretrained_dict = load_weights(weight_file, net.embedder.state_dict(), prefix_file='embedder.')
    net.embedder.load_state_dict(pretrained_dict)


def compute_embedding(embedder, feature):
    print('Compute initial embedding...')
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


# TODO: DO NOT FORGET THIS BEFORE TRAINING!
# add a tiny bit of noise, so each parameter in mapping contributes     # TODO: CHECK IF THIS IS ACTUALLY HELPING
# for name, param in net.mapping.named_parameters():
#     if name.endswith('weight'):
#         param.data.copy_(param.data + torch.rand(param.shape).type_as(param.data) * 1e-8)
def train(net, feature, embedding, idx_modified, idx_old_neighbors, idx_new_neighbors,
          lr=1e-4, experiment_id=None, socket_id=None):
    # log and saving options
    timestamp = time.strftime('%m-%d-%H-%M')
    exp_name = 'MapNet_' + timestamp

    if experiment_id is not None:
        exp_name = experiment_id + '_' + exp_name

    log = TBPlotter(os.path.join('runs/mapping', 'tensorboard', exp_name))
    log.print_logdir()

    save_interval = 10
    outpath_model = os.path.join('runs/mapping', exp_name, 'models')
    os.makedirs(outpath_model)
    outfile_embedding = h5py.File(os.path.join('runs/mapping', 'embeddings.hdf5'), 'w')

    # general
    N = len(feature)
    use_cuda = torch.cuda.is_available()
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.from_numpy(embedding.copy())
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
                                     np.concatenate([idx_old_neighbors, idx_new_neighbors])),
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
            emb_mod = net.embedder(torch.nn.funcional.relu(fts_mod))

            kl_loss = kl_criterion(fts_mod, emb_mod, indices)
            kl_losses.update(kl_loss.data, len(data))

            l2_loss = l2_criterion(emb_mod, embedding[torch.cat(fixpoint_indices, sample_indices)].type_as(emb_mod))
            l2_losses.update(l2_loss.data, len(data))

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
            torch.save({'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        }, os.path.join(outpath_model, 'epoch_{:04d}'.format(epoch)))

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
            nodes = make_nodes(position=new_embedding)
            send_payload(nodes, socket_id)

        epoch += 1
    outfile_embedding.close()

