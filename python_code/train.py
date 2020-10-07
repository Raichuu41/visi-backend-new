import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import collections
from sklearn.model_selection import train_test_split
import deepdish as dd
import warnings
from functools import partial

from .helpers import IndexDataset, BalancedBatchSampler, PartiallyLabeledBatchSampler
from .aux import AverageMeter, TBPlotter, load_weights
from .loss import (TSNEWrapper, TSNEWrapperMapNet, TripletLossWrapper, TripletSelector, select_semihard,
    L1RegWrapper, select_random, TripletLossWrapper_pretraining)
from .model import MapNet, load_featurenet


if not torch.cuda.is_available():
    warnings.warn('GPU not available - Computation Times might be very long.')


def train(model_fn, dataloader, optimizer, loss_fns, epoch, loss_fns_weights=None,
          use_gpu=True, verbose=False, log_interval=None, logger=None):
    """
    :param model_fn:
    :param dataloader:
    :param optimizer:
    :param loss_fns: iterable containing the loss functions. Loss functions have to take the arguments (data, output).
    :param epoch:
    :param loss_fns_weights:
    :param use_gpu:
    :param verbose:
    :param log_interval:
    :param log_dir:
    :return:
    """
    if not isinstance(loss_fns, collections.Iterable):
        loss_fns = (loss_fns, )

    avg_meters = [AverageMeter() for i in range(len(loss_fns) + 1)]     # additional for the sum of all losses

    if loss_fns_weights is not None:
        if not isinstance(loss_fns_weights, collections.Iterable):
            loss_fns_weights = (loss_fns_weights, )

        if len(loss_fns_weights) != len(loss_fns):
            raise RuntimeError('Number of loss functions ({}) and number of weights for them ({}) do not match.'
                               .format(len(loss_fns), len(loss_fns_weights)))

    else:
        loss_fns_weights = (1, ) * len(loss_fns)

    for i, data in enumerate(dataloader):
        if verbose:
            print('{}/{}'.format(i+1, len(dataloader)))
        if isinstance(data, list) or isinstance(data, tuple):
            input = data[0].cuda() if use_gpu else data[0]
        else:
            input = data.cuda() if use_gpu else data
        output = model_fn(input)

        loss = torch.tensor(0., device=input.device, requires_grad=True)
        for loss_fn, loss_fn_weight, avg_meter in zip(loss_fns, loss_fns_weights, avg_meters):
            loss_ = loss_fn_weight * loss_fn(data, output)
            avg_meter.update(loss_.data.cpu(), input.shape[0])
            loss = loss + loss_.type_as(loss)

        avg_meters[-1].update(loss.data.cpu(), input.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if log_interval is not None and (i+1) % log_interval == 0:
            losses_message = '\t'.join(['{}: {:.4f} ({:.4f})'.format(loss_fn_name, avg_meter.val, avg_meter.avg)
                            for loss_fn_name, avg_meter in zip(loss_fns, avg_meters)])

            log_message = \
                'Train Epoch: {}' \
                '\tLosses:' \
                '\t{}' \
                '\tTotal Loss: {:.4f} ({:.4f})' \
                '\tLR: {:.6f}'.format(
                    epoch,
                    losses_message,
                    avg_meters[-1].val, avg_meters[-1].avg,
                    optimizer.param_groups[-1]['lr']
                )
            print(log_message)

    losses_message = '\t'.join(['{}: {:.4f} ({:.4f})'.format(loss_fn_name, avg_meter.val, avg_meter.avg)
                                    for loss_fn_name, avg_meter in zip(loss_fns, avg_meters)])
    log_message = \
        'Train Epoch: {}' \
        '\tLosses:' \
        '\t{}' \
        '\tTotal Loss: {:.4f}' \
        '\tLR: {:.6f}'.format(
            epoch,
            losses_message,
            avg_meters[-1].avg,
            optimizer.param_groups[-1]['lr']
        )
    print(log_message)

    if logger is not None:
        for loss_fn, avg_meter in zip(loss_fns, avg_meters):
            logger.write('{}'.format(loss_fn), float(avg_meter.avg), epoch, test=False)

    return [avg_meter.avg for avg_meter in avg_meters]


def test(model_fn, dataloader, loss_fns, epoch, loss_fns_weights=None, lr_scheduler=None,
         use_gpu=True, verbose=False, logger=None):

    if not isinstance(loss_fns, collections.Iterable):
        loss_fns = (loss_fns, )

    avg_meters = [AverageMeter() for i in range(len(loss_fns) + 1)]     # additional for the sum of all losses

    if loss_fns_weights is not None:
        if not isinstance(loss_fns_weights, collections.Iterable):
            loss_fns_weights = (loss_fns_weights, )

        if len(loss_fns_weights) != len(loss_fns):
            raise RuntimeError('Number of loss functions ({}) and number of weights for them ({}) do not match.'
                               .format(len(loss_fns), len(loss_fns_weights)))

    else:
        loss_fns_weights = (1, ) * len(loss_fns)

    for i, data in enumerate(dataloader):
        if verbose:
            print('{}/{}'.format(i+1, len(dataloader)))
        if isinstance(data, list) or isinstance(data, tuple):
            input = data[0].cuda() if use_gpu else data[0]
        else:
            input = data.cuda() if use_gpu else data
        output = model_fn(input)

        loss = torch.tensor(0., device=input.device, requires_grad=False)
        for loss_fn, loss_fn_weight, avg_meter in zip(loss_fns, loss_fns_weights, avg_meters):
            loss_ = loss_fn_weight * loss_fn(data, output)
            avg_meter.update(loss_.data.cpu(), input.shape[0])
            loss = loss + loss_.type_as(loss)

        avg_meters[-1].update(loss.data.cpu(), input.shape[0])
    testloss = avg_meters[-1].avg
    if lr_scheduler is not None:
        lr_scheduler.step(testloss)

    losses_message = '\t'.join(['{}: {:.4f} ({:.4f})'.format(loss_fn_name, avg_meter.val, avg_meter.avg)
                                    for loss_fn_name, avg_meter in zip(loss_fns, avg_meters)])

    log_message = \
        'Test Epoch: {}' \
        '\tLosses:' \
        '\t{}' \
        '\tTotal Loss: {:.4f}'.format(
            epoch,
            losses_message,
            testloss,
        )
    print(log_message)

    if logger is not None:
        for loss_fn, avg_meter in zip(loss_fns, avg_meters):
            logger.write('{}'.format(loss_fn), float(avg_meter.avg), epoch, test=True)

    return [avg_meter.avg for avg_meter in avg_meters]


def train_embedder(embedder, features, lr=1e-3, batch_size=2000, random_state=123, use_gpu=True,
                   verbose=False, outpath=None, max_epochs=100, log_dir=None, plot_fn=None):
    if outpath is not None and not os.path.isdir(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))

    if log_dir is not None:
        logger = TBPlotter(log_dir)
        logger.print_logdir()
    else:
        logger = None

    loader_kwargs = {'num_workers': 4} if use_gpu else {}

    feature_train, feature_test = train_test_split(features, test_size=0.2, random_state=random_state, shuffle=True)
    trainloader = DataLoader(IndexDataset(feature_train), batch_size=batch_size, shuffle=True,
                             drop_last=True,
                             **loader_kwargs)
    testloader = DataLoader(IndexDataset(feature_test), batch_size=batch_size, shuffle=False,
                            drop_last=len(feature_test) > batch_size,       # drop last only if there are enough samples for one full batch
                             **loader_kwargs)

    optimizer = optim.Adam(embedder.parameters(), lr=lr, weight_decay=2e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=1e-3,
                                                        verbose=verbose)
    loss_fn_train = TSNEWrapper(N=len(feature_train), use_gpu=use_gpu)
    loss_fn_test = TSNEWrapper(N=len(feature_test), use_gpu=use_gpu)

    if verbose:
        print('Compute beta for TSNE-Loss...')
    loss_fn_train._compute_beta(torch.from_numpy(feature_train))
    loss_fn_test._compute_beta(torch.from_numpy(feature_test))
    if verbose:
        print('Done.')

    # train network until scheduler reduces learning rate to threshold value
    lr_threshold = 5e-5
    epoch = 1

    best_loss = float('inf')
    best_epoch = -1
    while optimizer.param_groups[-1]['lr'] >= lr_threshold:
        train(embedder.forward, trainloader, optimizer, loss_fn_train, epoch,
              use_gpu=use_gpu, verbose=verbose, logger=logger)
        testlosses = test(embedder.forward, testloader, loss_fn_test, epoch, lr_scheduler=lr_scheduler,
                        use_gpu=use_gpu, verbose=verbose, logger=logger)
        testloss = testlosses[-1]
        if testloss < best_loss:
            best_loss = testloss
            best_epoch = epoch
            if outpath is not None:         # save checkpoint
                torch.save({
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'state_dict': embedder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict()
                }, outpath)

        if plot_fn is not None:         # compute 2-dimensional embedding and plot it with plot_fn
            projection = evaluate_model(embedder.forward, data=features, batch_size=batch_size,
                                        use_gpu=use_gpu, verbose=False)
            plot_fn(projection.numpy(), epoch=epoch)

        epoch += 1
        if epoch > max_epochs:
            break

    print('Finished training embedder with best loss: {:.4f} from epoch {}'.format(best_loss, best_epoch))


def train_mapnet(model, features, labels, weights=None,
                 test_features=None, test_labels=None,
                 lr=1e-3, batch_size=2000, batch_frac_labeled=0.7,
                 random_state=123, use_gpu=True,
                 verbose=False, outpath=None, log_dir=None, max_epochs=float('inf'),
                 plot_fn=None):
    use_test = test_features is not None and test_labels is not None
    use_multi_features = features.ndim == 3      # multi features have shape FTS_VARIANTS X N_SAMPLES X FEATURE_DIM
    multi_features = None if not use_multi_features else features.copy()

    def sample_features():
        features = multi_features.copy()

        # pick a random feature variant for each sample
        features = np.stack([
            features[np.random.randint(0, len(features)), i, :]
            for i in range(features.shape[1])
        ])

        return features

    if use_multi_features:
        features = sample_features()

    if not isinstance(labels, torch.LongTensor):
        labels = torch.LongTensor(labels)
    if use_test and not isinstance(test_labels, torch.LongTensor):
        test_labels = torch.LongTensor(test_labels)
    if weights is not None and not isinstance(weights, torch.FloatTensor):
        weights = torch.FloatTensor(weights)

    def eval_fn(input):
        fts = model.mapping.forward(input)
        fts = fts / fts.norm(dim=1, keepdim=True)
        proj = model.embedder.forward(fts)

        return fts, proj

    if outpath is not None and not os.path.isdir(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))

    if log_dir is not None:
        logger = TBPlotter(log_dir)
        logger.print_logdir()
    else:
        logger = None

    loader_kwargs = {'num_workers': 4} if use_gpu else {}

    N_unlabeled_total = batch_size
    classweights = {l: 1./3 if l < 0 else 2./3 for l in set(labels.numpy()) if l != 0}      # 2/3 positive labeled, 1/3 class specific negatives
    batchsampler = PartiallyLabeledBatchSampler(labels, frac_labeled=batch_frac_labeled, batch_size=batch_size,
                                                N_unlabeled_total=N_unlabeled_total, classweights=classweights)

    trainloader = DataLoader(IndexDataset(features), batch_sampler=batchsampler,
                             **loader_kwargs)
    if use_test:
        testloader = DataLoader(IndexDataset(test_features), batch_size=batch_size, shuffle=False, drop_last=False,
                                **loader_kwargs)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=1e-3,
                                                        verbose=verbose)

    triplet_selector = TripletSelector(margin=torch.tensor(0.2), negative_selection_fn=select_semihard)
    loss_fns_train = [
        TripletLossWrapper(triplet_selector, labels, data_weights=weights,
                           N_labels_per_minibatch=10, N_samples_per_label=5, N_pure_negatives=15,
                           average=True),
        TSNEWrapperMapNet(N=len(features), use_gpu=use_gpu),
        # L1RegWrapper(model=model.mapping)
    ]
    if use_test:
        triplet_selector_test = TripletSelector(margin=torch.tensor(0.2), negative_selection_fn=select_random)
        loss_fns_test = [
            TripletLossWrapper(triplet_selector_test, test_labels,
                               N_labels_per_minibatch=10, N_samples_per_label=5, N_pure_negatives=15,
                               average=True),
            TSNEWrapperMapNet(N=len(test_features), use_gpu=use_gpu)
        ]

    # train network until scheduler reduces learning rate to threshold value
    lr_threshold = 5e-5
    epoch = 1

    best_loss = float('inf')
    best_loss_test = float('inf')
    best_epoch = -1
    stopcriterion = False
    while not stopcriterion:
        if use_multi_features:          # for each epoch sample random feature variants
            features = sample_features()

        # compute beta for mapped features
        if verbose:
            print('Compute beta for TSNE-Loss...')
        mapped_features = torch.from_numpy(features) if epoch == 1 else \
            evaluate_model(model.mapping, features, batch_size=batch_size, use_gpu=use_gpu, verbose=verbose)
        mapped_features /= mapped_features.norm(dim=1, keepdim=True)
        loss_fns_train[1]._compute_beta(mapped_features)
        if verbose:
            print('Done.')
        model.train()
        trainlosses = train(eval_fn, trainloader, optimizer, loss_fns_train, epoch, loss_fns_weights=(100, 1),
                            use_gpu=use_gpu, verbose=verbose, logger=logger)
        trainloss = trainlosses[-1]
        if use_test:
            model.eval()
            testlosses = test(eval_fn, testloader, loss_fns_test, epoch, loss_fns_weights=(100, 1),
                              use_gpu=use_gpu, verbose=verbose, logger=logger)
            testloss = testlosses[-1]
        else:
            lr_scheduler.step(trainloss)
        if trainloss < best_loss:
            best_loss = trainloss
            best_epoch = epoch
            if outpath is not None:
                torch.save({                # save best model
                    'state_dict': model.state_dict(),
                }, outpath)

        if use_test and testloss < best_loss_test:
            best_loss_test = testloss
            if outpath is not None:
                torch.save({                # save best model
                    'state_dict': model.state_dict(),
                }, outpath.replace('.pth.tar', '_test.pth.tar'))

        if log_dir is not None:         # save checkpoint
            torch.save({
                'epoch': epoch,
                'best_loss': best_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict()
            }, os.path.join(log_dir, 'checkpoint.pth.tar'))

        if plot_fn is not None:         # compute 2-dimensional embedding and plot it with plot_fn
            projection = evaluate_model(model.forward, data=features, batch_size=batch_size, use_gpu=use_gpu, verbose=False)
            plot_fn(projection.numpy(), epoch=epoch)

        epoch += 1
        if epoch > max_epochs:
            break
        stopcriterion = (optimizer.param_groups[-1]['lr'] < lr_threshold) or \
                        (trainlosses[0] < 1)

    # refine projection
    best_weights = load_weights(outpath, model.state_dict())
    model.load_state_dict(best_weights)
    model.eval()
    mapped_features = evaluate_model(model.mapping, features, batch_size=batch_size, use_gpu=use_gpu, verbose=verbose)
    mapped_features /= mapped_features.norm(dim=1, keepdim=True)
    train_embedder(model.embedder, mapped_features.cpu().numpy(), lr=1e-3, batch_size=2000,
                   random_state=random_state, use_gpu=use_gpu,
                   verbose=verbose, outpath=None, log_dir=None, plot_fn=plot_fn)

    # save model with refined projection
    if outpath is not None:
        torch.save({
            'state_dict': model.state_dict(),
        }, outpath)

    print('Finished training mapnet with best training loss: {:.4f} from epoch {}.'.format(best_loss, best_epoch))
    if outpath is not None:
        print('Saved model to {}.'.format(outpath))

def train_mapnet_cosine(model, features, labels, weights=None,
                        test_features=None, test_labels=None,
                        lr=1e-3, batch_size=2000, batch_frac_labeled=0.7,
                        random_state=123, use_gpu=True,
                        verbose=False, outpath=None, log_dir=None,
                        max_epochs=float('inf'), plot_fn=None):
    use_test = test_features is not None and test_labels is not None
    use_multi_features = features.ndim == 3      # multi features have shape FTS_VARIANTS X N_SAMPLES X FEATURE_DIM
    multi_features = None if not use_multi_features else features.copy()

    def sample_features():
        features = multi_features.copy()

        # pick a random feature variant for each sample
        features = np.stack([
            features[np.random.randint(0, len(features)), i, :]
            for i in range(features.shape[1])
        ])

        return features

    if use_multi_features:
        features = sample_features()

    if not isinstance(labels, torch.LongTensor):
        labels = torch.LongTensor(labels)
    if use_test and not isinstance(test_labels, torch.LongTensor):
        test_labels = torch.LongTensor(test_labels)
    if weights is not None and not isinstance(weights, torch.FloatTensor):
        weights = torch.FloatTensor(weights)

    def eval_fn(input):
        fts = model.mapping.forward(input)
        fts = fts / fts.norm(dim=1, keepdim=True)
        proj = model.embedder.forward(fts)

        return fts, proj

    if outpath is not None and not os.path.isdir(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))

    if log_dir is not None:
        logger = TBPlotter(log_dir)
        logger.print_logdir()
    else:
        logger = None

    loader_kwargs = {'num_workers': 4} if use_gpu else {}

    N_unlabeled_total = batch_size
    classweights = {l: 1./3 if l < 0 else 2./3 for l in set(labels.numpy()) if l != 0}      # 2/3 positive labeled, 1/3 class specific negatives
    batchsampler = PartiallyLabeledBatchSampler(labels, frac_labeled=batch_frac_labeled, batch_size=batch_size,
                                                N_unlabeled_total=N_unlabeled_total, classweights=classweights)

    trainloader = DataLoader(IndexDataset(features), batch_sampler=batchsampler,
                             **loader_kwargs)
    if use_test:
        testloader = DataLoader(IndexDataset(test_features), batch_size=batch_size, shuffle=False, drop_last=False,
                                **loader_kwargs)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=1e-3,
                                                        verbose=verbose)

    triplet_selector = TripletSelector(margin=torch.tensor(0.2), negative_selection_fn=select_semihard)
    loss_fns_train = [
        TripletLossWrapper(triplet_selector, labels, data_weights=weights,
                           N_labels_per_minibatch=10, N_samples_per_label=5, N_pure_negatives=15,
                           average=True),
        TSNEWrapperMapNet(N=len(features), use_gpu=use_gpu),
        # L1RegWrapper(model=model.mapping)
    ]
    if use_test:
        triplet_selector_test = TripletSelector(margin=torch.tensor(0.2), negative_selection_fn=select_random)
        loss_fns_test = [
            TripletLossWrapper(triplet_selector_test, test_labels,
                               N_labels_per_minibatch=10, N_samples_per_label=5, N_pure_negatives=15,
                               average=True),
            TSNEWrapperMapNet(N=len(test_features), use_gpu=use_gpu)
        ]

    # train network until scheduler reduces learning rate to threshold value
    lr_threshold = 5e-5
    epoch = 1

    best_loss = float('inf')
    best_loss_test = float('inf')
    best_epoch = -1
    stopcriterion = False
    while not stopcriterion:
        if use_multi_features:          # for each epoch sample random feature variants
            features = sample_features()

        # compute beta for mapped features
        if verbose:
            print('Compute beta for TSNE-Loss...')
        mapped_features = torch.from_numpy(features) if epoch == 1 else \
            evaluate_model(model.mapping, features, batch_size=batch_size, use_gpu=use_gpu, verbose=verbose)
        mapped_features /= mapped_features.norm(dim=1, keepdim=True)
        loss_fns_train[1]._compute_beta(mapped_features)
        if verbose:
            print('Done.')
        model.train()
        trainlosses = train(eval_fn, trainloader, optimizer, loss_fns_train, epoch, loss_fns_weights=(100, 1),
                            use_gpu=use_gpu, verbose=verbose, logger=logger)
        trainloss = trainlosses[-1]
        if use_test:
            model.eval()
            testlosses = test(eval_fn, testloader, loss_fns_test, epoch, loss_fns_weights=(100, 1),
                              use_gpu=use_gpu, verbose=verbose, logger=logger)
            testloss = testlosses[-1]
        else:
            lr_scheduler.step(trainloss)
        if trainloss < best_loss:
            best_loss = trainloss
            best_epoch = epoch
            if outpath is not None:
                torch.save({                # save best model
                    'state_dict': model.state_dict(),
                }, outpath)

        if use_test and testloss < best_loss_test:
            best_loss_test = testloss
            if outpath is not None:
                torch.save({                # save best model
                    'state_dict': model.state_dict(),
                }, outpath.replace('.pth.tar', '_test.pth.tar'))

        if log_dir is not None:         # save checkpoint
            torch.save({
                'epoch': epoch,
                'best_loss': best_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict()
            }, os.path.join(log_dir, 'checkpoint.pth.tar'))

        if plot_fn is not None:         # compute 2-dimensional embedding and plot it with plot_fn
            projection = evaluate_model(model.forward, data=features, batch_size=batch_size, use_gpu=use_gpu, verbose=False)
            plot_fn(projection.numpy(), epoch=epoch)

        epoch += 1
        if epoch > max_epochs:
            break
        stopcriterion = (optimizer.param_groups[-1]['lr'] < lr_threshold) or \
                        (trainlosses[0] < 1)

    # refine projection
    best_weights = load_weights(outpath, model.state_dict())
    model.load_state_dict(best_weights)
    model.eval()
    mapped_features = evaluate_model(model.mapping, features, batch_size=batch_size, use_gpu=use_gpu, verbose=verbose)
    mapped_features /= mapped_features.norm(dim=1, keepdim=True)
    train_embedder(model.embedder, mapped_features.cpu().numpy(), lr=1e-3, batch_size=2000,
                   random_state=random_state, use_gpu=use_gpu,
                   verbose=verbose, outpath=None, log_dir=None, plot_fn=plot_fn)

    # save model with refined projection
    if outpath is not None:
        torch.save({
            'state_dict': model.state_dict(),
        }, outpath)

    print('Finished training mapnet with best training loss: {:.4f} from epoch {}.'.format(best_loss, best_epoch))
    if outpath is not None:
        print('Saved model to {}.'.format(outpath))


def train_mapnet_weights(model, features, labels, weights=None,
                 test_features=None, test_labels=None,
                 lr=1e-3, batch_size=2000, batch_frac_labeled=0.7,
                 random_state=123, use_gpu=True,
                 verbose=False, outpath=None, log_dir=None, max_epochs=float('inf'),
                 plot_fn=None):
    use_test = test_features is not None and test_labels is not None
    use_multi_features = features.ndim == 3      # multi features have shape FTS_VARIANTS X N_SAMPLES X FEATURE_DIM
    multi_features = None if not use_multi_features else features.copy()

    def sample_features():
        features = multi_features.copy()

        # pick a random feature variant for each sample
        features = np.stack([
            features[np.random.randint(0, len(features)), i, :]
            for i in range(features.shape[1])
        ])

        return features

    if use_multi_features:
        features = sample_features()

    if not isinstance(labels, torch.LongTensor):
        labels = torch.LongTensor(labels)
    if use_test and not isinstance(test_labels, torch.LongTensor):
        test_labels = torch.LongTensor(test_labels)
    if weights is not None and not isinstance(weights, torch.FloatTensor):
        weights = torch.FloatTensor(weights)

    def eval_fn(input):
        fts = model.mapping.forward(input)
        fts = fts / fts.norm(dim=1, keepdim=True)
        proj = model.embedder.forward(fts)

        return fts, proj

    if outpath is not None and not os.path.isdir(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))

    if log_dir is not None:
        logger = TBPlotter(log_dir)
        logger.print_logdir()
    else:
        logger = None

    loader_kwargs = {'num_workers': 4} if use_gpu else {}

    N_unlabeled_total = batch_size
    classweights = {l: 1./3 if l < 0 else 2./3 for l in set(labels.numpy()) if l != 0}      # 2/3 positive labeled, 1/3 class specific negatives
    batchsampler = PartiallyLabeledBatchSampler(labels, frac_labeled=batch_frac_labeled, batch_size=batch_size,
                                                N_unlabeled_total=N_unlabeled_total, classweights=classweights)

    trainloader = DataLoader(IndexDataset(features), batch_sampler=batchsampler,
                             **loader_kwargs)
    if use_test:
        testloader = DataLoader(IndexDataset(test_features), batch_size=batch_size, shuffle=False, drop_last=False,
                                **loader_kwargs)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=1e-3,
                                                        verbose=verbose)

    triplet_selector = TripletSelector(margin=torch.tensor(0.2), negative_selection_fn=select_semihard)
    loss_fns_train = [
        TripletLossWrapper(triplet_selector, labels, data_weights=weights,
                           N_labels_per_minibatch=10, N_samples_per_label=5, N_pure_negatives=15,
                           average=True),
        TSNEWrapperMapNet(N=len(features), use_gpu=use_gpu),
        # L1RegWrapper(model=model.mapping)
    ]
    if use_test:
        triplet_selector_test = TripletSelector(margin=torch.tensor(0.2), negative_selection_fn=select_random)
        loss_fns_test = [
            TripletLossWrapper(triplet_selector_test, test_labels,
                               N_labels_per_minibatch=10, N_samples_per_label=5, N_pure_negatives=15,
                               average=True),
            TSNEWrapperMapNet(N=len(test_features), use_gpu=use_gpu)
        ]

    # train network until scheduler reduces learning rate to threshold value
    lr_threshold = 5e-5
    epoch = 1

    best_loss = float('inf')
    best_loss_test = float('inf')
    best_epoch = -1
    stopcriterion = False
    while not stopcriterion:
        if use_multi_features:          # for each epoch sample random feature variants
            features = sample_features()

        # compute beta for mapped features
        if verbose:
            print('Compute beta for TSNE-Loss...')
        mapped_features = torch.from_numpy(features) if epoch == 1 else \
            evaluate_model(model.mapping, features, batch_size=batch_size, use_gpu=use_gpu, verbose=verbose)
        mapped_features /= mapped_features.norm(dim=1, keepdim=True)
        loss_fns_train[1]._compute_beta(mapped_features)
        if verbose:
            print('Done.')
        model.train()
        trainlosses = train(eval_fn, trainloader, optimizer, loss_fns_train, epoch, loss_fns_weights=(1000, 1),
                            use_gpu=use_gpu, verbose=verbose, logger=logger)
        trainloss = trainlosses[-1]
        if use_test:
            model.eval()
            testlosses = test(eval_fn, testloader, loss_fns_test, epoch, loss_fns_weights=(1000, 1),
                              use_gpu=use_gpu, verbose=verbose, logger=logger)
            testloss = testlosses[-1]
        else:
            lr_scheduler.step(trainloss)
        if trainloss < best_loss:
            best_loss = trainloss
            best_epoch = epoch
            if outpath is not None:
                torch.save({                # save best model
                    'state_dict': model.state_dict(),
                }, outpath)

        if use_test and testloss < best_loss_test:
            best_loss_test = testloss
            if outpath is not None:
                torch.save({                # save best model
                    'state_dict': model.state_dict(),
                }, outpath.replace('.pth.tar', '_test.pth.tar'))

        if log_dir is not None:         # save checkpoint
            torch.save({
                'epoch': epoch,
                'best_loss': best_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict()
            }, os.path.join(log_dir, 'checkpoint.pth.tar'))

        if plot_fn is not None:         # compute 2-dimensional embedding and plot it with plot_fn
            projection = evaluate_model(model.forward, data=features, batch_size=batch_size, use_gpu=use_gpu, verbose=False)
            plot_fn(projection.numpy(), epoch=epoch)

        epoch += 1
        if epoch > max_epochs:
            break
        stopcriterion = (optimizer.param_groups[-1]['lr'] < lr_threshold) or \
                        (trainlosses[0] < 1)

    # refine projection
    best_weights = load_weights(outpath, model.state_dict())
    model.load_state_dict(best_weights)
    model.eval()
    mapped_features = evaluate_model(model.mapping, features, batch_size=batch_size, use_gpu=use_gpu, verbose=verbose)
    mapped_features /= mapped_features.norm(dim=1, keepdim=True)
    train_embedder(model.embedder, mapped_features.cpu().numpy(), lr=1e-3, batch_size=2000,
                   random_state=random_state, use_gpu=use_gpu,
                   verbose=verbose, outpath=None, log_dir=None, plot_fn=plot_fn)

    # save model with refined projection
    if outpath is not None:
        torch.save({
            'state_dict': model.state_dict(),
        }, outpath)

    print('Finished training mapnet with best training loss: {:.4f} from epoch {}.'.format(best_loss, best_epoch))
    if outpath is not None:
        print('Saved model to {}.'.format(outpath))



def pretrain_mapnet(model, trainset, labels, testset, test_labels,
                 lr=1e-3, random_state=123, use_gpu=True,
                 verbose=False, outpath=None, log_dir=None,
                 plot_fn=None):

    if not isinstance(labels, torch.LongTensor):
        labels = torch.LongTensor(labels)
    if not isinstance(test_labels, torch.LongTensor):
        test_labels = torch.LongTensor(test_labels)

    def eval_fn(input):
        fts = model.forward(input)
        return fts, None

    if outpath is not None and not os.path.isdir(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))

    if log_dir is not None:
        logger = TBPlotter(log_dir)
        logger.print_logdir()
    else:
        logger = None

    loader_kwargs = {'num_workers': 4} if use_gpu else {}

    trainloader = DataLoader(IndexDataset(trainset),
                             batch_sampler=BalancedBatchSampler(labels.cpu().numpy(),
                                        n_labels=10,
                                        n_samples=5,
                                        n_pure_negatives=0),
                             **loader_kwargs)
    testloader = DataLoader(IndexDataset(testset), batch_sampler=BalancedBatchSampler(test_labels.cpu().numpy(),
                                        n_labels=10,
                                        n_samples=5,
                                        n_pure_negatives=0),
                                **loader_kwargs)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=1e-3,
                                                        verbose=verbose)

    triplet_selector = TripletSelector(margin=torch.tensor(0.2), negative_selection_fn=select_semihard)
    loss_fns_train = [
        TripletLossWrapper_pretraining(triplet_selector, labels, average=True)
        # L1RegWrapper(model=model.mapping)
    ]

    triplet_selector_test = TripletSelector(margin=torch.tensor(0.2), negative_selection_fn=select_random)
    loss_fns_test = [
        TripletLossWrapper_pretraining(triplet_selector_test, test_labels, average=True)
        # TSNEWrapperMapNet(N=len(test_features), use_gpu=use_gpu)
    ]

    # train network until scheduler reduces learning rate to threshold value
    lr_threshold = 5e-5
    epoch = 1

    best_loss = float('inf')
    best_epoch = -1
    stopcriterion = False
    while not stopcriterion:
        # # compute beta for mapped features
        # if verbose:
        #     print('Compute beta for TSNE-Loss...')
        # mapped_features = torch.from_numpy(features) if epoch == 1 else \
        #     evaluate_model(model.mapping, features, batch_size=batch_size, use_gpu=use_gpu, verbose=verbose)
        # mapped_features /= mapped_features.norm(dim=1, keepdim=True)
        # loss_fns_train[1]._compute_beta(mapped_features)
        # if verbose:
        #     print('Done.')
        model.train()
        trainlosses = train(eval_fn, trainloader, optimizer, loss_fns_train, epoch, loss_fns_weights=(1,),
                            use_gpu=use_gpu, verbose=False, logger=logger, log_interval=100)
        model.eval()
        testlosses = test(eval_fn, testloader, loss_fns_test, epoch, loss_fns_weights=(1,),
                          lr_scheduler=lr_scheduler,
                          use_gpu=use_gpu, verbose=False, logger=logger)
        testloss = testlosses[-1]
        if testloss < best_loss:
            best_loss = testloss
            best_epoch = epoch
            if outpath is not None:
                torch.save({                # save best model
                    'state_dict': model.state_dict(),
                }, outpath)

        if log_dir is not None:         # save checkpoint
            torch.save({
                'epoch': epoch,
                'best_loss': best_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict()
            }, os.path.join(log_dir, 'checkpoint.pth.tar'))


        epoch += 1
        stopcriterion = (optimizer.param_groups[-1]['lr'] < lr_threshold)

    print('Finished training mapnet with best training loss: {:.4f} from epoch {}.'.format(best_loss, best_epoch))
    if outpath is not None:
        print('Saved model to {}.'.format(outpath))


def evaluate_model(model_fn, data, batch_size=100, use_gpu=True, verbose=False):
    loader_kwargs = {'num_workers': 4, 'drop_last': False} if use_gpu else {'drop_last': False}
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False,
                             **loader_kwargs)
    data_out = []
    for i, data in enumerate(dataloader):
        if verbose:
            print('{}/{}'.format(i+1, len(dataloader)))
        if isinstance(data, list) or isinstance(data, tuple):
            input = data[0].cuda() if use_gpu else data[0]
        else:
            input = data.cuda() if use_gpu else data
        output = model_fn(input)
        if isinstance(output, tuple):
            data_out.append([o.data.cpu() for o in output])
        else:
            data_out.append(output.data.cpu())
    if isinstance(data_out[0], list):
        output = []
        def mapfn(x, i):
            return x[i]
        for n in range(len(data_out[0])):
            func = partial(mapfn, i=n)
            output.append(torch.cat(map(func, data_out)))
        return output
    else:
        return torch.cat(data_out)


def init_identity_like_weights(model):
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            m.weight.data = torch.eye(m.weight.shape[0], m.weight.shape[1]).type_as(m.weight.data)
            # m.weight.data += torch.rand_like(m.weight.data) * 1e-3
            m.bias.data.fill_(0)
            # m.bias.data += torch.rand_like(m.bias.data) * 1e-3
    model.apply(init_weights)


if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    feature_file = 'features/Wikiart_Elgammal_test_512.hdf5'
    label_file = '../MapNetCode/pretraining/wikiart_datasets/info_elgammal_subset_test.hdf5'

    experiment_id = 'TEST'
    projection_dim = 2

    data = dd.io.load(feature_file)
    features = data['features'].copy()
    # normalize features
    features /= np.linalg.norm(features, axis=1, keepdims=True)

    feature_dim = features.shape[1]

    model = MapNet(feature_dim=feature_dim, output_dim=projection_dim)
    if use_gpu:
        model = model.cuda()


    # if True:#not os.path.isfile(weightfile_embedder):
    #     train_embedder(model.embedder, features, batch_size=100, verbose=False, outpath=weightfile_embedder)
    #
    # pretrained_dict = load_weights(weightfile_embedder, model.embedder.state_dict())
    # model.embedder.load_state_dict(pretrained_dict)
    #
    # projection = evaluate_model(model.embedder.forward, features, batch_size=2000, use_gpu=use_gpu, verbose=False)

    # test with ground truth labels
    weightfile_mapnet = './models/{}_mapnet.pth.tar'.format(experiment_id)
    log_dir = './.log/{}'.format(experiment_id)
    gt_labels = dd.io.load(label_file)['df']['artist_name']
    valid = gt_labels.dropna().index

    features = features[valid]
    labels = gt_labels[valid]
    label_to_int = {l: i for i, l in enumerate(set(labels))}
    labels = torch.tensor(map(lambda x: label_to_int[x], labels))
    train_mapnet(model, features, labels, outpath=weightfile_mapnet, log_dir=log_dir, verbose=False,
                 max_epochs=100)
