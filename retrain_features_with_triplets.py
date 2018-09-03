import os
import pickle
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import time
import sys
import numpy as np
sys.path.append('../SmallNets')
from wikiart_dataset import Wikiart
from models import mobilenet_v1, mobilenet_v2, squeezenet, vgg16_bn, inception_v3, alexnet, MetricNet
sys.path.append('../FullPipeline')
from aux import AverageMeter, TBPlotter, save_checkpoint, write_config
from triplet_dataset import BalancedBatchSamplerMulticlass
from triplets_loss import TripletLoss
from triplets_utils import SemihardNegativeTripletSelector, RandomNegativeTripletSelector, \
    HardestNegativeTripletSelector, KHardestNegativeTripletSelector, pdist
from itertools import combinations
import torch.backends.cudnn as cudnn
from collections import Counter


def create_trainset(train_file, label_file, im_path, train_transform, classes):
    # load the new labels
    with open(label_file, 'rb') as f:
        data = pickle.load(f)
    labels = data['labels']
    image_names = data['image_name']

    trainset = Wikiart(path_to_info_file=train_file, path_to_images=im_path,
                       classes=list(classes), transform=train_transform)
    assert np.all(image_names == trainset.df['image_id'].values), \
        'image names in user label do not match image names in train file'
    for c, l in zip(classes, labels.transpose()):
        trainset.df[c] = l
    # filter labels for samples which are not labeled at all
    trainset.df = trainset.df[np.any(labels != None, axis=1)]
    trainset.df.index = range(len(trainset))

    trainset.labels_to_ints = trainset._get_labels_to_ints()
    print('Created trainset of length {} from label file {}'.format(len(trainset), label_file))
    return trainset


def create_valset(val_file, im_path, val_transform, labels_to_ints_train):
    classes = labels_to_ints_train.keys()
    valset = Wikiart(path_to_info_file=val_file, path_to_images=im_path,
                      classes=classes, transform=val_transform)
    valid_samples = np.array([True] * len(valset))
    for c in classes:
        labels = labels_to_ints_train[c].keys()
        assert np.all(np.isin(labels, valset.labels_to_ints[c].keys())), \
            'some labels of {} are not in valset'.format(labels)        # check validity of labels
        valid_samples *= valset.df[c].isin(labels).values

    valset.df = valset.df[valid_samples]
    valset.df.index = range(len(valset))
    valset.labels_to_ints = labels_to_ints_train
    valset.classes = classes

    return valset


def GTEMulticlass(outputs, target):
    gtes = []
    dist_ap = []
    dist_an = []
    for op, tgt in zip(outputs, target):            # compute triplets class wise
        # filter unlabeled samples if there are any (have label -1)
        labeled = (tgt != -1).nonzero().view(len(tgt))
        op, tgt = op[labeled], tgt[labeled]

        # evaluate distance on all triplets
        labels = torch.unique(tgt.data.cpu()).type_as(tgt)
        for label in labels:
            positives = (tgt == label).nonzero()
            negatives = (tgt != label).nonzero()
            if len(positives) == 0 or len(negatives) == 0:
                continue
            else:
                positives = positives[:, 0]
                negatives = negatives[:, 0]
            ap_pairs = list(combinations(positives, 2))
            for a, p in ap_pairs:
                d_ap = torch.norm(op[a]-op[p], 2).repeat(len(negatives))
                d_an = torch.norm(op[a].repeat(len(negatives), 1)-op[negatives], 2, dim=1)
                gtes.append(torch.sum(d_an < d_ap).type_as(op) / len(negatives))
                dist_ap.append(torch.mean(d_ap))
                dist_an.append(torch.mean(d_an))

    return torch.mean(torch.stack(gtes)), torch.mean(torch.stack(dist_ap)), torch.mean(torch.stack(dist_an))


# train_file = '../wikiart/datasets/info_artist_49_multilabel_test.hdf5'
# test_file = '../wikiart/datasets/info_artist_49_multilabel_val.hdf5'
# stat_file = '../wikiart/datasets/info_artist_49_multilabel_train_mean_std.pkl'
# model='mobilenet_v2'
# classes=('style',)
# label_file='largest_4_styles_labelfile.pkl'
# im_path='/export/home/kschwarz/Documents/Data/Wikiart_artist49_images'
# chkpt=None
# weight_file='../SmallNets/runs/multilabel/06-24-15-14_OctopusNet_model_best.pth.tar'
# triplet_selector='semihard'
# margin=0.2
# labels_per_class=4
# samples_per_label=4
# use_gpu=True
# device=0
# epochs=100
# batch_size=32
# lr=1e-4
# momentum=0.9
# log_interval=10
# log_dir='runs/trash'
# exp_name=None
# seed=123

def train_multiclass(train_file, test_file, stat_file,
                     model='mobilenet_v2',
                     classes=('artist_name', 'genre', 'style', 'technique', 'century'),
                     label_file='_user_labels.pkl',
                     im_path='/export/home/kschwarz/Documents/Data/Wikiart_artist49_images',
                     chkpt=None, weight_file=None,
                     triplet_selector='semihard', margin=0.2,
                     labels_per_class=4, samples_per_label=4,
                     use_gpu=True, device=0,
                     epochs=100, batch_size=32, lr=1e-4, momentum=0.9,
                     log_interval=10, log_dir='runs',
                     exp_name=None, seed=123):
    argvars = locals().copy()
    torch.manual_seed(seed)

    # LOAD DATASET
    with open(stat_file, 'r') as f:
        data = pickle.load(f)
        mean, std = data['mean'], data['std']
        mean = [float(m) for m in mean]
        std = [float(s) for s in std]
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                normalize,
    ])

    if model.lower() == 'inception_v3':            # change input size to 299
        train_transform.transforms[0].size = (299, 299)
        val_transform.transforms[0].size = (299, 299)
    trainset = create_trainset(train_file, label_file, im_path, train_transform, classes)
    for c in classes:
        if len(trainset.labels_to_ints[c]) < labels_per_class:
            print('less labels in class {} than labels_per_class, use all available labels ({})'
                  .format(c, len(trainset.labels_to_ints[c])))
    valset = create_valset(test_file, im_path, val_transform, trainset.labels_to_ints)
    # PARAMETERS
    use_cuda = use_gpu and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)

    if model.lower() not in ['squeezenet', 'mobilenet_v1', 'mobilenet_v2', 'vgg16_bn', 'inception_v3', 'alexnet']:
        assert False, 'Unknown model {}\n\t+ Choose from: ' \
                      '[sqeezenet, mobilenet_v1, mobilenet_v2, vgg16_bn, inception_v3, alexnet].'.format(model)
    elif model.lower() == 'mobilenet_v1':
        bodynet = mobilenet_v1(pretrained=weight_file is None)
    elif model.lower() == 'mobilenet_v2':
        bodynet = mobilenet_v2(pretrained=weight_file is None)
    elif model.lower() == 'vgg16_bn':
        bodynet = vgg16_bn(pretrained=weight_file is None)
    elif model.lower() == 'inception_v3':
        bodynet = inception_v3(pretrained=weight_file is None)
    elif model.lower() == 'alexnet':
        bodynet = alexnet(pretrained=weight_file is None)
    else:       # squeezenet
        bodynet = squeezenet(pretrained=weight_file is None)

    # Load weights for the body network
    if weight_file is not None:
        print("=> loading weights from '{}'".format(weight_file))
        pretrained_dict = torch.load(weight_file, map_location=lambda storage, loc: storage)['state_dict']
        state_dict = bodynet.state_dict()
        pretrained_dict = {k.replace('bodynet.', ''): v for k, v in pretrained_dict.items()         # in case of multilabel weight file
                           if (k.replace('bodynet.', '') in state_dict.keys() and v.shape == state_dict[k.replace('bodynet.', '')].shape)}  # number of classes might have changed
        # check which weights will be transferred
        if not pretrained_dict == state_dict:  # some changes were made
            for k in set(state_dict.keys() + pretrained_dict.keys()):
                if k in state_dict.keys() and k not in pretrained_dict.keys():
                    print('\tWeights for "{}" were not found in weight file.'.format(k))
                elif k in pretrained_dict.keys() and k not in state_dict.keys():
                    print('\tWeights for "{}" were are not part of the used model.'.format(k))
                elif state_dict[k].shape != pretrained_dict[k].shape:
                    print('\tShapes of "{}" are different in model ({}) and weight file ({}).'.
                          format(k, state_dict[k].shape, pretrained_dict[k].shape))
                else:  # everything is good
                    pass

        state_dict.update(pretrained_dict)
        bodynet.load_state_dict(state_dict)

    net = MetricNet(bodynet, len(classes))

    n_parameters = sum([p.data.nelement() for p in net.parameters() if p.requires_grad])
    if use_cuda:
        net = net.cuda()
    print('Using {}\n\t+ Number of params: {}'.format(str(net).split('(', 1)[0], n_parameters))

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # tensorboard summary writer
    timestamp = time.strftime('%m-%d-%H-%M')
    expname = timestamp + '_' + str(net).split('(', 1)[0]
    if exp_name is not None:
        expname = expname + '_' + exp_name
    log = TBPlotter(os.path.join(log_dir, 'tensorboard', expname))
    log.print_logdir()

    # allow auto-tuner to find best algorithm for the hardware
    cudnn.benchmark = True

    with open(label_file, 'rb') as f:
        labels = pickle.load(f)['labels']
        n_labeled = '\t'.join([str(Counter(l).items()) for l in labels.transpose()])

    write_config(argvars, os.path.join(log_dir, expname), extras={'n_labeled': n_labeled})


    # ININTIALIZE TRAINING
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, threshold=1e-1, verbose=True)

    if triplet_selector.lower() not in ['random', 'semihard', 'hardest', 'mixed', 'khardest']:
        assert False, 'Unknown option {} for triplet selector. Choose from "random", "semihard", "hardest" or "mixed"' \
                      '.'.format(triplet_selector)
    elif triplet_selector.lower() == 'random':
        criterion = TripletLoss(margin=margin,
                                triplet_selector=RandomNegativeTripletSelector(margin, cpu=not use_cuda))
    elif triplet_selector.lower() == 'semihard' or triplet_selector.lower() == 'mixed':
        criterion = TripletLoss(margin=margin,
                                triplet_selector=SemihardNegativeTripletSelector(margin, cpu=not use_cuda))
    elif triplet_selector.lower() == 'khardest':
        criterion = TripletLoss(margin=margin,
                                triplet_selector=KHardestNegativeTripletSelector(margin, k=3, cpu=not use_cuda))
    else:
        criterion = TripletLoss(margin=margin,
                                triplet_selector=HardestNegativeTripletSelector(margin, cpu=not use_cuda))
    if use_cuda:
        criterion = criterion.cuda()

    kwargs = {'num_workers': 4} if use_cuda else {}
    multilabel_train = np.stack([trainset.df[c].values for c in classes]).transpose()
    train_batch_sampler = BalancedBatchSamplerMulticlass(multilabel_train, n_label=labels_per_class,
                                                         n_per_label=samples_per_label, ignore_label=None)
    trainloader = DataLoader(trainset, batch_sampler=train_batch_sampler, **kwargs)
    multilabel_val = np.stack([valset.df[c].values for c in classes]).transpose()
    val_batch_sampler = BalancedBatchSamplerMulticlass(multilabel_val, n_label=labels_per_class,
                                                       n_per_label=samples_per_label, ignore_label=None)
    valloader = DataLoader(valset, batch_sampler=val_batch_sampler, **kwargs)

    # optionally resume from a checkpoint
    start_epoch = 1
    if chkpt is not None:
        if os.path.isfile(chkpt):
            print("=> loading checkpoint '{}'".format(chkpt))
            checkpoint = torch.load(chkpt, map_location=lambda storage, loc: storage)
            start_epoch = checkpoint['epoch']
            best_acc_score = checkpoint['best_acc_score']
            best_acc = checkpoint['acc']
            net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(chkpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(chkpt))

    def train(epoch):
        losses = AverageMeter()
        gtes = AverageMeter()
        non_zero_triplets = AverageMeter()
        distances_ap = AverageMeter()
        distances_an = AverageMeter()

        # switch to train mode
        net.train()
        for batch_idx, (data, target) in enumerate(trainloader):
            target = torch.stack(target)
            if use_cuda:
                data, target = Variable(data.cuda()), [Variable(t.cuda()) for t in target]
            else:
                data, target = Variable(data), [Variable(t) for t in target]

            # compute output
            outputs = net(data)

            # normalize features
            for i in range(len(classes)):
                outputs[i] = torch.nn.functional.normalize(outputs[i], p=2, dim=1)

            loss = Variable(torch.Tensor([0]), requires_grad=True).type_as(data[0])
            n_triplets = 0
            for op, tgt in zip(outputs, target):
                # filter unlabeled samples if there are any (have label -1)
                labeled = (tgt != -1).nonzero().view(-1)
                op, tgt = op[labeled], tgt[labeled]

                l, nt = criterion(op, tgt)
                loss += l
                n_triplets += nt

            non_zero_triplets.update(n_triplets, target[0].size(0))
            # measure GTE and record loss
            gte, dist_ap, dist_an = GTEMulticlass(outputs, target)           # do not compute ap pairs for concealed classes
            gtes.update(gte.data, target[0].size(0))
            distances_ap.update(dist_ap.data, target[0].size(0))
            distances_an.update(dist_an.data, target[0].size(0))
            losses.update(loss.data[0], target[0].size(0))

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{}]\t'
                      'Loss: {:.4f} ({:.4f})\t'
                      'GTE: {:.2f}% ({:.2f}%)\t'
                      'Non-zero Triplets: {:d} ({:d})'.format(
                    epoch, batch_idx * len(target[0]), len(trainloader) * len(target[0]),
                    float(losses.val), float(losses.avg),
                    float(gtes.val) * 100., float(gtes.avg) * 100.,
                    int(non_zero_triplets.val), int(non_zero_triplets.avg)))

        # log avg values to somewhere
        log.write('loss', float(losses.avg), epoch, test=False)
        log.write('gte', float(gtes.avg), epoch, test=False)
        log.write('non-zero trplts', int(non_zero_triplets.avg), epoch, test=False)
        log.write('dist_ap', float(distances_ap.avg), epoch, test=False)
        log.write('dist_an', float(distances_an.avg), epoch, test=False)

    def test(epoch):
        losses = AverageMeter()
        gtes = AverageMeter()
        non_zero_triplets = AverageMeter()
        distances_ap = AverageMeter()
        distances_an = AverageMeter()

        # switch to evaluation mode
        net.eval()
        for batch_idx, (data, target) in enumerate(valloader):
            target = torch.stack(target)
            if use_cuda:
                data, target = Variable(data.cuda()), [Variable(t.cuda()) for t in target]
            else:
                data, target = Variable(data), [Variable(t) for t in target]
            # compute output
            outputs = net(data)

            # normalize features
            for i in range(len(classes)):
                outputs[i] = torch.nn.functional.normalize(outputs[i], p=2, dim=1)

            loss = Variable(torch.Tensor([0]), requires_grad=True).type_as(data[0])
            n_triplets = 0
            for op, tgt in zip(outputs, target):
                # filter unlabeled samples if there are any (have label -1)
                labeled = (tgt != -1).nonzero().view(-1)
                op, tgt = op[labeled], tgt[labeled]

                l, nt = criterion(op, tgt)
                loss += l
                n_triplets += nt

            non_zero_triplets.update(n_triplets, target[0].size(0))
            # measure GTE and record loss
            gte, dist_ap, dist_an = GTEMulticlass(outputs, target)
            gtes.update(gte.data.cpu(), target[0].size(0))
            distances_ap.update(dist_ap.data.cpu(), target[0].size(0))
            distances_an.update(dist_an.data.cpu(), target[0].size(0))
            losses.update(loss.data[0].cpu(), target[0].size(0))

        print('\nVal set: Average loss: {:.4f} Average GTE {:.2f}%, '
              'Average non-zero triplets: {:d} LR: {:.6f}'.format(float(losses.avg), float(gtes.avg) * 100.,
                                                       int(non_zero_triplets.avg),
                                                                  optimizer.param_groups[-1]['lr']))
        log.write('loss', float(losses.avg), epoch, test=True)
        log.write('gte', float(gtes.avg), epoch, test=True)
        log.write('non-zero trplts', int(non_zero_triplets.avg), epoch, test=True)
        log.write('dist_ap', float(distances_ap.avg), epoch, test=True)
        log.write('dist_an', float(distances_an.avg), epoch, test=True)
        return losses.avg, 1 - gtes.avg

    if start_epoch == 1:         # compute baseline:
        _, best_acc = test(epoch=0)
    else:       # checkpoint was loaded
        best_acc = best_acc

    for epoch in range(start_epoch, epochs + 1):
        if triplet_selector.lower() == 'mixed' and epoch == 26:
            criterion.triplet_selector = HardestNegativeTripletSelector(margin, cpu=not use_cuda)
            print('Changed negative selection from semihard to hardest.')
        # train for one epoch
        train(epoch)
        # evaluate on validation set
        val_loss, val_acc = test(epoch)
        scheduler.step(val_loss)

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'best_acc': best_acc,
        }, is_best, expname, directory=log_dir)

        if optimizer.param_groups[-1]['lr'] < 1e-5:
            print('Learning rate reached minimum threshold. End training.')
            break

    # report best values
    best = torch.load(os.path.join(log_dir, expname + '_model_best.pth.tar'), map_location=lambda storage, loc: storage)
    print('Finished training after epoch {}:\n\tbest acc score: {}'
          .format(best['epoch'], best['acc']))
    print('Best model mean accuracy: {}'.format(best_acc))

#
# import deepdish as dd
# def generate_label_file():
#     classes = ['style']
#     labels = {'style': ['impressionism', 'realism', 'post-impressionism', 'surrealism']}
#     info_file = '../wikiart/datasets/info_artist_49_multilabel_test.hdf5'
#     name = 'largest_4_styles_labelfile.pkl'
#
#     df = dd.io.load(info_file)['df']
#     image_names = df['image_id'].values
#     targets = []
#     for c in classes:
#         label = labels[c]
#         mask = df[c].isin(label).values
#         df[c][np.logical_not(mask)] = None
#         targets.append(df[c].values)
#
#     targets = np.stack(targets).transpose()
#     with open(name, 'wb') as f:
#         pickle.dump({'image_name': image_names, 'labels': targets}, f)
