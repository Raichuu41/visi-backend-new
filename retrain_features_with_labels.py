import os
import pickle
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import time
import shutil
import sys
import numpy as np
sys.path.append('../SmallNets')
from wikiart_dataset import Wikiart
from models import mobilenet_v1, mobilenet_v2, squeezenet, vgg16_bn, inception_v3, alexnet, OctopusNet
sys.path.append('../FullPipeline')
from aux import AverageMeter, TBPlotter, save_checkpoint, write_config
import torch.backends.cudnn as cudnn



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


def train_multiclass(train_file, test_file, stat_file,
                     model='mobilenet_v2',
                     classes=('artist_name', 'genre', 'style', 'technique', 'century'),
                     im_path='/export/home/kschwarz/Documents/Data/Wikiart_artist49_images',
                     label_file='_user_labels.pkl',
                     chkpt=None, weight_file=None,
                     use_gpu=True, device=0,
                     epochs=100, batch_size=32, lr=1e-4, momentum=0.9,
                     log_interval=10, log_dir='runs',
                     exp_name=None, seed=123):
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
    valset = create_valset(test_file, im_path, val_transform, trainset.labels_to_ints)
    num_labels = [len(trainset.labels_to_ints[c]) for c in classes]

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

    net = OctopusNet(bodynet, n_labels=num_labels)

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

    write_config(locals(), os.path.join(log_dir, expname))


    # ININTIALIZE TRAINING
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, threshold=1e-1, verbose=True)
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        criterion = criterion.cuda()

    kwargs = {'num_workers': 4} if use_cuda else {}
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, **kwargs)

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
        accs = AverageMeter()
        class_acc = [AverageMeter() for i in range(len(classes))]

        # switch to train mode
        net.train()
        for batch_idx, (data, target) in enumerate(trainloader):
            if use_cuda:
                data, target = Variable(data.cuda()), [Variable(t.cuda()) for t in target]
            else:
                data, target = Variable(data), [Variable(t) for t in target]

            # compute output
            outputs = net(data)
            preds = [torch.max(outputs[i], 1)[1] for i in range(len(classes))]

            loss = Variable(torch.Tensor([0])).type_as(data[0])
            for i, o, t, p in zip(range(len(classes)), outputs, target, preds):
                # filter unlabeled samples if there are any (have label -1)
                labeled = (t != -1).nonzero().view(len(t))
                o, t, p = o[labeled], t[labeled], p[labeled]
                loss += criterion(o, t)
                # measure class accuracy and record loss
                class_acc[i].update((torch.sum(p == t).type(torch.FloatTensor) / t.size(0)).data)
            accs.update(torch.mean(torch.stack([class_acc[i].val for i in range(len(classes))])), target[0].size(0))
            losses.update(loss.data, target[0].size(0))

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{}]\t'
                      'Loss: {:.4f} ({:.4f})\t'
                      'Acc: {:.2f}% ({:.2f}%)'.format(
                    epoch, batch_idx * len(target), len(trainloader.dataset),
                    float(losses.val), float(losses.avg),
                    float(accs.val) * 100., float(accs.avg) * 100.))
                print('\t' + '\n\t'.join(['{}: {:.2f}%'.format(classes[i], float(class_acc[i].val) * 100.)
                                          for i in range(len(classes))]))

        # log avg values to somewhere
        log.write('loss', float(losses.avg), epoch, test=False)
        log.write('acc', float(accs.avg), epoch, test=False)
        for i in range(len(classes)):
            log.write('class_acc', float(class_acc[i].avg), epoch, test=False)

    def test(epoch):
        losses = AverageMeter()
        accs = AverageMeter()
        class_acc = [AverageMeter() for i in range(len(classes))]

        # switch to evaluation mode
        net.eval()
        for batch_idx, (data, target) in enumerate(valloader):
            if use_cuda:
                data, target = Variable(data.cuda()), [Variable(t.cuda()) for t in target]
            else:
                data, target = Variable(data), [Variable(t) for t in target]

            # compute output
            outputs = net(data)
            preds = [torch.max(outputs[i], 1)[1] for i in range(len(classes))]

            loss = Variable(torch.Tensor([0])).type_as(data[0])
            for i, o, t, p in zip(range(len(classes)), outputs, target, preds):
                labeled = (t != -1).nonzero().view(len(t))
                loss += criterion(o[labeled], t[labeled])
                # measure class accuracy and record loss
                class_acc[i].update((torch.sum(p[labeled] == t[labeled]).type(torch.FloatTensor) / t[labeled].size(0)).data)
            accs.update(torch.mean(torch.stack([class_acc[i].val for i in range(len(classes))])), target[0].size(0))
            losses.update(loss.data, target[0].size(0))

        score = accs.avg - torch.std(torch.stack([class_acc[i].avg for i in range(len(classes))])) / accs.avg        # compute mean - std/mean as measure for accuracy
        print('\nVal set: Average loss: {:.4f} Average acc {:.2f}% Acc score {:.2f} LR: {:.6f}'
              .format(float(losses.avg), float(accs.avg) * 100., float(score), optimizer.param_groups[-1]['lr']))
        print('\t' + '\n\t'.join(['{}: {:.2f}%'.format(classes[i], float(class_acc[i].avg) * 100.)
                                  for i in range(len(classes))]))
        log.write('loss', float(losses.avg), epoch, test=True)
        log.write('acc', float(accs.avg), epoch, test=True)
        for i in range(len(classes)):
            log.write('class_acc', float(class_acc[i].avg), epoch, test=True)
        return losses.avg.cpu().numpy(), float(score), float(accs.avg), [float(class_acc[i].avg) for i in range(len(classes))]

    if start_epoch == 1:  # compute baseline:
        _, best_acc_score, best_acc, _ = test(epoch=0)
    else:  # checkpoint was loaded
        best_acc_score = best_acc_score
        best_acc = best_acc

    for epoch in range(start_epoch, epochs + 1):
        # train for one epoch
        train(epoch)
        # evaluate on validation set
        val_loss, val_acc_score, val_acc, val_class_accs = test(epoch)
        scheduler.step(val_loss)

        # remember best acc and save checkpoint
        is_best = val_acc_score > best_acc_score
        best_acc_score = max(val_acc_score, best_acc_score)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'best_acc_score': best_acc_score,
            'acc': val_acc,
            'class_acc': {c: a for c, a in zip(classes, val_class_accs)}
        }, is_best, expname, directory=log_dir)

        if val_acc > best_acc:
            shutil.copyfile(os.path.join(log_dir, expname + '_checkpoint.pth.tar'),
                            os.path.join(log_dir, expname + '_model_best_mean_acc.pth.tar'))
        best_acc = max(val_acc, best_acc)

        if optimizer.param_groups[-1]['lr'] < 1e-5:
            print('Learning rate reached minimum threshold. End training.')
            break

    # report best values
    best = torch.load(os.path.join(log_dir, expname + '_model_best.pth.tar'), map_location=lambda storage, loc: storage)
    print('Finished training after epoch {}:\n\tbest acc score: {}\n\tacc: {}\n\t class acc: {}'
          .format(best['epoch'], best['best_acc_score'], best['acc'], best['class_acc']))
    print('Best model mean accuracy: {}'.format(best_acc))


