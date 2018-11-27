import os
import sys
import argparse
import pickle
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import time
import shutil
import torch.backends.cudnn as cudnn

if not os.getcwd().endswith('MapNetCode/pretraining/'):
    os.chdir('/export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/pretraining/')
sys.path.append('./')

from dataset import Wikiart
from model import mobilenet_v2, vgg16_bn, narrownet, OctopusNet

sys.path.append('/export/home/kschwarz/Documents/Masters/FullPipeline')
import matplotlib as mpl
mpl.use('TkAgg')
from aux import AverageMeter, TBPlotter, save_checkpoint, write_config

sys.path.append('/export/home/kschwarz/Documents/Data/Geometric_Shapes')
from shape_dataset import ShapeDataset


parser = argparse.ArgumentParser(description='Multitask training on multiple labels of Wikiarts dataset.')
# wikiart dataset
parser.add_argument('--val_file', default='wikiart_datasets/info_artist_val.hdf5',
                    type=str, help='Path to hdf5 file containing dataframe of validation set.')
parser.add_argument('--train_file', default='wikiart_datasets/info_artist_train.hdf5',
                    type=str, help='Path to hdf5 file containing dataframe of train set.')
parser.add_argument('--im_path', default='/export/home/kschwarz/Documents/Data/Wikiart_artist49_images', type=str,
                    help='Path to Wikiart images')
parser.add_argument('--stat_file', default='wikiart_datasets/info_artist_49_multilabel_train_mean_std.pkl',
                    type=str, help='.pkl file containing artist dataset mean and std.')
parser.add_argument('--shape_dataset', default=False, action='store_true', help='Use artificial shape dataset')
parser.add_argument('--office_dataset', default=False, action='store_true', help='Use office dataset')
parser.add_argument('--bam_dataset', default=False, action='store_true', help='Use bam dataset')

parser.add_argument('--task_selection', default=None, type=str, help='Task (categories) on which to train')


# model
parser.add_argument('--model', type=str, help='Choose from: mobilenet_v2, vgg16_bn')
parser.add_argument('--not_narrow', default=False, action='store_true', help='Do not narrow feature down to 128 dim.')
parser.add_argument('--feature_dim', type=int, default=128, help='Dimensionality of features before classification.')
parser.add_argument('--chkpt', default=None, type=str, help='Checkpoint file to resume from.')

# training parameters
parser.add_argument('--use_gpu', default=True, type=bool, help='Use gpu or not.')
parser.add_argument('--device', default=0, type=int, help='Number of gpu device to use.')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train for.')
parser.add_argument('--batch_size', default=100, type=int, help='Batch size.')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate.')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum used in SGD.')

# logging
parser.add_argument('--log_interval', default=10, type=int, help='Output frequency.')
parser.add_argument('--log_dir', default='runs/', type=str, help='Directory to save logs.')
parser.add_argument('--exp_name', default=None, type=str, help='Name of experiment.')

parser.add_argument('--seed', default=123, type=int, help='Random state.')

# sys.argv = []
args = parser.parse_args()
# args.val_file = 'wikiart_datasets/info_elgammal_subset_val.hdf5'
# args.train_file = 'wikiart_datasets/info_elgammal_subset_train.hdf5'
# args.im_path = '/export/home/asanakoy/workspace/wikiart/images'
# args.stat_file = 'wikiart_datasets/info_elgammal_subset_train_mean_std.pkl'
# args.task_selection = 'genre,artist_name'
# args.model = 'vgg16_bn'
# args.device = 1
# args.lr = 1e-3
# args.exp_name = 'TEST'


def main():
    args.task_selection = args.task_selection.split(',')

    torch.manual_seed(args.seed)

    # LOAD DATASET
    stat_file = args.stat_file
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
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
    ])

    if not args.shape_dataset:
        if args.task_selection is not None:
            classes = args.task_selection
        elif args.office_dataset:
            classes = ['style', 'genre']
        elif args.bam_dataset:
            classes = ['content', 'emotion', 'media']
        else:
            classes = ['artist_name', 'genre', 'style', 'technique', 'century']
        valset = Wikiart(path_to_info_file=args.val_file, path_to_images=args.im_path,
                         classes=classes, transform=val_transform)
        trainset = Wikiart(path_to_info_file=args.train_file, path_to_images=args.im_path,
                           classes=classes, transform=train_transform)
    else:
        if args.task_selection is not None:
            classes = args.task_selection
        else:
            classes = ['shape', 'n_shapes', 'color_shape', 'color_background']
        valset = ShapeDataset(root_dir='/export/home/kschwarz/Documents/Data/Geometric_Shapes', split='val',
                              classes=classes, transform=val_transform)
        trainset = ShapeDataset(root_dir='/export/home/kschwarz/Documents/Data/Geometric_Shapes', split='train',
                                classes=classes, transform=train_transform)

    if not trainset.labels_to_ints == valset.labels_to_ints:
        print('validation set and training set int labels do not match. Use int conversion of trainset')
        print(trainset.labels_to_ints, valset.labels_to_ints)
        valset.labels_to_ints = trainset.labels_to_ints.copy()

    num_labels = [len(trainset.labels_to_ints[c]) for c in classes]

    # PARAMETERS
    use_cuda = args.use_gpu and torch.cuda.is_available()
    device_nb = args.device
    if use_cuda:
        torch.cuda.set_device(device_nb)
        torch.cuda.manual_seed_all(args.seed)

    # INITIALIZE NETWORK
    if args.model.lower() not in ['mobilenet_v2', 'vgg16_bn']:
        raise NotImplementedError('Unknown Model {}\n\t+ Choose from: [mobilenet_v2, vgg16_bn].'
                                  .format(args.model))
    elif args.model.lower() == 'mobilenet_v2':
        featurenet = mobilenet_v2(pretrained=True)
    elif args.model.lower() == 'vgg16_bn':
        featurenet = vgg16_bn(pretrained=True)
    if args.not_narrow:
        bodynet = featurenet
    else:
        bodynet = narrownet(featurenet, dim_feature_out=args.feature_dim)
    net = OctopusNet(bodynet, n_labels=num_labels)
    n_parameters = sum([p.data.nelement() for p in net.parameters() if p.requires_grad])
    if use_cuda:
        net = net.cuda()
    print('Using {}\n\t+ Number of params: {}'.format(str(bodynet).split('(')[0], n_parameters))

    # LOG/SAVE OPTIONS
    log_interval = args.log_interval
    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # tensorboard summary writerR
    timestamp = time.strftime('%m-%d-%H-%M')
    if args.shape_dataset:
        expname = timestamp + '_ShapeDataset_' + str(bodynet).split('(')[0]
    else:
        expname = timestamp + '_' + str(bodynet).split('(')[0]
    if args.exp_name is not None:
        expname = expname + '_' + args.exp_name
    log = TBPlotter(os.path.join(log_dir, 'tensorboard', expname))
    log.print_logdir()

    # allow auto-tuner to find best algorithm for the hardware
    cudnn.benchmark = True

    write_config(args, os.path.join(log_dir, expname))

    # ININTIALIZE TRAINING
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, threshold=1e-1, verbose=True)
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        criterion = criterion.cuda()

    kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, **kwargs)

    # optionally resume from a checkpoint
    start_epoch = 1
    if args.chkpt is not None:
        if os.path.isfile(args.chkpt):
            print("=> loading checkpoint '{}'".format(args.chkpt))
            checkpoint = torch.load(args.chkpt, map_location=lambda storage, loc: storage)
            start_epoch = checkpoint['epoch']
            best_acc_score = checkpoint['best_acc_score']
            best_acc = checkpoint['acc']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.chkpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.chkpt))

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
                # in case of None labels
                mask = t != -1
                if mask.sum() == 0:
                    continue
                o, t, p = o[mask], t[mask], p[mask]
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
                # in case of None labels
                mask = t != -1
                if mask.sum() == 0:
                    continue
                o, t, p = o[mask], t[mask], p[mask]
                loss += criterion(o, t)
                # measure class accuracy and record loss
                class_acc[i].update((torch.sum(p == t).type(torch.FloatTensor) / t.size(0)).data)
            accs.update(torch.mean(torch.stack([class_acc[i].val for i in range(len(classes))])), target[0].size(0))
            losses.update(loss.data, target[0].size(0))

        score = accs.avg - torch.std(torch.stack([class_acc[i].avg for i in range(
            len(classes))])) / accs.avg  # compute mean - std/mean as measure for accuracy
        print('\nVal set: Average loss: {:.4f} Average acc {:.2f}% Acc score {:.2f} LR: {:.6f}'
              .format(float(losses.avg), float(accs.avg) * 100., float(score), optimizer.param_groups[-1]['lr']))
        print('\t' + '\n\t'.join(['{}: {:.2f}%'.format(classes[i], float(class_acc[i].avg) * 100.)
                                  for i in range(len(classes))]))
        log.write('loss', float(losses.avg), epoch, test=True)
        log.write('acc', float(accs.avg), epoch, test=True)
        for i in range(len(classes)):
            log.write('class_acc', float(class_acc[i].avg), epoch, test=True)
        return losses.avg.cpu().numpy(), float(score), float(accs.avg), [float(class_acc[i].avg) for i in
                                                                         range(len(classes))]

    if start_epoch == 1:  # compute baseline:
        _, best_acc_score, best_acc, _ = test(epoch=0)
    else:  # checkpoint was loaded
        best_acc_score = best_acc_score
        best_acc = best_acc

    for epoch in range(start_epoch, args.epochs + 1):
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
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
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
    try:
        best = torch.load(os.path.join(log_dir, expname + '_model_best.pth.tar'), map_location=lambda storage, loc: storage)
    except IOError:         # could be only one task
        best = torch.load(os.path.join(log_dir, expname + '_model_best_mean_acc.pth.tar'), map_location=lambda storage, loc: storage)
    print('Finished training after epoch {}:\n\tbest acc score: {}\n\tacc: {}\n\t class acc: {}'
          .format(best['epoch'], best['best_acc_score'], best['acc'], best['class_acc']))
    print('Best model mean accuracy: {}'.format(best_acc))

    try:
        shutil.copyfile(os.path.join(log_dir, expname + '_model_best.pth.tar'),
                        os.path.join('models', expname + '_model_best.pth.tar'))
    except IOError:  # could be only one task
        shutil.copyfile(os.path.join(log_dir, expname + '_model_best_mean_acc.pth.tar'),
                        os.path.join('models', expname + '_model_best.pth.tar'))

if __name__ == '__main__':
    main()