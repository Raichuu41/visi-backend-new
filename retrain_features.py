import os
import argparse
import pickle
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import time
import shutil
import sys
import torch.backends.cudnn as cudnn

sys.path.append('../SmallNets')
from wikiart_dataset import Wikiart
from models import mobilenet_v1, mobilenet_v2, squeezenet, vgg16_bn, inception_v3, remove_fc
from triplet_dataset import BalancedBatchSampler, BalancedBatchSampler_withNones, BalancedBatchSampler_withNonesandNoise
from triplets_loss import OnlineTripletLoss, TripletLoss
from triplets_utils import SemihardNegativeTripletSelector, RandomNegativeTripletSelector, \
    HardestNegativeTripletSelector, KHardestNegativeTripletSelector, pdist
from itertools import combinations
from collections import Counter
from mine_triplets import get_labels
sys.path.append('../FullPipeline')
from aux import AverageMeter, TBPlotter, save_checkpoint, write_config



parser = argparse.ArgumentParser(description='Train small network on Wikiarts artist dataset')
# wikiart artist dataset
parser.add_argument('--test_file', default='/export/home/kschwarz/Documents/Masters/wikiart/datasets/info_artist_test.hdf5',
                    type=str, help='Path to "info_artist_test.hdf5"')
parser.add_argument('--train_file', default='/export/home/kschwarz/Documents/Masters/wikiart/datasets/info_artist_train.hdf5',
                    type=str, help='Path to "info_artist_train.hdf5"')
parser.add_argument('--im_path', default='/export/home/asanakoy/workspace/wikiart/images', type=str,
                    help='Path to Wikiart images')
parser.add_argument('--stat_file', default='/export/home/kschwarz/Documents/Masters/wikiart/datasets/artist_train_mean_std.pkl',
                    type=str, help='.pkl file containing artist dataset mean and std.')
parser.add_argument('--class_label', type=str, help='Class in wikiart used for labels.')

# model
parser.add_argument('--model', type=str, help='Choose from: squeezenet, mobilenet_v1, mobilenet_v2, '
                                              'vgg16_bn, inception_v3')
parser.add_argument('--weight_file', type=str, help='File with pretrained model weights.')
parser.add_argument('--chkpt', default=None, type=str, help='Checkpoint file to resume from.')

# triplet options
parser.add_argument('--margin', default=.2, type=float, help='Margin in triplet loss.')
parser.add_argument('--triplet_selector', default='semihard', type=str, help='Method to choose negatives. Choose from'
                                                                             '"random", "semihard", "hardest", "khardest" or'
                                                                             '"mixed)" (combines semihard and hardest)')
parser.add_argument('--conf_thresh', default=None, type=float, help='Minimum confidence of loaded labels.')
parser.add_argument('--label_file', default='_svm_labels.pkl', help='Label file.')
parser.add_argument('--n_labels_ap', type=int, help='Number of classes to use for anchor/positives.')
parser.add_argument('--conceal_smallest', default=False, action='store_true', help='Keep the n_labels_ap largest classes in validation set.')
# parser.add_argument('--class_noise', default=0, type=float, help='Fraction of hidden labels per class.')

# training parameters
parser.add_argument('--use_gpu', default=True, type=bool, help='Use gpu or not.')
parser.add_argument('--device', default=0, type=int, help='Number of gpu device to use.')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train for.')
parser.add_argument('--classes_per_batch', default=5, type=int, help='Number of different classes per batch.')
parser.add_argument('--samples_per_class', default=5, type=int, help='Number of samples per class.')
parser.add_argument('--pure_negatives_per_batch', default=10, type=int, help='Number of pure negatives per batch.')
parser.add_argument('--n_train_samples_per_class', default=None, type=int, help='Number of total training images per class.')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate.')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum used in SGD.')

# logging
parser.add_argument('--log_interval', default=10, type=int, help='Output frequency.')
parser.add_argument('--log_dir', default='runs/', type=str, help='Directory to save logs.')
parser.add_argument('--exp_name', default=None, type=str, help='Name of experiment.')

parser.add_argument('--seed', default=123, type=int, help='Random state.')

# sys.argv = []
args = parser.parse_args()

if args.n_train_samples_per_class is not None:
    assert args.samples_per_class <= args.n_train_samples_per_class, 'Number of distinct samples per class is larger ' \
                                                                     'than available train samples per class.'


# args.model = 'mobilenet_v2'
# args.device = 2
# args.exp_name = 'style_largest4'
# args.class_label = 'style'
# args.train_file = '../wikiart/datasets/info_artist_49_style_train_small.hdf5'
# args.test_file = '../wikiart/datasets/info_artist_49_style_test.hdf5'
# args.stat_file = '../wikiart/datasets/info_artist_49_style_train_small_mean_std.pkl'
# args.weight_file = '../SmallNets/runs/06-02-23-27_MobileNetV2_artist_49_ft_model_best.pth.tar'
# args.label_file = 'N100_p_100_r_100_labels_style_4largest_localSVM.pkl'
# args.triplet_selector = 'khardest'
# args.conf_thresh = 0.5
# args.epochs = 100
# args.lr = 1e-3
# args.n_labels_ap = 4
# args.conceal_smallest = True
# args.classes_per_batch = 4
# args.samples_per_class = 6
# args.log_dir = 'runs/trash/'


defaults = {}
for key in vars(args):
    defaults[key] = parser.get_default(key)


def GTE(outputs, target, concealed_classes=None):
    # evaluate distance on all triplets
    gtes = []
    dist_ap = []
    dist_an = []
    concealed_classes = [] if concealed_classes is None else concealed_classes
    labels = set(target.data).difference(concealed_classes)
    for label in labels:
        positives = np.where((target == label).data)[0]
        negatives = np.where((target != label).data)[0]
        ap_pairs = np.array(list(combinations(positives, 2)))
        for a, p in ap_pairs:
            d_ap = torch.norm(outputs[a]-outputs[p], 2).repeat(len(negatives))
            d_an = torch.norm(outputs[a].unsqueeze(0).repeat(len(negatives), 1)-outputs[negatives], 2, dim=1)
            gtes.append(torch.sum(d_an < d_ap).type(torch.FloatTensor) / len(negatives))
            dist_ap.append(torch.mean(d_ap))
            dist_an.append(torch.mean(d_an))

    return torch.cat(gtes), np.mean(dist_ap), np.mean(dist_an)


def load_labels(dset, class_label, label_file='_svm_labels.pkl', confidence_threshold=None):
    skip_class_label = -2
    labels, num_classes = get_labels(label_file=label_file,
                                     confidence_threshold=confidence_threshold,
                                     skip_class_label=skip_class_label)
    assert len(dset) == len(labels), 'number of given labels does not match number of samples in dataset'

    # if label belongs to skip class delete the sample from the dataset
    valid_samples = np.where(labels != skip_class_label)[0]

    df = dset.df.loc[valid_samples]
    df[class_label] = labels[valid_samples]
    df.index = range(len(valid_samples))

    dset.df = df
    dset.labels_to_ints = dset._get_labels_to_ints()            # actualize label dict to new int labels
    print('loaded labels from {} into dataset'.format(label_file))


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                normalize,
    ])

    if args.model.lower() == 'inception_v3':            # change input size to 299
        train_transform.transforms[0].size = (299, 299)
        val_transform.transforms[0].size = (299, 299)

    impath = args.im_path
    val_file = args.test_file
    train_file = args.train_file
    valset = Wikiart(path_to_info_file=val_file, path_to_images=impath,
                     classes=[args.class_label], transform=val_transform)
    trainset = Wikiart(path_to_info_file=train_file, path_to_images=impath,
                       classes=[args.class_label], transform=train_transform)
    load_labels(dset=trainset, label_file=args.label_file, class_label=args.class_label, confidence_threshold=args.conf_thresh)
    num_classes = 16 #len(trainset.labels_to_ints[args.class_label])                    # TODO: THIS GETS NUMBER OF CORRECT CLASSES AUTOMATICALLY FROM LABELS!!!
    if args.n_labels_ap is not None:
        if args.conceal_smallest:
            class_stat = sorted(Counter(valset.df[args.class_label]).items(), key=lambda x: x[1])
            concealed_classes_val = [name for name, _ in class_stat[:-args.n_labels_ap]]
        else:
            concealed_classes_val = np.random.choice(valset.labels_to_ints[args.class_label].keys(),
                                 len(valset.labels_to_ints[args.class_label].keys()) - args.n_labels_ap,
                                 replace=False)
    else:
        concealed_classes_val = None
    concealed_classes_train = [-1]

    # PARAMETERS
    use_cuda = args.use_gpu and torch.cuda.is_available()
    device_nb = args.device
    if use_cuda:
        torch.cuda.set_device(device_nb)
        torch.cuda.manual_seed_all(args.seed)

    if args.model.lower() not in ['squeezenet', 'mobilenet_v1', 'mobilenet_v2', 'vgg16_bn', 'inception_v3']:
        assert False, 'Unknown model {}\n\t+ Choose from: ' \
                      '[sqeezenet, mobilenet_v1, mobilenet_v2, vgg16_bn].'.format(args.model)
    elif args.model.lower() == 'mobilenet_v1':
        net = mobilenet_v1(pretrained=False, num_classes=num_classes)
    elif args.model.lower() == 'mobilenet_v2':
        net = mobilenet_v2(pretrained=False, num_classes=num_classes)
    elif args.model.lower() == 'vgg16_bn':
        net = vgg16_bn(pretrained=True, num_classes=num_classes)
    elif args.model.lower() == 'inception_v3':
        net = inception_v3(pretrained=True)
    else:       # squeezenet
        net = squeezenet(pretrained=True, num_classes=num_classes)

    if args.weight_file:
        print("=> loading weights from '{}'".format(args.weight_file))
        pretrained_dict = torch.load(args.weight_file, map_location=lambda storage, loc: storage)['state_dict']
        state_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if (k in state_dict.keys() and v.shape == state_dict[k].shape)}      # number of classes might have changed
        state_dict.update(pretrained_dict)
        net.load_state_dict(state_dict)

    # remove classifier and replace by Normalization Layer
    remove_fc(net, inplace=True)
    net.num_classes = -1

    n_parameters = sum([p.data.nelement() for p in net.parameters() if p.requires_grad])
    if use_cuda:
        net = net.cuda()
    print('Using {}\n\t+ Number of params: {}'.format(str(net).split('(', 1)[0], n_parameters))

    n_epochs = args.epochs
    lr = args.lr
    momentum = args.momentum

    log_interval = args.log_interval
    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    # tensorboard summary writer
    timestamp = time.strftime('%m-%d-%H-%M')
    expname = timestamp + '_triplets_' + str(net).split('(', 1)[0]
    if args.exp_name is not None:
        expname = expname + '_' + args.exp_name
    log = TBPlotter(os.path.join(log_dir, 'tensorboard', expname))
    log.print_logdir()

    # resume from checkpoint file
    chkpt_file = args.chkpt

    # allow auto-tuner to find best algorithm for the hardware
    cudnn.benchmark = True


    # ININTIALIZE TRAINING
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, threshold=1e-2, verbose=True)

    if args.triplet_selector.lower() not in ['random', 'semihard', 'hardest', 'mixed', 'khardest']:
        assert False, 'Unknown option {} for triplet selector. Choose from "random", "semihard", "hardest" or "mixed"' \
                      '.'.format(args.triplet_selector)
    elif args.triplet_selector.lower() == 'random':
        criterion = TripletLoss(margin=args.margin,
                                triplet_selector=RandomNegativeTripletSelector(args.margin, cpu=not use_cuda))
    elif args.triplet_selector.lower() == 'semihard' or args.triplet_selector.lower() == 'mixed':
        criterion = TripletLoss(margin=args.margin,
                                triplet_selector=SemihardNegativeTripletSelector(args.margin, cpu=not use_cuda))
    elif args.triplet_selector.lower() == 'khardest':
        criterion = TripletLoss(margin=args.margin,
                                triplet_selector=KHardestNegativeTripletSelector(args.margin, k=3, cpu=not use_cuda))
    else:
        criterion = TripletLoss(margin=args.margin,
                                triplet_selector=HardestNegativeTripletSelector(args.margin, cpu=not use_cuda))
    if use_cuda:
        criterion = criterion.cuda()

    kwargs = {'num_workers': 16} if use_cuda else {}


    counter = Counter(trainset.df[args.class_label].values)
    n_train_samples = np.sum([v for k, v in counter.items() if k not in concealed_classes_train])
    print('Start training with {} samples.'.format(n_train_samples))

    write_config(args, os.path.join(log_dir, expname), defaults,
                 extras={'concealed_classes': concealed_classes_val, 'n_train': n_train_samples})


    train_batch_sampler = BalancedBatchSampler_withNonesandNoise(trainset, selected_class=args.class_label,
                                                        n_classes=args.classes_per_batch,
                                                        n_samples=args.samples_per_class,
                                                        concealed_classes=concealed_classes_train,
                                                        n_concealed=args.pure_negatives_per_batch)
    trainloader = DataLoader(trainset, batch_sampler=train_batch_sampler, **kwargs)
    val_batch_sampler = BalancedBatchSampler_withNonesandNoise(valset, selected_class=args.class_label,
                                                       n_classes=args.classes_per_batch,
                                                       n_samples=args.samples_per_class,
                                                       concealed_classes=concealed_classes_val,
                                                       n_concealed=args.pure_negatives_per_batch)
    valloader = DataLoader(valset, batch_sampler=val_batch_sampler, **kwargs)

    # optionally resume from a checkpoint
    start_epoch = 1
    if chkpt_file is not None:
        if os.path.isfile(chkpt_file):
            print("=> loading checkpoint '{}'".format(chkpt_file))
            checkpoint = torch.load(chkpt_file, map_location=lambda storage, loc: storage)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(chkpt_file, checkpoint['epoch']))
            # copy tensorboard files
            chkpt_logdir = os.path.join('runs/tensorboard', chkpt_file.split('_checkpoint')[0])
            if os.path.isdir(chkpt_logdir):
                shutil.copytree(os.path.join(chkpt_logdir, 'test'), os.path.join(log_dir, 'test'))
                shutil.copytree(os.path.join(chkpt_logdir, 'train'), os.path.join(log_dir, 'train'))
            else:
                print("did not find tensorboard logfiles for checkpoint at {}.".format(chkpt_logdir))
        else:
            print("=> no checkpoint found at '{}'".format(chkpt_file))

    def train(epoch):
        losses = AverageMeter()
        gtes = AverageMeter()
        non_zero_triplets = AverageMeter()
        distances_ap = AverageMeter()
        distances_an = AverageMeter()
        # emb_norms = AverageMeter()

        # switch to train mode
        net.train()
        for batch_idx, (data, target) in enumerate(trainloader):
            target = target[0]
            if use_cuda:
                data, target = Variable(data.cuda()), Variable(target.cuda())
            else:
                data, target = Variable(data), Variable(target)

            # compute output
            outputs = net(data)

            loss_triplet, n_triplets = criterion(outputs, target, concealed_classes=concealed_classes_train)
            # loss_embedd = outputs.norm(2)
            # loss = loss_triplet + 0.001 * loss_embedd

            non_zero_triplets.update(n_triplets, target.size(0))
            # measure GTE and record loss
            gte, dist_ap, dist_an = GTE(outputs, target, concealed_classes_train)           # do not compute ap pairs for concealed classes
            gtes.update(gte.data[0], target.size(0))
            distances_ap.update(dist_ap.data[0], target.size(0))
            distances_an.update(dist_an.data[0], target.size(0))
            losses.update(loss_triplet.data[0], target.size(0))
            # emb_norms.update(loss_embedd.data[0], target.size(0))

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss_triplet.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{}]\t'
                      'Loss: {:.4f} ({:.4f})\t'
                      'GTE: {:.2f}% ({:.2f}%)\t'
                      'Non-zero Triplets: {:d} ({:d})'.format(
                    epoch, batch_idx * len(target), len(trainloader) * len(target),
                    float(losses.val), float(losses.avg),
                    float(gtes.val) * 100., float(gtes.avg) * 100.,
                    int(non_zero_triplets.val), int(non_zero_triplets.avg)))

        # log avg values to somewhere
        log.write('loss', float(losses.avg), epoch, test=False)
        log.write('gte', float(gtes.avg), epoch, test=False)
        log.write('non-zero trplts', int(non_zero_triplets.avg), epoch, test=False)
        log.write('dist_ap', float(distances_ap.avg), epoch, test=False)
        log.write('dist_an', float(distances_an.avg), epoch, test=False)
        # log.write('emb_norms', float(emb_norms.avg), epoch, test=False)


    def test(epoch):
        losses = AverageMeter()
        gtes = AverageMeter()
        non_zero_triplets = AverageMeter()
        distances_ap = AverageMeter()
        distances_an = AverageMeter()
        # emb_norms = AverageMeter()

        # switch to evaluation mode
        net.eval()
        for batch_idx, (data, target) in enumerate(valloader):
            target = target[0]
            if use_cuda:
                data, target = Variable(data.cuda()), Variable(target.cuda())
            else:
                data, target = Variable(data), Variable(target)
            # compute output
            outputs = net(data)

            loss_triplet, n_triplets = criterion(outputs, target)
            # loss_embedd = outputs.norm(2)

            non_zero_triplets.update(n_triplets, target.size(0))
            # measure GTE and record loss
            gte, dist_ap, dist_an = GTE(outputs, target, concealed_classes_val)  # do not compute ap pairs for concealed classes
            gtes.update(gte.data[0], target.size(0))
            distances_ap.update(dist_ap.data[0], target.size(0))
            distances_an.update(dist_an.data[0], target.size(0))
            losses.update(loss_triplet.data[0], target.size(0))
            # emb_norms.update(loss_embedd.data[0], target.size(0))


        print('\nVal set: Average loss: {:.4f} Average GTE {:.2f}%, '
              'Average non-zero triplets: {:d} LR: {:.6f}'.format(float(losses.avg), float(gtes.avg) * 100.,
                                                       int(non_zero_triplets.avg),
                                                                  optimizer.param_groups[-1]['lr']))
        log.write('loss', float(losses.avg), epoch, test=True)
        log.write('gte', float(gtes.avg), epoch, test=True)
        log.write('non-zero trplts', int(non_zero_triplets.avg), epoch, test=True)
        log.write('dist_ap', float(distances_ap.avg), epoch, test=True)
        log.write('dist_an', float(distances_an.avg), epoch, test=True)
        # log.write('emb_norms', float(emb_norms.avg), epoch, test=True)
        return losses.avg, 1 - gtes.avg

    if start_epoch == 1:         # compute baseline:
        _, best_acc = test(epoch=0)
    else:       # checkpoint was loaded
        best_acc = best_acc

    for epoch in range(start_epoch, n_epochs + 1):
        if args.triplet_selector.lower() == 'mixed' and epoch == 26:
            criterion.triplet_selector = HardestNegativeTripletSelector(args.margin, cpu=not use_cuda)
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


if __name__ == '__main__':
    main()
