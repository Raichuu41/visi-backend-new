import torch
import argparse
import os
import numpy as np
import deepdish as dd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import vgg16_bn
from tqdm import tqdm
from time import sleep

from helpers import BalancedBatchSampler
from train import pretrain_mapnet
from initialization import Initializer
from model import load_featurenet, MapNet_pretraining_1, MapNet_pretraining_2, \
    MapNet_pretraining_3, MapNet_pretraining_4
from aux import load_weights


def get_features(model, dataset, batchsize=16, use_gpu=True, verbose=False):
    use_gpu = use_gpu and torch.cuda.is_available()

    if use_gpu:
        model = model.cuda()

    loader_kwargs = {'num_workers': 4} if use_gpu else {}
    dataloader = DataLoader(dataset, batchsize, shuffle=False, drop_last=False, **loader_kwargs)

    features = []
    for i, data in enumerate(dataloader):
        if verbose:
            print('{}/{}'.format(i + 1, len(dataloader)))
        if isinstance(data, list) or isinstance(data, tuple):
            input = data[0].cuda() if use_gpu else data[0]
        else:
            input = data.cuda() if use_gpu else data
        output = model(input)
        features.append(output.data.cpu())

    features = torch.cat(features)
    return features.numpy()


def block(gb, secs):
    def _block():
        element_size = 4  # torch.FloatTensor uses 4 Bytes
        n_elements = int(gb / element_size * 1e9)

        block = torch.ones(n_elements)
        block.cuda()
        sleep(secs)
    _block()
    torch.cuda.empty_cache()


parser = argparse.ArgumentParser(description='Pretrain Mapnet on ImageNet.')

# general configurations
parser.add_argument('--n_layers', default=1, type=int,
                    help='Number of mapping layers.')
parser.add_argument('--wait', default=0, type=int,
                    help='Number of seconds to wait before execution.')
parser.add_argument('--block', default=0, type=int,
                    help='GB to block before execution.')

args = parser.parse_args()
if args.n_layers == 1:
    mapnet_model = MapNet_pretraining_1
elif args.n_layers == 2:
    mapnet_model = MapNet_pretraining_2
elif args.n_layers == 3:
    mapnet_model = MapNet_pretraining_3
elif args.n_layers == 4:
    mapnet_model = MapNet_pretraining_4
else:
    raise AttributeError('Number of layers has to be [1,2,3,4].')

if __name__ == '__main__':
    if args.wait > 0:
        if args.block > 0:
            block(args.block, args.wait)
        else:
            sleep(args.wait)
    impath = '/net/hci-storage02/groupfolders/compvis/bbrattol/ImageNet'
    traindir = os.path.join(impath, 'ILSVRC2012_img_train')
    valdir = os.path.join(impath, 'ILSVRC2012_img_val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    dataset_prefix = 'ImageNet_label'

    # train mapnet
    outpath = './pretraining'
    weightfile_name = '{}_train_nlayers_{}.pth.tar'.format(dataset_prefix, args.n_layers)
    logdir = './pretraining/.log/nlayers_{}'.format(args.n_layers)
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    train_labels = np.array(map(lambda x: x[1], train_dataset.imgs)) + 1         # 0 is reserved as negative label
    test_labels = np.array(map(lambda x: x[1], val_dataset.imgs)) + 1
    assert (np.all(train_labels != 0) and np.all(test_labels != 0)), '0-Label must not be used.'

    model = mapnet_model(feature_dim=512, output_dim=2)
    model = model.cuda()
    # fix the weights of the feature network
    for param in model.featurenet.parameters():
        param.requires_grad = False

    pretrain_mapnet(model, train_dataset, train_labels,
                    testset=val_dataset, test_labels=test_labels,
                    lr=1e-3,
                    random_state=123, use_gpu=True,
                    verbose=True, outpath=os.path.join(outpath, weightfile_name), log_dir=logdir,
                    plot_fn=None)

    # extract features to train projection
    print('Extract features...')
    best_weights = load_weights(os.path.join(outpath, weightfile_name), model.state_dict())
    model.load_state_dict(best_weights)
    model.eval()
    train_ids = np.array(map(lambda x: x[0].split('/')[-1].replace('.JPEG', ''), train_dataset.imgs))
    train_features = get_features(model=model, dataset=train_dataset, batchsize=16,
                                  use_gpu=True, verbose=True)
    # save train_features
    outdir = './features'
    outdict = {'image_id': train_ids, 'features': train_features}
    outfilename = os.path.join(outdir, weightfile_name.replace('.pth.tar', '.h5'))
    dd.io.save(outfilename, outdict)

    # save val_features
    val_features = get_features(model=model, dataset=val_dataset, batchsize=16,
                                use_gpu=True, verbose=True)
    outfilename = outfilename.replace('train', 'test')
    val_ids = np.array(map(lambda x: x[0].split('/')[-1].replace('.JPEG', ''), val_dataset.imgs))
    outdict = {'image_id': val_ids, 'features': val_features}
    dd.io.save(outfilename, outdict)
