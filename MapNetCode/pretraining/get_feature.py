import os
import sys
import argparse
import pickle
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import h5py

from dataset import Wikiart
from model import mobilenet_v2, vgg16_bn

parser = argparse.ArgumentParser(description='Extract features from wikiart dataset.')
parser.add_argument('--exp_name', type=str, help='Name appended to generated output file name.')
parser.add_argument('--model', type=str, help='Choose from "mobilenet_v2", or "vgg16_bn".')
parser.add_argument('--weight_file', type=str, help='File to load pretrained weights from.')
parser.add_argument('--output_dir', default='output', type=str, help='Directory to which features are saved.')

parser.add_argument('--info_file', type=str, help='Path to dataset info file.')
parser.add_argument('--stat_file', type=str, help='.pkl file containing artist dataset mean and std.')
parser.add_argument('--im_path', default='/export/home/asanakoy/workspace/wikiart/images', type=str,
                    help='Path to Wikiart images')

parser.add_argument('--batch_size', default=64, type=int, help='Batch size".')

parser.add_argument('--use_gpu', default=True, type=bool, help='Use gpu or not.')
parser.add_argument('--device', default=0, type=int, help='Number of gpu device to use.')

# sys.argv = []
args = parser.parse_args()
# args.model = 'mobilenet_v2'
# args.weight_file = 'runs/06-02-23-27_MobileNetV2_artist_49_ft_model_best.pth.tar'
# args.info_file = '../wikiart/datasets/info_artist_49_test.hdf5'
# args.stat_file = '../wikiart/datasets/info_artist_49_train_mean_std.pkl'


def main():
    # LOAD DATASET
    stat_file = args.stat_file
    with open(stat_file, 'r') as f:
        data = pickle.load(f)
        mean, std = data['mean'], data['std']
        mean = [float(m) for m in mean]
        std = [float(s) for s in std]
    normalize = transforms.Normalize(mean=mean, std=std)
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    if args.model.lower() == 'inception_v3':  # change input size to 299
        img_transform.transforms[0].size = (299, 299)

    impath = args.im_path
    dataset = Wikiart(path_to_info_file=args.info_file, path_to_images=impath,
                      classes=['artist_name'], transform=img_transform)

    # PARAMETERS
    use_cuda = args.use_gpu and torch.cuda.is_available()
    device_nb = args.device
    if use_cuda:
        torch.cuda.set_device(device_nb)

    if args.model.lower() not in ['squeezenet', 'mobilenet_v1', 'mobilenet_v2', 'mobilenet_v3', 'mobilenet_v4', 'vgg16_bn', 'inception_v3']:
        assert False, 'Unknown model {}\n\t+ Choose from: ' \
                      '[sqeezenet, mobilenet_v1, mobilenet_v2, mobilenet_v3, mobilenet_v4, vgg16_bn].'.format(args.model)
    elif args.model.lower() == 'mobilenet_v1':
        net = mobilenet_v1(pretrained=args.weight_file is None)
    elif args.model.lower() == 'mobilenet_v2':
        net = mobilenet_v2(pretrained=args.weight_file is None)
    elif args.model.lower() == 'mobilenet_v3':
        net = mobilenet_v3(pretrained=args.weight_file is None)
    elif args.model.lower() == 'mobilenet_v4':
        net = mobilenet_v4(pretrained=args.weight_file is None)
    elif args.model.lower() == 'vgg16_bn':
        net = vgg16_bn(pretrained=args.weight_file is None)
    elif args.model.lower() == 'inception_v3':
        net = inception_v3(pretrained=args.weight_file is None)
    else:  # squeezenet
        net = squeezenet(pretrained=args.weight_file is None)

    if args.weight_file:
        print("=> loading weights from '{}'".format(args.weight_file))
        pretrained_dict = torch.load(args.weight_file, map_location=lambda storage, loc: storage)['state_dict']
        state_dict = net.state_dict()
        pretrained_dict = {k.replace('bodynet.', ''): v for k, v in pretrained_dict.items()
                           # in case of multilabel weight file
                           if (k.replace('bodynet.', '') in state_dict.keys() and v.shape == state_dict[
            k.replace('bodynet.', '')].shape)}  # number of classes might have changed
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
        net.load_state_dict(state_dict)

    remove_fc(net, inplace=True)
    if use_cuda:
        net = net.cuda()

    kwargs = {'num_workers': 4} if use_cuda else {}
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    net.eval()
    features = []
    targets = []
    for i, data in enumerate(loader):
        if isinstance(data, tuple) or isinstance(data, list):         # loader returns data, label
            targets.append(data[1])
            data = data[0]
        if (i+1) % 10 == 0:
            print('{}/{}'.format(i+1, len(loader)))
        input = Variable(data, requires_grad=False) if not use_cuda else Variable(data.cuda(), requires_grad=False)
        output = net(input)
        features.append(output.data.cpu())

    features = torch.cat(features)
    features = features.numpy()
    image_names = dataset.df['image_id'].values.astype(str)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    expname = '' if args.exp_name is None else '_' + args.exp_name
    outfile = os.path.join(args.output_dir, str(net).split('(')[0] + '_' +
                           args.info_file.split('/')[-1].split('.')[0] + expname + '.hdf5')

    with h5py.File(outfile, 'w') as f:
        f.create_dataset('features', features.shape, dtype=features.dtype, data=features)
        f.create_dataset('image_names', image_names.shape, dtype=image_names.dtype, data=image_names)
    print('Saved features to {}'.format(outfile))


if __name__ == '__main__':
    main()
