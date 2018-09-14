import os
import sys
import argparse
import pickle
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import h5py

from dataset import Wikiart
from model import mobilenet_v2, vgg16_bn, narrownet, remove_fc

sys.path.append('/export/home/kschwarz/Documents/Masters/FullPipeline')
import matplotlib as mpl
mpl.use('TkAgg')
from aux import AverageMeter, TBPlotter, save_checkpoint, write_config, load_weights

sys.path.append('/export/home/kschwarz/Documents/Data/Geometric_Shapes')
from shape_dataset import ShapeDataset


parser = argparse.ArgumentParser(description='Extract features from wikiart dataset.')
parser.add_argument('--exp_name', type=str, help='Name appended to generated output file name.')
parser.add_argument('--model', type=str, help='Choose from "mobilenet_v2", or "vgg16_bn".')
parser.add_argument('--not_narrow', default=False, action='store_true', help='Do not narrow feature down to 128 dim.')
parser.add_argument('--weight_file', type=str, help='File to load pretrained weights from.')
parser.add_argument('--output_dir', default='../features', type=str, help='Directory to which features are saved.')

parser.add_argument('--info_file', type=str, help='Path to hdf5 file containing dataframe of dataset.')
parser.add_argument('--stat_file', default='wikiart_datasets/info_artist_49_multilabel_train_mean_std.pkl', type=str,
                    help='.pkl file containing dataset mean and std.')
parser.add_argument('--im_path', default='/export/home/kschwarz/Documents/Data/Wikiart_artist49_images', type=str,
                    help='Path to Wikiart images')
parser.add_argument('--shape_dataset', default=False, action='store_true', help='Use artificial shape dataset')

parser.add_argument('--batch_size', default=64, type=int, help='Batch size".')

parser.add_argument('--use_gpu', default=True, type=bool, help='Use gpu or not.')
parser.add_argument('--device', default=0, type=int, help='Number of gpu device to use.')

# sys.argv = []
args = parser.parse_args()
# args.model = 'mobilenet_v2'
# args.weight_file = 'MapNetCode/pretraining/models/09-10-17-39_VGG_model_best.pth.tar'
# args.info_file = 'wikiart_datasets/info_artist_49_multilabel_val.hdf5'


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

    if args.shape_dataset:
        classes = ['shape']
        dataset = ShapeDataset(root_dir='/export/home/kschwarz/Documents/Data/Geometric_Shapes', split=args.info_file,
                              classes=classes, transform=img_transform)
    else:
        dataset = Wikiart(path_to_info_file=args.info_file, path_to_images=args.im_path,
                          classes=['artist_name'], transform=img_transform)

    # PARAMETERS
    use_cuda = args.use_gpu and torch.cuda.is_available()
    device_nb = args.device
    if use_cuda:
        torch.cuda.set_device(device_nb)

    # INITIALIZE NETWORK
    if args.model.lower() not in ['mobilenet_v2', 'vgg16_bn']:
        raise NotImplementedError('Unknown Model {}\n\t+ Choose from: [mobilenet_v2, vgg16_bn].'
                                  .format(args.model))
    elif args.model.lower() == 'mobilenet_v2':
        featurenet = mobilenet_v2(pretrained=True)
    elif args.model.lower() == 'vgg16_bn':
        featurenet = vgg16_bn(pretrained=True)
    if args.not_narrow:
        net = featurenet
    else:
        net = narrownet(featurenet)
    if use_cuda:
        net = net.cuda()

    remove_fc(net, inplace=True)
    print('Extract features using {}.'.format(str(net)))

    if args.weight_file:
        pretrained_dict = load_weights(args.weight_file, net.state_dict(), prefix_file='bodynet.')
        net.load_state_dict(pretrained_dict)

    if use_cuda:
        net = net.cuda()

    kwargs = {'num_workers': 8} if use_cuda else {}
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    net.eval()
    features = []
    for i, data in enumerate(loader):
        if isinstance(data, tuple) or isinstance(data, list):         # loader returns data, label
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
    if args.shape_dataset:
        outfile = os.path.join(args.output_dir, 'ShapeDataset_' + str(net).split('(')[0] + '_' +
                               args.info_file.split('/')[-1].split('.')[0] + expname + '.hdf5')
    else:
        outfile = os.path.join(args.output_dir, str(net).split('(')[0] + '_' +
                               args.info_file.split('/')[-1].split('.')[0] + expname + '.hdf5')

    with h5py.File(outfile, 'w') as f:
        f.create_dataset('features', features.shape, dtype=features.dtype, data=features)
        f.create_dataset('image_names', image_names.shape, dtype=image_names.dtype, data=image_names)
    print('Saved features to {}'.format(outfile))


if __name__ == '__main__':
    main()
