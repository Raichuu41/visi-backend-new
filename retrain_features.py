import argparse
from retrain_features_with_labels import train_multiclass as train_w_labels
from retrain_features_with_triplets import train_multiclass as train_w_triplets


parser = argparse.ArgumentParser(description='Retrain features either with triplets or labels')

parser.add_argument('--use_triplets', default=False, action='store_true', help='Use triplets instead of labels.')
parser.add_argument('--device', default=0, type=int, help='CUDA device number')
parser.add_argument('--classes', type=str, help='classes of label file, separate them with "," ')
parser.add_argument('--label_file', default='_user_labels.pkl', type=str, help='File containing training set labels.')
parser.add_argument('--weight_file', default=None, type=str, help='Pretrained weights for network (mobilenetV2).')
parser.add_argument('--exp_name', default=None, type=str, help='Name of experiment.')


args = parser.parse_args()

train_file = '../wikiart/datasets/info_artist_49_multilabel_test.hdf5'
test_file = '../wikiart/datasets/info_artist_49_multilabel_val.hdf5'
stat_file = '../wikiart/datasets/info_artist_49_multilabel_train_mean_std.pkl'
model = 'mobilenet_v2'

classes = args.classes.split(',')
label_file = args.label_file

im_path = '/export/home/kschwarz/Documents/Data/Wikiart_artist49_images'
chkpt = None
weight_file = args.weight_file
triplet_selector = 'semihard'
margin = 0.2
labels_per_class = 4
samples_per_label = 4
use_gpu = True
device = args.device
epochs = 100
batch_size = 32
lr = 1e-3
momentum = 0.9
log_interval = 10
log_dir = 'runs/'
exp_name = args.exp_name
seed = 123

train_func = train_w_triplets if args.use_triplets else train_w_labels

if __name__ == '__main__':
    if args.use_triplets:
        train_w_triplets(train_file=train_file, test_file=test_file, stat_file=stat_file,
                         model=model, classes=classes, label_file=label_file, im_path=im_path,
                         chkpt=chkpt, weight_file=weight_file, triplet_selector=triplet_selector,
                         margin=margin, labels_per_class=labels_per_class, samples_per_label=samples_per_label,
                         use_gpu=use_gpu, device=device, epochs=epochs, batch_size=batch_size, lr=lr,
                         momentum=momentum, log_interval=log_interval, log_dir=log_dir, exp_name=exp_name,
                         seed=seed)
    else:
        train_w_labels(train_file=train_file, test_file=test_file, stat_file=stat_file,
                       model=model, classes=classes, label_file=label_file, im_path=im_path,
                       chkpt=chkpt, weight_file=weight_file,
                       use_gpu=use_gpu, device=device, epochs=epochs, batch_size=batch_size, lr=lr,
                       momentum=momentum, log_interval=log_interval, log_dir=log_dir, exp_name=exp_name,
                       seed=seed)