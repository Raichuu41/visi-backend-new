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
import warnings

from aux import TBPlotter
from model import load_featurenet, MapNet_pretraining_1, MapNet_pretraining_2, \
    MapNet_pretraining_3, MapNet_pretraining_4, MapNet_pretraining_0

class LossLogger:
    def __init__(self):
        self.current_epoch = []
        self.logged = []

    def log(self, losses):
            self.current_epoch.extend(losses)
    
    def end_epoch(self):
        mean = sum(self.current_epoch) / float(len(self.current_epoch))
        self.logged.append(mean)
        self.current_epoch = []
        return mean



def train_loop(model, trainset, testset, lr=1e-3, use_gpu=True, outpath=None, resume=False):

    if outpath is not None and not os.path.isdir(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))
    
    file_path = os.path.join(outpath, "nclass_best_loss_{}.pt".format(type(model).__name__))

    loader_kwargs = {'num_workers': 4} if use_gpu else {}

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, **loader_kwargs)
    testloader = DataLoader(testset, batch_size=64, **loader_kwargs)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=1e-3, verbose=True)
    loss_fn = torch.nn.CrossEntropyLoss(size_average=False, reduce=False)
    epoch = 1

    if resume:
        sd = torch.load(file_path)
        model.load_state_dict(sd["state_dict"])
        optimizer.load_state_dict(sd["optimizer"])
        lr_scheduler.load_state_dict(sd["scheduler"])
        epoch = sd["epoch"]

    """
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
    """
    # train network until scheduler reduces learning rate to threshold value
    lr_threshold = 5e-5

    best_loss = float('inf')
    best_epoch = -1

    train_logger = LossLogger()
    val_logger = LossLogger()
    acc_logger = LossLogger()

    while not (optimizer.param_groups[-1]['lr'] < lr_threshold):

        # training
        model.train()
        for i, data in tqdm(enumerate(trainloader), desc="[tr]epoch {}".format(epoch)):
            if isinstance(data, list) or isinstance(data, tuple):
                inp = data[0].cuda() if use_gpu else data[0]
                lbl = data[1].cuda() if use_gpu else data[1]
            else:
                raise RuntimeError("No labels given for training.")
            output = model.class_head(model(inp))

            """
            torch.set_printoptions(profile="full")
            print("!!!DEBUG!!!")
            print(output.size())
            #print(output.argmax(dim=1))
            print(lbl)
            print("???DEBUG???")
            torch.set_printoptions(profile="default")
            """

            loss = loss_fn(output, lbl)
            train_logger.log(loss.tolist())
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        trainloss = train_logger.end_epoch()

        model.eval()
        for i, data in tqdm(enumerate(testloader), desc="[va]epoch {}".format(epoch)):
            if isinstance(data, list) or isinstance(data, tuple):
                inp = data[0].cuda() if use_gpu else data[0]
                lbl = data[1].cuda() if use_gpu else data[1]
            else:
                raise RuntimeError("No labels given for validation.")
            output = model.class_head(model(inp))

            loss = loss_fn(output, lbl)
            acc_logger.log((output.argmax(dim=1) == lbl).tolist())
            val_logger.log(loss.tolist())
        
        testacc  = acc_logger.end_epoch()
        testloss = val_logger.end_epoch()
        lr_scheduler.step(testloss)

        if testloss < best_loss:
            best_loss = testloss
            best_epoch = epoch
            if outpath is not None:
                torch.save({                # save best model + checkpoint
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'best_loss': best_loss,
                    #'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict()
                }, file_path)

        epoch += 1
        tqdm.write("Ended Epoch {} with {:.3f}tl; {:.3f}vl; {:.3f}%".format(epoch, trainloss, testloss, testacc))


    print('Finished training mapnet with best training loss: {:.4f} from epoch {}.'.format(best_loss, best_epoch))
    if outpath is not None:
        print('Saved model to {}.'.format(outpath))

def use_classification_head(model, n_labels):
    model.class_head = torch.nn.Sequential(torch.nn.ReLU(inplace=False),torch.nn.Linear(model.mapping[-1].out_features, n_labels))



# mostly copied from pretraining.py

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='Pretrain Mapnet on ImageNet.')

    # general configurations
    parser.add_argument('--n_layers', default=2, type=int,
                        help='Number of mapping layers.')
    parser.add_argument('--device', default=0, type=int,
                        help='CUDA device')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='Resume from an earlier snapshot.')

    """
    parser.add_argument('--wait', default=0, type=int,
                        help='Number of seconds to wait before execution.')
    parser.add_argument('--block', default=0, type=int,
                        help='GB to block before execution.')
    """
    
    args = parser.parse_args()
    if args.n_layers == 0:
        mapnet_model = MapNet_pretraining_0
    elif args.n_layers == 1:
        mapnet_model = MapNet_pretraining_1
    elif args.n_layers == 2:
        mapnet_model = MapNet_pretraining_2
    elif args.n_layers == 3:
        mapnet_model = MapNet_pretraining_3
    elif args.n_layers == 4:
        mapnet_model = MapNet_pretraining_4
    else:
        raise AttributeError('Number of layers has to be [0,1,2,3,4].')

    with torch.cuda.device(args.device):

        """
        if args.wait > 0:
            if args.block > 0:
                block(args.block, args.wait)
            else:
                sleep(args.wait)
        """
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

        dataset_prefix = 'ImageNet_class'

        # train mapnet
        outpath = './pretraining'
        weightfile_name = '{}_train_nlayers_{}.pth.tar'.format(dataset_prefix, args.n_layers)
        logdir = './pretraining/.log/class_nlayers_{}'.format(args.n_layers)
        if not os.path.isdir(outpath):
            os.makedirs(outpath)
        if not os.path.isdir(logdir):
            os.makedirs(logdir)

        """
        train_labels = np.array(map(lambda x: x[1], train_dataset.imgs))         # 0 is reserved as negative label
        test_labels = np.array(map(lambda x: x[1], val_dataset.imgs))
        """

        model = mapnet_model(feature_dim=512, output_dim=2)

        use_classification_head(model, len(train_dataset.classes))

        model = model.cuda()
        # fix the weights of the feature network
        for param in model.featurenet.parameters():
            param.requires_grad = False

        """
        pretrain_mapnet(model, train_dataset, train_labels,
                        testset=val_dataset, test_labels=test_labels,
                        lr=1e-3,
                        random_state=123, use_gpu=True,
                        verbose=True, outpath=os.path.join(outpath, weightfile_name), log_dir=logdir,
                        plot_fn=None)
        """

        warnings.filterwarnings("ignore", category=UserWarning) # ignore PIL EXIF warnings
        train_loop(model, train_dataset, val_dataset, outpath=outpath, resume=args.resume)

        exit()
        # not important form here on (for now...)

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
