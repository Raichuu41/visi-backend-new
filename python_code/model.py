import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn
from aux import load_weights


def load_featurenet(verbose=False):
    if verbose:
        print('Load torchvision version of VGG16_bn with all feature layers and first classification layer.')
    model = vgg16_bn(pretrained=True)

    # modify classification layers to produce 4096 dimensional features
    fc_remain = model.classifier[0]
    model.__setattr__('classifier', fc_remain)

    return model


class MapNet(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(MapNet, self).__init__()
        self.in_features = feature_dim

        # set up the mapping layers (from feature space dimension to feature space dimension)
        self.mapping = torch.nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
        )

        # set up the projection layers, do not use inplace operation as mapped features might be used as output
        self.embedder = torch.nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=1024),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=128, out_features=output_dim)
        )

    def forward(self, x):
        x = F.relu(self.mapping(x))

        x = self.embedder(x)
        return x

class MapNet_pretraining_0(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(MapNet_pretraining_0, self).__init__()
        self.initial_dim = 4096
        self.in_features = feature_dim
        self.featurenet = load_featurenet()

        # set up the mapping layers (from feature space dimension to feature space dimension)
        self.mapping = torch.nn.Sequential(
            nn.Linear(in_features=self.initial_dim, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
        )

        # set up the projection layers, do not use inplace operation as mapped features might be used as output
        self.embedder = torch.nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=512),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=512, out_features=64),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=64, out_features=output_dim)
        )

    def forward(self, x):
        x = F.relu(self.featurenet(x))
        x = self.mapping(x)

        # x = self.embedder(F.relu(x))
        return x


class MapNet_pretraining_1(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(MapNet_pretraining_1, self).__init__()
        self.initial_dim = 4096
        self.in_features = feature_dim
        self.featurenet = load_featurenet()

        # set up the mapping layers (from feature space dimension to feature space dimension)
        self.mapping = torch.nn.Sequential(
            nn.Linear(in_features=self.initial_dim, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
        )

        # set up the projection layers, do not use inplace operation as mapped features might be used as output
        self.embedder = torch.nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=1024),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=128, out_features=output_dim)
        )

    def forward(self, x):
        x = F.relu(self.featurenet(x))
        x = self.mapping(x)

        # x = self.embedder(F.relu(x))
        return x


class MapNet_pretraining_2(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(MapNet_pretraining_2, self).__init__()
        self.initial_dim = 4096
        self.in_features = feature_dim
        self.featurenet = load_featurenet()

        # set up the mapping layers (from feature space dimension to feature space dimension)
        self.mapping = torch.nn.Sequential(
            nn.Linear(in_features=self.initial_dim, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
        )

        # set up the projection layers, do not use inplace operation as mapped features might be used as output
        self.embedder = torch.nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=1024),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=128, out_features=output_dim)
        )

    def forward(self, x):
        x = F.relu(self.featurenet(x))
        x = self.mapping(x)

        # x = self.embedder(F.relu(x))
        return x


class MapNet_pretraining_3(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(MapNet_pretraining_3, self).__init__()
        self.initial_dim = 4096
        self.in_features = feature_dim
        self.featurenet = load_featurenet()

        # set up the mapping layers (from feature space dimension to feature space dimension)
        self.mapping = torch.nn.Sequential(
            nn.Linear(in_features=self.initial_dim, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
        )

        # set up the projection layers, do not use inplace operation as mapped features might be used as output
        self.embedder = torch.nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=1024),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=128, out_features=output_dim)
        )

    def forward(self, x):
        x = F.relu(self.featurenet(x))
        x = self.mapping(x)

        # x = self.embedder(F.relu(x))
        return x


class MapNet_pretraining_4(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(MapNet_pretraining_4, self).__init__()
        self.initial_dim = 4096
        self.in_features = feature_dim
        self.featurenet = load_featurenet()

        # set up the mapping layers (from feature space dimension to feature space dimension)
        self.mapping = torch.nn.Sequential(
            nn.Linear(in_features=self.initial_dim, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
        )

        # set up the projection layers, do not use inplace operation as mapped features might be used as output
        self.embedder = torch.nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=1024),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=128, out_features=output_dim)
        )

    def forward(self, x):
        x = F.relu(self.featurenet(x))
        x = self.mapping(x)

        # x = self.embedder(F.relu(x))
        return x


def mapnet_1(pretrained=False, feature_dim=512, output_dim=2, new_pretrain=False):
    model = MapNet_pretraining_1(feature_dim=feature_dim, output_dim=output_dim)
    if pretrained:
        if new_pretrain:
            weight_file = "./pretraining/class_best_loss_MapNet_pretraining_0.pt"
        else:
            weight_file = './pretraining/ImageNet_label_train_nlayers_1.pth.tar'
        pretrained_dict = load_weights(weightfile=weight_file, state_dict_model=model.state_dict())
        model.load_state_dict(pretrained_dict)
    return model


def mapnet_2(pretrained=False, feature_dim=512, output_dim=2):
    model = MapNet_pretraining_2(feature_dim=feature_dim, output_dim=output_dim)
    if pretrained:
        weight_file = './pretraining/ImageNet_label_train_nlayers_2.pth.tar'
        pretrained_dict = load_weights(weightfile=weight_file, state_dict_model=model.state_dict())
        model.load_state_dict(pretrained_dict)
    return model


def mapnet_3(pretrained=False, feature_dim=512, output_dim=2):
    model = MapNet_pretraining_3(feature_dim=feature_dim, output_dim=output_dim)
    if pretrained:
        weight_file = './pretraining/ImageNet_label_train_nlayers_3.pth.tar'
        pretrained_dict = load_weights(weightfile=weight_file, state_dict_model=model.state_dict())
        model.load_state_dict(pretrained_dict)
    return model


def mapnet_4(pretrained=False, feature_dim=512, output_dim=2):
    model = MapNet_pretraining_4(feature_dim=feature_dim, output_dim=output_dim)
    if pretrained:
        weight_file = './pretraining/ImageNet_label_train_nlayers_4.pth.tar'
        pretrained_dict = load_weights(weightfile=weight_file, state_dict_model=model.state_dict())
        model.load_state_dict(pretrained_dict)
    return model