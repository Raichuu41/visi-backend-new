import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn


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
        x = F.relu(self.mapping(x))

        x = self.embedder(x)
        return x
