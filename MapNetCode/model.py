from torch import nn
import torch
import torch.nn.functional as F


class MapNet(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(MapNet, self).__init__()
        self.in_features = feature_dim

        # set up intermediate layer for mapping feature space
        self.mapping = torch.nn.Sequential(
            # nn.Linear(in_features=self.in_features, out_features=2048),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=True),
            # nn.Linear(in_features=2048, out_features=4096),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            # nn.Dropout(p=0.0001),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
            nn.ReLU(inplace=True),
            # nn.Linear(in_features=2048, out_features=self.in_features),
            nn.Linear(in_features=self.in_features, out_features=self.in_features),
        )
        # initialize weights to produce identity mapping
        for name, param in self.mapping.named_parameters():
            if name.endswith('weight'):
                param.data.copy_(torch.eye(param.shape[0], param.shape[1]))
            if name.endswith('bias'):
                param.data.copy_(torch.zeros(param.shape))

        # set up the tsne layers
        self.embedder = torch.nn.Sequential(
            # nn.Linear(in_features=self.in_features, out_features=1024),
            # nn.ReLU(inplace=True),
            nn.Linear(in_features=self.in_features, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=output_dim)
        )

    def forward(self, x):
        x = F.relu(self.mapping(x))
        x = self.embedder(x)
        return x