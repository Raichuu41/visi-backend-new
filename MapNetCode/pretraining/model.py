import torch
from MobileNetV2 import MobileNetV2             # 3505960 parameters
import torchvision.models as models
import copy
import torch.nn as nn

weight_file_v2 = 'mobilenetv2_718.pth.tar'


def mobilenet_v2(pretrained=False, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        pretrained_dict = torch.load(weight_file_v2, map_location=lambda storage, loc: storage)
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if (k in state_dict.keys() and v.shape == state_dict[k].shape)}
        state_dict.update(pretrained_dict)
        model.load_state_dict(state_dict)
    return model


def vgg16_bn(pretrained=False, **kwargs):
    model = models.vgg16_bn(**kwargs)
    if pretrained:
        pretrained_dict = models.vgg16_bn(pretrained=True).state_dict()
        state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if (k in state_dict.keys() and v.shape == state_dict[k].shape)}
        state_dict.update(pretrained_dict)
        model.load_state_dict(state_dict)
    return model


class NarrowNet(torch.nn.Module):
    """Similar to MobileNetV2 just with more fc layers at the end."""
    def __init__(self, featurenet, dim_feature_out=128, num_classes=1000):
        super(NarrowNet, self).__init__()
        self.featurenet = featurenet
        self.num_classes = num_classes
        self.dim_feature_out = dim_feature_out
        # initialise body network
        module_names = [name for name, _ in list(self.featurenet.named_children())]
        assert 'fc' in module_names or 'classifier' in module_names, 'require featurenet with layer type "fc" or "classifier"'
        # remove classifier and generate a bunch of new ones
        self.in_features = None
        if 'fc' in module_names:
            for m in self.featurenet.fc.modules():
                if isinstance(m, torch.nn.Linear):
                    self.in_features = m.in_features
                    break
        else:
            for m in self.featurenet.classifier.modules():
                if isinstance(m, torch.nn.Linear):
                    self.in_features = m.in_features
                    break
        assert self.in_features is not None, 'could not find Linear layer in classifier of body network'
        # self.featurenet = remove_fc(self.featurenet, inplace=False)
        # remove fc layer
        for name, module in self.featurenet.named_children():
            if name == 'classifier' or name == 'fc':
                self.featurenet.__setattr__(name, Identity())

        self.featurenet.num_classes = -1

        # some fully connected layers
        self.linear1 = torch.nn.Linear(self.in_features, self.dim_feature_out, bias=True)

        self.classifier = torch.nn.Linear(self.dim_feature_out, self.num_classes, bias=True)

    def forward(self, x):
        x = self.featurenet.forward(x)
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.classifier(x)
        return x

    def __repr__(self):
        name = 'NarrowNet{}_'.format(self.dim_feature_out) + self.featurenet.__repr__().split('(')[0]
        return name


# Identity Layer
class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def remove_fc(net, inplace=False):
    model = copy.copy(net) if inplace else copy.deepcopy(net)
    if str(net).startswith('VGG'):
        for name, module in model.named_children():
            if name == 'classifier' or name == 'fc':
                fc1 = module[0]
                model.__delattr__(name)
                model.__setattr__(name, fc1)
    elif str(net).startswith('MobileNetV2'):
        for name, module in model.named_children():
            if name == 'classifier' or name == 'fc':
                fc1 = nn.Linear(in_features=1280, out_features=1024)
                model.__delattr__(name)
                model.__setattr__(name, fc1)
    else:
        for name, module in model.named_children():
            if name == 'classifier' or name == 'fc':
                model.__setattr__(name, Identity())
    if not inplace:
        return model


def make_featurenet(net, inplace=False):
    model = copy.copy(net) if inplace else copy.deepcopy(net)
    if str(net).startswith('AlexNet'):
        del model.classifier[2:]
        feature_dim = 4096
    elif str(net).startswith('VGG'):
        del model.classifier[1:]
        feature_dim = 4096
    elif str(net).startswith('MobileNetV2'):
        del model.classifier[:]
        feature_dim = 1280
    else:
        raise NotImplementedError
    if not inplace:
        return model, feature_dim
    return feature_dim


def narrownet(featurenet, **kwargs):
    model = NarrowNet(featurenet, **kwargs)
    return model


class OctopusNet(torch.nn.Module):
    def __init__(self, bodynet, n_labels):
        super(OctopusNet, self).__init__()
        # initialise body network
        module_names = [name for name, _ in list(bodynet.named_children())]
        assert 'fc' in module_names or 'classifier' in module_names, 'require network with layer type "fc" or "classifier"'
        # remove classifier and generate a bunch of new ones
        self.in_features = None
        if 'fc' in module_names:
            for m in bodynet.fc.modules():
                if isinstance(m, torch.nn.Linear):
                    self.in_features = m.in_features
                    break
        else:
            if str(bodynet).startswith('VGG'):
                self.in_features = 4096
            elif str(bodynet).startswith('MobileNetV2'):
                self.in_features = 1024
            else:
                for m in bodynet.classifier.modules():
                    if isinstance(m, torch.nn.Linear):
                        self.in_features = m.in_features
                        break
        assert self.in_features is not None, 'could not find Linear layer in classifier of body network'
        self.bodynet = remove_fc(bodynet, inplace=False)
        self.bodynet.num_classes = -1

        # set up the classifiers
        self.n_labels = n_labels
        self.classifiers = torch.nn.ModuleList([
            torch.nn.Linear(self.in_features, out_features=n, bias=True) for n in self.n_labels])

    def forward(self, x):
        x = self.bodynet.forward(x)
        outputs = []
        for clf in self.classifiers:
            outputs.append(clf.forward(x))
        return outputs




