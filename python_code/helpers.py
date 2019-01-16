import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import BatchSampler


def is_imfile(filename, extensions=('.jpg', '.png')):
    if isinstance(extensions, str):
        extensions = (extensions, )
    is_valid = [filename.endswith(ext) for ext in extensions]
    return any(is_valid)


def get_imgid(filename):
    return filename.split('/')[-1].split('.')[0]


class ImageDataset(Dataset):
    """Dataset with images only, i.e. no labels.
    Returns PIL images."""
    def __init__(self, impath, extensions=('.jpg', '.png'), transform=None):
        super(ImageDataset, self).__init__()
        self.impath = impath
        self.extensions = extensions
        self.transform = transform
        self.filenames = self.get_valid_images()

    def get_valid_images(self):
        all_files = os.listdir(self.impath)
        mapfunc = lambda f: is_imfile(f, extensions=self.extensions)
        is_valid = map(mapfunc, all_files)
        return np.array(all_files)[is_valid]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.impath,
                                self.filenames[idx])
        image = Image.open(img_name)
        if not image.mode == 'RGB':
            image = image.convert(mode='RGB')

        if self.transform:
            image = self.transform(image)

        return image


class IndexDataset(Dataset):
    def __init__(self, data, transform=None):
        super(IndexDataset, self).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform:
            data = self.transform(data)
        return data, index


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples.
    Assume positives labels to be > 0, class specific negatives have -positive_lbl
    and negatives for all classes have 0.
    """

    def __init__(self, labels, n_labels, n_samples, n_pure_negatives=0):
        self.labels = labels
        self.labelset = np.unique(self.labels)
        self.label_to_indices = {}
        for label in self.labelset:
            lbl_idcs = np.where(self.labels == label)[0]
            np.random.shuffle(lbl_idcs)
            self.label_to_indices[label] = lbl_idcs
        self.used_label_indices_count = {label: 0 for label in self.labelset}

        self.n_labels = n_labels
        self.n_samples = n_samples
        self.n_pure_negatives = n_pure_negatives if 0 in self.labelset else 0
        self.batch_size = self.n_samples * self.n_labels + self.n_pure_negatives

        self.count = 0

    def __iter__(self):
        self.count = 0

        while self.count + (self.n_labels/2) * self.n_samples <= (self.labels > 0).sum():
            # choose the labelset
            positive_labels = self.labelset[self.labelset > 0]
            labelset = []
            while len(labelset) < self.n_labels:
                available_labels = np.setdiff1d(positive_labels, labelset)
                if len(available_labels) > 0:
                    pos_lbl = np.random.choice(available_labels)
                else:
                    pos_lbl = np.random.choice(positive_labels)
                labelset.append(pos_lbl)
                if (len(labelset) < self.n_labels) and (-pos_lbl in self.labelset):
                    labelset.append(-pos_lbl)
            # labelset = np.random.choice(positive_labels, self.n_labels, replace=False)

            if 0 in self.labelset:          # if negatives for all classes exist, use them definitely
                labelset = np.append(labelset, [0])

            indices = []
            for label_ in labelset:
                if label_ == 0:
                    n_samples = min(self.n_pure_negatives, len(self.label_to_indices[label_]))
                else:
                    n_samples = min(self.n_samples, len(self.label_to_indices[label_]))         # allow to draw less than n_samples if there are not enough
                indices.extend(self.label_to_indices[label_][
                               self.used_label_indices_count[label_]:
                               self.used_label_indices_count[label_] + n_samples
                               ])
                self.used_label_indices_count[label_] += n_samples
                if self.used_label_indices_count[label_] + n_samples > len(self.label_to_indices[label_]):
                    np.random.shuffle(self.label_to_indices[label_])
                    self.used_label_indices_count[label_] = 0
            yield indices
            self.count += (np.array(labelset) > 0).sum() * self.n_samples

    def __len__(self):
        return int((self.labels > 0).sum() * 1.0 / (self.n_labels/2) * self.n_samples)
