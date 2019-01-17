import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import BatchSampler, WeightedRandomSampler
import math
from collections import Counter


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


# class PartiallyLabeledBatchSampler(BatchSampler):
#     def __init__(self, labels, frac_labeled, batch_size, N_unlabeled_total=0):
#         """
#         Create batches from partially labeled data. Batches are created such that each labeled sample is used at least
#         once.
#         :param labels: unlabeled samples have to have label 0
#         :param frac_labeled: percentage of labeled samples in the batch
#         :param batch_size:
#         :param N_unlabeled_total: minimum number of used unlabeled samples
#         :param classweights: dict of form: {l0: w0, l1: w1, ...}
#         """
#         self.labels = labels.numpy().astype(np.long) if isinstance(labels, torch.LongTensor) \
#             else np.array(labels).astype(long)
#         islabeled = self.labels != 0
#         self.idcs_labeled = np.where(islabeled)[0]
#         self.idcs_unlabeled = np.where(islabeled.__invert__())[0]
#
#         np.random.shuffle(self.idcs_labeled)
#         np.random.shuffle(self.idcs_unlabeled)
#
#         self.N_labeled = len(self.idcs_labeled)
#         self.N_unlabeled = len(self.idcs_unlabeled)
#
#         self.frac_labeled = frac_labeled
#         self.batch_size = batch_size
#         self.N_unlabeled_total = N_unlabeled_total
#
#         self.N_labeled_batch = min(self.N_labeled, int(math.ceil(self.frac_labeled * self.batch_size)))
#         self.N_unlabeled_batch = min(self.N_unlabeled, self.batch_size - self.N_labeled_batch)
#
#         self.count = 0
#
#     def __len__(self):
#         N_from_labeled = int(math.ceil(self.N_labeled * 1.0 / self.N_labeled_batch))
#         N_from_unlabeled_total = int(math.ceil(self.N_unlabeled_total * 1.0 / self.N_unlabeled_batch))
#         return max(N_from_labeled, N_from_unlabeled_total)
#
#     def __iter__(self):
#         self.count = 0
#         count_used_labeled = 0
#         count_used_unlabeled = 0
#
#         while self.count < len(self):
#             indices = []
#             indices.extend(self.idcs_labeled[count_used_labeled:count_used_labeled + self.N_labeled_batch])
#             indices.extend(self.idcs_unlabeled[count_used_unlabeled:count_used_unlabeled + self.N_unlabeled_batch])
#
#             count_used_labeled += self.N_labeled_batch
#             count_used_unlabeled += self.N_unlabeled_batch
#
#             if count_used_labeled + self.N_labeled_batch > self.N_labeled:
#                 np.random.shuffle(self.idcs_labeled)
#                 count_used_labeled = 0
#             if count_used_unlabeled + self.N_unlabeled_batch > self.N_unlabeled:
#                 np.random.shuffle(self.idcs_unlabeled)
#                 count_used_unlabeled = 0
#             yield indices
#             self.count += 1


class PartiallyLabeledBatchSampler(BatchSampler):
    def __init__(self, labels, frac_labeled, batch_size, N_unlabeled_total=0, classweights=None):
        """
        Create batches from partially labeled data. Batches are created such that each labeled sample is used at least
        once.
        :param labels: unlabeled samples have to have label 0
        :param frac_labeled: percentage of labeled samples in the batch
        :param batch_size:
        :param N_unlabeled_total: minimum number of used unlabeled samples
        :param classweights: dict of form: {l0: w0, l1: w1, ...}
            if classweights are provided batch size may vary due to uniqueness in batch but weighted sampling
        """
        self.labels = labels if isinstance(labels, torch.LongTensor) \
            else torch.LongTensor(labels)
        self.idcs_labeled = torch.nonzero(self.labels).view(-1)
        self.idcs_unlabeled = torch.nonzero(self.labels == 0).view(-1)

        self.N_labeled = len(self.idcs_labeled)
        self.N_unlabeled = len(self.idcs_unlabeled)

        self.frac_labeled = frac_labeled
        self.batch_size = batch_size
        self.N_unlabeled_total = N_unlabeled_total

        self.N_labeled_batch = min(self.N_labeled, int(math.ceil(self.frac_labeled * self.batch_size)))
        self.N_unlabeled_batch = min(self.N_unlabeled, self.batch_size - self.N_labeled_batch)

        if classweights is not None:
            # transform classweights to sampleweights
            #   P_c = sum_{i=1}^{N_c} p_i^c --> p_i^c = P_c / N_c
            label_counter = Counter(self.labels.numpy())
            map_fn = lambda x: classweights[x] * 1.0 / label_counter[x]
            sampleweights = np.array(map(map_fn, self.labels[self.idcs_labeled].numpy()))
        else:
            sampleweights = np.ones(self.N_labeled)

        self.sampler_labeled = WeightedRandomSampler(weights=sampleweights, num_samples=self.N_labeled,
                                                     replacement=classweights is not None)      # use all if no classweights are given

        self.count = 0

    def __len__(self):
        N_from_labeled = int(math.ceil(self.N_labeled * 1.0 / self.N_labeled_batch))
        N_from_unlabeled_total = int(math.ceil(self.N_unlabeled_total * 1.0 / self.N_unlabeled_batch))
        return max(N_from_labeled, N_from_unlabeled_total)

    def __iter__(self):
        self.count = 0
        count_used_labeled = 0
        count_used_unlabeled = 0

        idcs_labeled = self.idcs_labeled[torch.stack(list(self.sampler_labeled))]
        self.idcs_unlabeled = self.idcs_unlabeled[torch.randperm(self.N_unlabeled)]

        while self.count < len(self):
            indices = []
            indices.extend(idcs_labeled[count_used_labeled:count_used_labeled + self.N_labeled_batch].numpy())
            indices.extend(self.idcs_unlabeled[count_used_unlabeled:count_used_unlabeled + self.N_unlabeled_batch].numpy())

            count_used_labeled += self.N_labeled_batch
            count_used_unlabeled += self.N_unlabeled_batch

            if count_used_labeled + self.N_labeled_batch > self.N_labeled:
                idcs_labeled = self.idcs_labeled[torch.stack(list(self.sampler_labeled))]
                count_used_labeled = 0
            if count_used_unlabeled + self.N_unlabeled_batch > self.N_unlabeled:
                self.idcs_unlabeled = self.idcs_unlabeled[torch.randperm(self.N_unlabeled)]
                count_used_unlabeled = 0
            yield list(np.unique(indices))
            self.count += 1



