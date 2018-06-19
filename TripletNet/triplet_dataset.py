import numpy as np
from torch.utils.data.sampler import BatchSampler
from wikiart_dataset import Wikiart
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def make_iterable(obj):
    if obj is None:
        return []
    else:
        if hasattr(obj, '__iter__'):
            return obj
        else:
            return [obj]


class TripletBatchSampler(BatchSampler):
    """
    BatchSampler - Samples n_classes and within these classes samples n_samples.
    Allows to add pure negatives for which exact class in unknown.
    Also allows to skip classes of a certain label completely.
    Returns batches of size n_classes * n_samples + n_pure_negatives.
    """

    def __init__(self, labels, triplet_configurations, n_samples_per_class,
                 pure_negative_class=None, n_pure_negatives=0):
        self.n_classes_per_batch = n_classes_per_batch
        self.n_samples_per_class = n_samples_per_class
        self.pure_negative_class = make_iterable(pure_negative_class)
        self.skip_class_label = make_iterable(skip_class_label)
        self.labels = labels
        self.pure_negative_label = -1 if self.labels.dtype == int else 'unknown'
        for pnl in self.pure_negative_label:
            self.labels[np.where(self.labels==pnl)[0]] = self.pure_negative_label


        self.n_concealed = n_concealed if n_concealed is not None else 0
        self.frac_noise = frac_noise
        self.skip_class = skip_class
        self.labels = dataset.df[selected_class].values.copy()
        self.concealed_label = -1 if self.labels.dtype == int else 'concealed'
        for cc in self.concealed_classes:
            self.labels[np.where(self.labels == cc)[0]] = self.concealed_label
        self.labels_set = np.unique(self.labels)
        if self.skip_class is not None:
            del_idx = np.where(self.labels_set == str(self.skip_class))[0]
            if len(del_idx) > 0:
                self.labels_set = np.delete(self.labels_set, del_idx[0])
        if frac_noise > 0:
            for label in self.labels_set:
                idcs = np.where(self.labels == label)[0]
                n_noisy = np.ceil(len(idcs) * self.frac_noise).astype(int)
                idcs = np.random.choice(idcs, n_noisy, replace=False)
                self.labels[idcs] = self.concealed_label
        print('Total amount of artificial noise in data {:.1f}%.'
              .format(np.sum(self.labels == self.concealed_label) * 100. / len(self.labels)))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(np.where(self.labels != self.concealed_label)[0]):
            classes = np.random.choice([l for l in self.labels_set if l != self.concealed_label],
                                       self.n_classes, replace=False)
            indices = []
            class_iter = classes if len(self.concealed_classes) == 0 else np.append(classes, self.concealed_label)
            for class_ in class_iter:
                n_samples = self.n_concealed if class_ == self.concealed_label else self.n_samples
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + n_samples])
                self.used_label_indices_count[class_] += n_samples
                if self.used_label_indices_count[class_] + n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(np.where(self.labels != self.concealed_label)[0]) // self.batch_size