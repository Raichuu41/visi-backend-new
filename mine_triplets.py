import numpy as np
from math import factorial
import sys
sys.path.append('../SmallNets')
from triplets_utils import SemihardNegativeTripletSelector, RandomNegativeTripletSelector, \
    HardestNegativeTripletSelector, KHardestNegativeTripletSelector
from triplet_dataset import BalancedLabelBatchSampler_withNonesandNoise
import torch
from itertools import combinations
import pickle
import os
from sklearn.metrics import precision_recall_fscore_support
import deepdish as dd


def get_labels(label_file='_svm_labels.pkl', confidence_threshold=None, skip_class_label=-2):
    """Read in label file. When no label was given by user value is set to -1, if label was given, but svm value is
    below confidence threshold the value is set to -2."""

    assert os.path.isfile(label_file), 'label file not found'
    with open(label_file, 'rb') as f:
        svm_label_dict = pickle.load(f)
    svm_labels = svm_label_dict['labels']
    if svm_labels.ndim == 1:
        svm_labels = svm_labels.reshape(-1, 1)

    # deal with multiple labels
    mltpl_lbls = np.where(np.sum(svm_labels, axis=1) > 1)[0]
    if len(mltpl_lbls) > 0:
        # highest_conf = np.argmax(svm_label_dict['confidence'][mltpl_lbls], axis=1)          # choose the one with highest confidence
        # for idx, idy in zip(mltpl_lbls, highest_conf):
        #     svm_labels[idx, :] = 0
        #     svm_labels[idx, idy] = 1

        n_labels = (svm_labels == 1).sum(axis=0)            # choose the label with the least members
        for idx in mltpl_lbls:
            idcs = np.where(svm_labels[idx] == 1)[0]
            idy = idcs[np.argmin(n_labels[idcs])]
            svm_labels[idx, :] = 0
            svm_labels[idx, idy] = 1

    labels = np.argmax(svm_labels, axis=1)
    linear_idx = np.ravel_multi_index([range(len(svm_labels)), labels], svm_labels.shape)
    labels[np.where(svm_labels.flatten()[linear_idx] == 0)[0]] = -1  # no label was given by user
    num_classes = len(np.unique(labels)) - 1
    print('User created {} different labels.'.format(num_classes))
    if confidence_threshold is not None:
        labels[np.abs(svm_label_dict['confidence'].flatten()[linear_idx]) < confidence_threshold] = skip_class_label
    return labels, num_classes


def mine_triplets(triplet_selector=None, features=None, ground_truth=None, margin=0.2, cpu=False,
                  seed=123, confidence_threshold=None):
    np.random.seed(seed)
    with open('_svm_labels.pkl', 'rb') as f:
        svm_label_dict = pickle.load(f)
    svm_labels = svm_label_dict['labels']
    if svm_labels.ndim == 1:
        svm_labels = svm_labels.reshape(-1, 1)

    # deal with multiple labels
    mltpl_lbls = np.where(np.sum(svm_labels, axis=1) > 1)[0]
    if len(mltpl_lbls) > 0:
        highest_conf = np.argmax(svm_label_dict['confidence'][mltpl_lbls], axis=1)
        for idx, idy in zip(mltpl_lbls, highest_conf):
            svm_labels[idx, :] = 0
            svm_labels[idx, idy] = 1

    n_labels = (svm_labels == 1).sum(axis=0)

    # compute number of possible positive anchor pairs
    batch_size = 10
    n_ap_batch = factorial(batch_size) / (2.0 * factorial(batch_size - 2))

    n_distinct_batches = (n_labels / 10).astype(int)
    n_distinct_triplets = n_ap_batch * n_distinct_batches

    print('number of distinct batches per class (batch_size={}): \n\t{}'.format(batch_size, n_distinct_batches))
    print('number of distinct triplets per class: \n\t{}'.format(n_distinct_triplets.astype(int)))

    labels = np.argmax(svm_labels, axis=1)
    linear_idx = np.ravel_multi_index([range(len(svm_labels)), labels], svm_labels.shape)
    labels[np.where(svm_labels.flatten()[linear_idx] == 0)[0]] = -1  # no label was given by user
    num_classes = len(np.unique(labels))-1
    print('User created {} different labels.'.format(num_classes))
    skip_class = None           # exclude uncertain labels
    if confidence_threshold is not None:
        labels[np.where(svm_label_dict['confidence'] < confidence_threshold)[0]] = -2
        skip_class = -2

    sampler = BalancedLabelBatchSampler_withNonesandNoise(labels.astype(str),
                                                          n_classes=min(10, num_classes), n_samples=8,
                                                          concealed_classes=['-1'],
                                                          n_concealed=10, skip_class=skip_class)
    idx_batches = list(sampler)
    batch_triplets = []

    if features is not None and triplet_selector is not None:             # TODO: CHECK ACC FOR HARDEST TRIPLET SELECTION!
        for idx_batch in idx_batches:
            idcs = np.array(idx_batch)
            batch_labels = labels[idcs]
            batch_fts = features[torch.LongTensor(idcs)]

            if triplet_selector.lower() not in ['random', 'semihard', 'hardest', 'khardest']:
                assert False, 'Unknown option {} for triplet selector. Choose from "random", "semihard", "hardest" or "khardest"' \
                              '.'.format(triplet_selector)
            elif triplet_selector.lower() == 'random':
                triplet_selector = RandomNegativeTripletSelector(margin, cpu)
            elif triplet_selector.lower() == 'semihard':
                triplet_selector = SemihardNegativeTripletSelector(margin, cpu)
            elif triplet_selector.lower() == 'khardest':
                triplet_selector = KHardestNegativeTripletSelector(margin, cpu)
            else:
                triplet_selector = HardestNegativeTripletSelector(margin, cpu)

            batch_triplets.append(triplet_selector.get_triplets(batch_fts, batch_labels, concealed_classes=[-1]))

    else:  # generate batches and compute ap pairs with random negative
        for idx_batch in idx_batches:
            idcs = np.array(idx_batch)
            batch_labels = labels[idcs]
            triplets = []

            for label in set(batch_labels).difference([-1]):
                label_mask = (batch_labels == label)
                label_indices = np.where(label_mask)[0]
                if len(label_indices) < 2:
                    continue
                negative_indices = np.where(np.logical_not(label_mask))[0]
                anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
                anchor_positives = np.array(anchor_positives)

                for anchor_positive in anchor_positives:
                    triplets.append((anchor_positive[0], anchor_positive[1], np.random.choice(negative_indices)))
            triplets = torch.LongTensor(np.array(triplets))
            batch_triplets.append(triplets)

    acc = None
    if ground_truth is not None:            # measure accuracy of mined triplets
        acc = []
        for t_batch, idx_batch in zip(batch_triplets, idx_batches):
            gt = ground_truth[idx_batch]
            gt_triplets = gt[t_batch.view(-1).numpy()].reshape(-1, 3)
            correct_triplet = (gt_triplets[:, 0] == gt_triplets[:, 1]) * (gt_triplets[:, 0] != gt_triplets[:, 2])
            acc.append(np.sum(correct_triplet) * 1.0 / len(correct_triplet))
        acc = np.mean(acc)
        print('Mined (average) triplet accuracy: {:.1f}% correct.'.format(acc * 100))

    return batch_triplets, acc


def modify_true_label_file(true_label_file='true_labels_4largest_styles.pkl', precision=1., recall=1.,
                           n_labeled=None, seed=123):
    np.random.seed(seed)
    with open(true_label_file, 'rb') as f:
        label_dict = pickle.load(f)
    labels = label_dict['labels']
    n_labels = labels.shape[1]
    unlabeled = np.where(np.sum(labels, axis=1) == 0)[0]
    new_labels = labels.copy().astype(int)

    for i in range(n_labels):
        if n_labeled is not None:
            idcs = np.where(labels[:, i])[0]
            idcs = np.array(list(set(idcs).difference(list(np.random.choice(idcs, min(len(idcs), n_labeled), replace=False))))).astype(int)
            new_labels[idcs, i] = -1
            valid_idcs = np.array(list(set(range(len(new_labels))).difference(list(idcs))))

        if recall < 1:
            # add some false negatives
            tp = np.sum(new_labels[:, i] == 1)
            n_fn = tp * (1-recall)
            idcs = np.random.choice(np.where(new_labels[:, i] == 1)[0], int(n_fn), replace=False)
            new_labels[idcs, i] = 0

        if precision < 1:
            # add some false positives to unlabeled data points
            tp = np.sum(new_labels[:, i] == 1)
            n_fp = tp * 1.0 / precision - tp
            assert int(n_fp) <= len(unlabeled), 'not enough unlabeled points to generate false positives'
            idcs = np.random.choice(unlabeled, int(n_fp), replace=False)
            unlabeled = np.array(list(set(unlabeled).difference(idcs)))
            new_labels[idcs, i] = 1

        # check
        (_, prec), (_, rec), _, _ = precision_recall_fscore_support(labels[valid_idcs, i], new_labels[valid_idcs, i])
        print(prec, rec)

    fname = true_label_file.replace('true', 'N{}_p_{:2.0f}_r_{:2.0f}'.format(n_labeled, precision*100, recall*100))
    with open(fname, 'wb') as f:
        pickle.dump({'labels': new_labels, 'confidence': label_dict['confidence']}, f)
    print('saved {}'.format(fname))


def concat_labels(label_files=['_svm_labels_impressionism.pkl', '_svm_labels_postimpressionism.pkl',
                               '_svm_labels_realism.pkl', '_svm_labels_surrealism.pkl']):
    labels_all = []
    confidences = []
    for file in label_files:
        with open(file, 'rb') as f:
            svm_dict = pickle.load(f)
        labels = svm_dict['labels']
        c = np.ravel_multi_index([range(len(labels)), np.argmax(svm_dict['confidence'], axis=1)], labels.shape)
        confidences.append(svm_dict['confidence'].flatten()[c])
        labels = labels.max(axis=1)
        labels_all.append(labels)

    labels = np.stack(labels_all).transpose()
    confidence = np.stack(confidences).transpose()
    confidence[confidence == -1] = 0
    with open('_svm_labels_style_4largest_localSVM.pkl', 'wb') as f:
        pickle.dump({'labels': labels, 'confidence': confidence}, f)


def evaluate_labelling(label_file='_svm_labels_style_4largest_localSVM_corrected_prec_thresh_1.pkl',
                       info_file='../wikiart/datasets/info_artist_49_style_train_small.hdf5',
                       targets=['impressionism', 'post-impressionism', 'realism', 'surrealism']):
    ground_truth = dd.io.load(info_file)['df']['style'].values
    lbls, _ = get_labels(label_file, 1.0)
    gt_to_int = {l: i for i, l in enumerate(targets)}

    for k, v in gt_to_int.items():
        gt = ground_truth == k
        pred = lbls == gt_to_int[k]

        (_, prec), (_, rec), _, _ = precision_recall_fscore_support(gt.astype(int), pred.astype(int))
        print(prec, rec)


def correct_labels(label_file='_svm_labels_style_4largest_localSVM.pkl',
                   info_file='../wikiart/datasets/info_artist_49_style_train_small.hdf5',
                   targets=['impressionism', 'post-impressionism', 'realism', 'surrealism']):
    with open(label_file, 'rb') as f:
        svm_label_dict = pickle.load(f)
    labels = svm_label_dict['labels']
    ground_truth = dd.io.load(info_file)['df']['style'].values
    ground_truth_mask = np.stack([ground_truth == t for t in targets]).transpose()
    linear_idx = np.ravel_multi_index([range(len(labels)), np.argmax(labels, axis=1)], labels.shape)
    mask = labels.flatten()[linear_idx] != -1
    confidence = svm_label_dict['confidence']
    mask[np.where(confidence.flatten()[linear_idx] < 1.0)[0]] = False
    confidence[mask.__invert__(), :] = 0

    labels[mask, :] = ground_truth_mask[mask, :]
    # confidence = np.zeros(labels.shape)
    # confidence[mask, :] = 1.

    with open('_svm_labels_style_4largest_localSVM_corrected_prec_thresh_1.pkl', 'wb') as f:
        pickle.dump({'labels': labels, 'confidence': confidence}, f)






