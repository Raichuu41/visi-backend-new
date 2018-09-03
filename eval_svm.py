import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
from faiss_master import faiss
import warnings
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from globals import get_globals
import deepdish as dd


def generate_triplets(positives, negatives, N, n_pos_pa=1, n_neg_pp=1, seed=123,
                      consider_neighborhood=False, embedding=None, n_nn_neg_pp=1):
    np.random.seed(seed)
    neighbor_sampling = consider_neighborhood and embedding is not None
    if neighbor_sampling:
        assert np.concatenate([positives, negatives]).max() < len(embedding), 'sample index out of embedding shape'

    n_pos_pa = min(n_pos_pa, len(positives) - 1)
    n_neg_pp = min(n_neg_pp, len(negatives) - 1)
    if n_pos_pa <= 0 or n_neg_pp <= 0:
        return np.array([], dtype=long)
    N_anchors = min(int(N * 1.0/(n_neg_pp * n_pos_pa)), len(positives))
    N_tot = N_anchors * n_pos_pa * n_neg_pp

    if N != N_tot:
        warnings.warn('Too few data to generate {} triplets. Instead generate {} triplets using:\n'
                      '{} anchors, {} positives per anchor, {} negatives per positive'.format(
            N, N_tot, N_anchors, n_pos_pa, n_neg_pp
        ), RuntimeWarning)
        N = N_tot

    triplets = np.empty((N, 3), dtype=np.long)

    anchors = np.random.choice(positives, N_anchors, replace=False)
    if neighbor_sampling:
        # get the embedding neighbors for the anchors
        index = faiss.IndexFlatL2(embedding.shape[1])  # build the index
        index.add(embedding.astype('float32'))  # add vectors to the index
        _, neighbors = index.search(embedding[anchors].astype('float32'), len(embedding))

    for i, a in enumerate(anchors):
        pos = np.random.choice(np.delete(positives, np.where(positives == a)[0][0]), n_pos_pa, replace=False)

        if neighbor_sampling:       # get the nearest negatives
            nn_negatives = np.array([nghbr for nghbr in neighbors[i] if nghbr in negatives])
            n_neg_neighbors = min(len(nn_negatives) - 1, n_pos_pa * n_nn_neg_pp)
            nn_negatives = nn_negatives[:n_neg_neighbors]
            outer_negatives = np.array([n for n in negatives if not n in nn_negatives])
            n_outer_neg_pp = min(n_neg_pp - n_nn_neg_pp, len(outer_negatives) - 1)

            if n_outer_neg_pp + n_nn_neg_pp != n_neg_pp:
                n_nn_neg_pp = n_neg_pp - n_outer_neg_pp
                warnings.warn('cannot generate {} negatives. Use {} negatives from neighborhood '
                              'and {} from outside.'.format(n_neg_pp, n_nn_neg_pp, n_outer_neg_pp))

        for j, p in enumerate(pos):
            if neighbor_sampling:
                nn_neg = np.random.choice(nn_negatives, n_nn_neg_pp, replace=False)
                neg = np.random.choice(outer_negatives, n_outer_neg_pp, replace=False)
                neg = np.concatenate([nn_neg, neg])
            else:
                neg = np.random.choice(negatives, n_neg_pp, replace=False)
            t = np.stack([np.repeat(a, n_neg_pp), np.repeat(p, n_neg_pp), neg], axis=1)
            i_start = (i * n_pos_pa + j) * n_neg_pp
            triplets[i_start:i_start + n_neg_pp] = t

    return triplets

plt.ion()
reset = True           # reset evaluation per labeled samples file
gte_fig = None
db_fig = None
globals = get_globals()


def make_evaluation(local_idcs, train_idcs, predictions, ground_truth,
                    d_decision_boundary=None, embedding=None, plot_GTE=False):
    global reset, gte_fig, db_fig

    test_idcs = np.setdiff1d(local_idcs, train_idcs)

    # compute precision, recall, true negative rate
    (tn_rate_train, prec_train), (_, recall_train), _, _ = precision_recall_fscore_support(ground_truth[train_idcs],
                                                                                           predictions[train_idcs])
    (tn_rate_test, prec_test), (_, recall_test), _, _ = precision_recall_fscore_support(ground_truth[test_idcs],
                                                                                        predictions[test_idcs])

    train_acc = np.sum(predictions[train_idcs] == ground_truth[train_idcs]) * 1.0 / len(train_idcs)
    test_acc = np.sum(predictions[test_idcs] == ground_truth[test_idcs]) * 1.0 / len(test_idcs)
    print('Train SVM: '
          '\n\taccuracy: {:2.1f}%'
          '\n\tprecision: {:.3f}'
          '\n\trecall: {:.3f}'
          '\n\ttn_rate: {:.3f}'
          '\nTest SVM: '
          '\n\taccuracy: {:2.1f}%'
          '\n\tprecision: {:.3f}'
          '\n\trecall: {:.3f}'
          '\n\ttn_rate: {:.3f}'
          .format(100 * train_acc, prec_train, recall_train, tn_rate_train,
                  100 * test_acc, prec_test, recall_test, tn_rate_test))

    if d_decision_boundary is not None:
        if db_fig is None:
            db_fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax = db_fig.axes
        for a in ax:
            a.clear()

        d_test = d_decision_boundary[test_idcs]
        corrects_test = predictions[test_idcs] == ground_truth[test_idcs]

        d_train = d_decision_boundary[train_idcs]
        corrects_train = predictions[train_idcs] == ground_truth[train_idcs]

        # Plot decision boundary accuracy
        ax[0].set_title('test')
        ax[0].plot(d_test[corrects_test],
                   np.zeros(sum(corrects_test)),
                   c='g', linewidth=0.0, marker='o', alpha=0.1)
        ax[0].plot(d_test[np.logical_not(corrects_test)],
                   np.zeros(sum(np.logical_not(corrects_test))),
                   c='r', linewidth=0.0, marker='o', alpha=0.1)

        # train
        ax[1].set_title('train')
        ax[1].plot(d_train[corrects_train],
                   np.zeros(sum(corrects_train)),
                   c='g', linewidth=0.0, marker='o', alpha=0.1)
        ax[1].plot(d_train[np.logical_not(corrects_train)],
                   np.zeros(sum(np.logical_not(corrects_train))),
                   c='r', linewidth=0.0, marker='o', alpha=0.1)

        plt.pause(5)

    if embedding is not None:
        N_test_triplets = 200
        positives = test_idcs[predictions[test_idcs] == 1]
        negatives = test_idcs[predictions[test_idcs] == 0]
        test_triplets = generate_triplets(positives, negatives,
                                          n_pos_pa=2, N=N_test_triplets, seed=123)

        # evaluate GTE
        GTE = 0.0
        for (a, p, n) in test_triplets:
            d_ap = np.linalg.norm(a-p)
            d_an = np.linalg.norm(a-n)
            if d_ap >= d_an:
                GTE += 1
        GTE = GTE / len(test_triplets)
        print('GTE: {}'.format(GTE))
    else:
        GTE = float('nan')

    # Write to evaluation file
    f = open('_eval.csv', 'wb') if reset else open('_eval.csv', 'a')
    outdict = {'n_labeled': len(train_idcs), 'test_acc': test_acc, 'GTE': GTE}
    writer = csv.DictWriter(f, fieldnames=outdict.keys())
    if reset:
        writer.writeheader()
    writer.writerow(outdict)
    f.close()

    if plot_GTE:
        # plot result
        n_labeled = []
        test_acc = []
        GTE = []
        with open('_eval.csv', 'rb') as f:
            reader = csv.DictReader(f, fieldnames=outdict.keys())
            next(reader, None)  # skip the headers
            for row in reader:
                n_labeled.append(int(row['n_labeled']))
                test_acc.append(float(row['test_acc']))
                GTE.append(float(row['GTE']))

        if gte_fig is None:
            gte_fig, ax = plt.subplots(1, 2)
        ax = gte_fig.axes
        for a in ax:
            a.clear()
        ax[0].set_title('test acc')
        ax[0].plot(n_labeled, test_acc)
        ax[1].set_title('GTE')
        n_labeled = np.array(n_labeled)
        GTE = np.array(GTE)
        ax[1].plot(n_labeled[np.isfinite(GTE)], GTE[np.isfinite(GTE)])
        plt.pause(5)

    reset = False


def find_svm_gt(positives_labeled, negatives_labeled, labels):
    print('Called find_svm_gt.')
    max_occurences = 0
    main_lbl = None
    svm_ground_truth = None
    for category_labels in labels.transpose():
        lbls = category_labels[positives_labeled]
        neg_lbls = category_labels[negatives_labeled]
        if len(lbls) == 0:
            continue
        lbl, occurences = sorted(Counter(lbls).items(), key=lambda x: x[1])[-1]         # choose label that occurs most often
        if occurences > max_occurences and lbl not in neg_lbls:
            max_occurences = occurences
            main_lbl = lbl
            svm_ground_truth = category_labels == main_lbl
    print('\tsvm ground truth was found to be "{}"'.format(main_lbl))
    return main_lbl, svm_ground_truth


def evaluate(local_idcs, train_idcs, predictions,
             d_decision_boundary=None, embedding=None, plot_GTE=False):
    global globals
    positives = train_idcs[predictions[train_idcs] == 1]
    negatives = train_idcs[predictions[train_idcs] == 0]
    main_label, svm_ground_truth = find_svm_gt(positives, negatives, globals.labels)
    if main_label is not None:
        make_evaluation(local_idcs, train_idcs, predictions, svm_ground_truth,
                        d_decision_boundary, embedding, plot_GTE)



def label_file_statistics(label_file, info_file, category, used_labels):
    gt = dd.io.load(info_file)['df'][category].values
    with open(label_file, 'rb') as f:
        data = pickle.load(f)
    image_names = data['image_name']
    assert np.all(image_names == dd.io.load(info_file)['df']['image_id'].values), 'image names do not match'
    labels = data['labels']
    precision, recall = [], []
    for l in used_labels:
        (_, prec), (_, rec), _, _ = precision_recall_fscore_support(gt==l, labels==l)
        precision.append(prec)
        recall.append(rec)
    precision = np.mean(precision)
    recall = np.mean(recall)
    frac_labeled = np.sum(labels != None) * 1.0 / np.sum(np.isin(gt, used_labels))

    print('Label file statistics:\n\tavg precision: {}\n\tavg recall: {}\n\tfrac labeled: {}'
          .format(precision, recall, frac_labeled))