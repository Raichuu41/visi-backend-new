import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
from faiss_master import faiss
import warnings
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter


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


def evaluate(ground_truth, plot_decision_boundary=True, plot_GTE=True, compute_GTE=True, eval_local=True):
    global reset, gte_fig, db_fig
    with open('_svm_prediction.pkl', 'rb') as f:
        svm_data = pickle.load(f)
        predictions = svm_data['labels']
        distances = svm_data['distance']
        local_idcs = svm_data['local_indices'] if eval_local else np.arange(len(predictions)).astype(int)
        train_idcs = np.concatenate([svm_data['idcs_positives_train'], svm_data['idcs_negatives_train']])
        if compute_GTE:
            local_embedding = svm_data['local_embedding']
            local_triplets = svm_data['local_triplets']

    test_idcs = np.array([idx for idx in range(len(ground_truth)) if (idx in local_idcs and idx not in train_idcs)], dtype=int)

    # evaluate precision, recall and true negative rate
    # training
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

    if plot_decision_boundary:
        if db_fig is None:
            db_fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax = db_fig.axes
        for a in ax:
            a.clear()

        # Plot decision boundary accuracy
        ax[0].set_title('test')
        ax[0].plot(distances[test_idcs][predictions[test_idcs] == ground_truth[test_idcs]],
                   np.zeros(len(distances[test_idcs][predictions[test_idcs] == ground_truth[test_idcs]])),
                   c='g', linewidth=0.0, marker='o', alpha=0.1)
        ax[0].plot(distances[test_idcs][predictions[test_idcs] != ground_truth[test_idcs]],
                   np.zeros(len(distances[test_idcs][predictions[test_idcs] != ground_truth[test_idcs]])),
                   c='r', linewidth=0.0, marker='o', alpha=0.1)

        # train
        ax[1].set_title('train')
        ax[1].plot(distances[train_idcs][predictions[train_idcs] == ground_truth[train_idcs]],
                   np.zeros(len(distances[train_idcs][predictions[train_idcs] == ground_truth[train_idcs]])),
                   c='g', linewidth=0.0, marker='o', alpha=0.1)
        ax[1].plot(distances[train_idcs][predictions[train_idcs] != ground_truth[train_idcs]],
                   np.zeros(len(distances[train_idcs][predictions[train_idcs] != ground_truth[train_idcs]])),
                   c='r', linewidth=0.0, marker='o', alpha=0.1)

        # all images
        ax[2].set_title('global')
        ax[2].plot(distances[predictions == ground_truth],
                   np.zeros(len(distances[predictions == ground_truth])),
                   c='g', linewidth=0.0, marker='o', alpha=0.1)
        ax[2].plot(distances[predictions != ground_truth],
                   np.zeros(len(distances[predictions != ground_truth])),
                   c='r', linewidth=0.0, marker='o', alpha=0.1)

        plt.pause(8)

    if compute_GTE:
        # evaluate triplet error in embedding
        positives = np.where(ground_truth[test_idcs] == 1)[0]
        negatives = np.where(ground_truth[test_idcs] == 0)[0]

        N_test_triplets = 200
        test_triplets = generate_triplets(positives, negatives, n_pos_pa=2, N=N_test_triplets, seed=234)
        for i, t in enumerate(test_triplets):
            if any([(t == lt).all() for lt in local_triplets]):
                print('duplicate')
                accept = False
                while not accept:
                    print('try new t')
                    t_new = generate_triplets(positives, negatives, N=1, seed=345+i)[0]
                    if not (any([(t_new == lt).all() for lt in test_triplets])
                        or any([(t_new == lt).all() for lt in local_triplets])):
                        accept = True
                test_triplets[i] = t_new

        # evaluate GTE
        # compute distances in local embedding
        index = faiss.IndexFlatL2(local_embedding.shape[1])   # build the index
        index.add(np.stack(local_embedding).astype('float32'))                  # add vectors to the index
        knn_distances, knn_indices = index.search(np.stack(local_embedding).astype('float32'), len(local_embedding))

        GTE = 0.0
        for (a, p, n) in test_triplets:
            if knn_distances[a, p] >= knn_distances[a, n]:
                GTE += 1
        GTE = GTE / len(test_triplets)
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
        lbl, occurences = sorted(Counter(lbls).items(), key=lambda x: x[1])[-1]         # choose label that occurs most often
        if occurences > max_occurences and lbl not in neg_lbls:
            max_occurences = occurences
            main_lbl = lbl
            svm_ground_truth = category_labels == main_lbl
    print('\tsvm ground truth was found to be "{}"'.format(main_lbl))
    return main_lbl, svm_ground_truth
