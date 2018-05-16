import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

sys.path.append('/export/home/kschwarz/Documents/Masters/Modify_TSNE/')
from modify_snack import snack_embed_mod

def get_multilabel(svms, features):
    multilabel = np.empty((features.shape[0], len(svms)), dtype=bool)
    # make multilabel
    for i, clf in enumerate(svms):
        predicted_labels = clf.predict(features)
        multilabel[:, i] = predicted_labels
    return multilabel


def multiclass_embedding_rf(svms, features, seed=123):
    labels = get_multilabel(svms, features)         # TODO: consider only using support vectors! Or using sample weights with decision function
    # introduce an outlier class
    outlier_class = np.where(labels.sum(axis=1) == 0, np.ones(labels.shape[0]), np.zeros(labels.shape[0]))
    labels = np.concatenate([labels, outlier_class.reshape(-1, 1)], axis=1)

    rfclf = RandomForestClassifier(n_estimators=5, criterion='entropy', class_weight='balanced', random_state=seed)
    clf = OneVsRestClassifier(rfclf)

    clf.fit(features, labels)
    scores = clf.predict_proba(features)

    tsne = snack_embed_mod
    kwargs = {'contrib_cost_tsne': 100, 'contrib_cost_triplets': 0.1, 'contrib_cost_position': 1.0,
              'perplexity': 5, 'theta': 0.5, 'no_dims': 2}  # kwargs for embedding_func
    embedding = tsne(scores.astype(np.double),
                     triplets=np.zeros((1, 3), dtype=np.long), weights_triplets=None,
                     position_constraints=np.zeros((1, 3)),
                     **kwargs)

    return embedding


def make_feature_vec(svms, features):
    pred_prob = []
    for clf in svms:
        pred_prob.append(clf.predict_proba(features)[:, 1])            # only use positive scores / rest is duplicates
    pred_prob = np.stack(pred_prob).reshape(features.shape[0], len(svms))
    return pred_prob


def multiclass_embedding(svms, features):
    scores = make_feature_vec(svms, features)
    tsne = snack_embed_mod
    kwargs = {'contrib_cost_tsne': 100, 'contrib_cost_triplets': 0.1, 'contrib_cost_position': 1.0,
              'perplexity': 30, 'theta': 0.5, 'no_dims': 2}  # kwargs for embedding_func
    embedding = tsne(scores.astype(np.double),
                     triplets=np.zeros((1, 3), dtype=np.long), weights_triplets=None,
                     position_constraints=np.zeros((1, 3)),
                     **kwargs)
    return embedding