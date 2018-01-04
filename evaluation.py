import numpy as np


def _compute_apk(targets, predictions, k):

    if len(predictions) > k:
        predictions = predictions[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predictions):
        if p in targets and p not in predictions[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not list(targets):
        return 0.0

    return score / min(len(targets), k)


def _compute_precision_recall(targets, predictions, k):

    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    precision = float(num_hit) / len(pred)
    recall = float(num_hit) / len(targets)
    return precision, recall


def evaluate_ranking(model, test, train=None, k=10):
    """
    Compute Precision@k, Recall@k scores and average precision (AP).
    One score is given for every user with interactions in the test
    set, representing the AP, Precision@k and Recall@k of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, rated items in
        interactions will be excluded.
    k: int or array of int,
        The maximum number of predicted items
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if not isinstance(k, list):
        ks = [k]
    else:
        ks = k

    precisions = [list() for _ in range(len(ks))]
    recalls = [list() for _ in range(len(ks))]
    apks = list()

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)
        predictions = predictions.argsort()

        if train is not None:
            rated = set(train[user_id].indices)
        else:
            rated = []

        predictions = [p for p in predictions if p not in rated]

        targets = row.indices

        for i, _k in enumerate(ks):
            precision, recall = _compute_precision_recall(targets, predictions, _k)
            precisions[i].append(precision)
            recalls[i].append(recall)

        apks.append(_compute_apk(targets, predictions, k=np.inf))

    precisions = [np.array(i) for i in precisions]
    recalls = [np.array(i) for i in recalls]

    if not isinstance(k, list):
        precisions = precisions[0]
        recalls = recalls[0]

    mean_aps = np.mean(apks)

    return precisions, recalls, mean_aps

