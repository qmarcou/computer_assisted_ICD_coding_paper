from sklearn import metrics as skmet
import numpy as np
import pandas as pd
# import warnings
# warnings.filterwarnings(
#     'ignore', 'No positive class found in y_true, recall is set to one for all thresholds.', )
# warnings.filterwarnings(
#     'ignore', 'No positive samples in y_true, true positive value should be meaningless')


def per_sample_auc_PR(y_true, probas_pred, pos_label=None):
    prec = {}
    rec = {}
    thresh = {}
    auc = {}
    y_true = np.asarray(y_true)
    probas_pred = np.asarray(probas_pred)
    for row in range(0, probas_pred.shape[0]):
        if y_true[row, :].any():
            prec[row], rec[row], thresh[row] = skmet.precision_recall_curve(
                y_true=y_true[row, :],
                probas_pred=probas_pred[row, :],
                pos_label=pos_label
            )
            auc[row] = skmet.auc(rec[row], prec[row])
        else:
            # Return NaN if no positive example
            auc[row] = np.nan
    return np.array(list(auc.values()))


def per_sample_f1(y_true, y_pred, *args, **kwargs):
    sample_f1 = skmet.f1_score(y_true=y_true.transpose(),
                               y_pred=y_pred.transpose(),
                               average=None,
                               zero_division='nan',
                               *args, **kwargs)
    return sample_f1


def per_sample_average_precision(y_true, y_score, *args, **kwargs):
    sample_AP = skmet.average_precision_score(y_true=y_true.transpose(),
                                              y_score=y_score.transpose(),
                                              average=None,
                                              *args, **kwargs)
    # Assign NaN to samples with 0 positive example
    sample_AP[~y_true.any(axis=1)] = np.nan
    return sample_AP


def per_sample_AUROC(y_true, y_score, **kwargs):
    fpr = {}
    tpr = {}
    thresh = {}
    auc = {}
    y_true = np.asarray(y_true)
    probas_pred = np.asarray(y_score)
    for row in range(0, probas_pred.shape[0]):
        if y_true[row, :].any():
            fpr[row], tpr[row], thresh[row] = skmet.roc_curve(
                y_true=y_true[row, :],
                y_score=probas_pred[row, :],
                **kwargs
            )
            auc[row] = skmet.auc(fpr[row], tpr[row])
        else:
            auc[row] = np.nan
    return np.array(list(auc.values()))


def per_sample_metric(metric_fn, y_true, y_score, *args, **kwargs):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    sample_metric = np.zeros(y_score.shape[0])
    for row in range(0, y_score.shape[0]):
        if y_true[row, :].any():
            sample_metric[row] = metric_fn(
                y_true[row:row+1, :], y_score[row:row+1, :], *args, **kwargs)
        else:
            sample_metric[row] = np.nan
    return sample_metric


def per_label_metric(metric_fn, y_true, y_score, *args, **kwargs):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    label_metric = np.zeros(y_score.shape[1])
    for col in range(0, y_score.shape[0]):
        if y_true[:, col].any():
            label_metric[col] = metric_fn(
                y_true[:, col], y_score[:, col], *args, **kwargs)
        else:
            label_metric[col] = np.nan
    return label_metric


def per_sample_LRAP(y_true, y_score, *args, **kwargs):
    return per_sample_metric(skmet.label_ranking_average_precision_score,
                             y_true, y_score, *args, **kwargs)


def per_sample_NDCG(y_true, y_score, *args, **kwargs):
    return per_sample_metric(skmet.ndcg_score, y_true,
                             y_score, *args, **kwargs)


def metric_averaging(metric_fn,
                     y_true, y_score,
                     nan_policy='ignore',
                     *args, **kwargs):
    metric = metric_fn(y_true, y_score,
                       *args, **kwargs)
    if nan_policy == 'ignore':
        return np.nanmean(metric)
    elif (nan_policy == 'nan') | (nan_policy == 'error'):
        if nan_policy == 'error':
            if np.isnan(metric).any():
                raise RuntimeError('NaN values encoutered during averaging')
        return np.mean(metric)
    else:
        raise ValueError(f'Unknown nan_policy value: {nan_policy}')


def sample_averaged_auc_PR(y_true, probas_pred, pos_label=None,
                           nan_policy='ignore'):
    return metric_averaging(
        metric_fn=per_sample_auc_PR,
        y_true=y_true,
        y_score=probas_pred,
        nan_policy=nan_policy,
        pos_label=pos_label
    )


def sample_averaged_AUROC(y_true, probas_pred, pos_label=None,
                          nan_policy='ignore'):
    return metric_averaging(metric_fn=per_sample_AUROC,
                            y_true=y_true,
                            y_score=probas_pred,
                            nan_policy=nan_policy,
                            pos_label=pos_label)


def micro_averaged_auc_PR(y_true, probas_pred, pos_label=None):
    y_true = np.asarray(y_true)
    probas_pred = np.asarray(probas_pred)
    prec, rec, thresh = skmet.precision_recall_curve(
        y_true=y_true.ravel(),
        probas_pred=probas_pred.ravel(),
        pos_label=pos_label)
    return skmet.auc(rec, prec)


def per_code_auc_PR(y_true, probas_pred, pos_label=None):
    prec = {}
    rec = {}
    thresh = {}
    auc = {}
    y_true = np.asarray(y_true)
    probas_pred = np.asarray(probas_pred)
    for col in range(0, probas_pred.shape[1]):
        if y_true[:, col].any():
            prec[col], rec[col], thresh[col] = skmet.precision_recall_curve(
                y_true=y_true[:, col],
                probas_pred=probas_pred[:, col],
                pos_label=pos_label)
            auc[col] = skmet.auc(rec[col], prec[col])
        else:
            auc[col] = np.nan
    return np.array(list(auc.values()))


def macro_averaged_auc_PR(y_true, probas_pred, pos_label=None,
                          nan_policy='ignore'):
    return metric_averaging(metric_fn=per_code_auc_PR,
                            y_true=y_true,
                            y_score=probas_pred,
                            nan_policy=nan_policy,
                            pos_label=pos_label)


def per_code_AUROC(y_true, probas_pred, pos_label=None):
    fpr = {}
    tpr = {}
    thresh = {}
    auc = {}
    y_true = np.asarray(y_true)
    probas_pred = np.asarray(probas_pred)
    for col in range(0, probas_pred.shape[1]):
        if y_true[:, col].any():
            fpr[col], tpr[col], thresh[col] = skmet.roc_curve(
                y_true=y_true[:, col],
                y_score=probas_pred[:, col],
                pos_label=pos_label)
            auc[col] = skmet.auc(fpr[col], tpr[col])
        else:
            auc[col] = np.nan
    return np.array(list(auc.values()))


def macro_averaged_AUROC(y_true, probas_pred, pos_label=None,
                         nan_policy='ignore'):
    return metric_averaging(metric_fn=per_code_AUROC,
                            y_true=y_true,
                            y_score=probas_pred,
                            nan_policy=nan_policy,
                            pos_label=pos_label)


def filter_no_positive_sample(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    not_all_neg = y_true.any(axis=1)
    return y_true[not_all_neg, :], y_score[not_all_neg, :]


def filter_no_positive_label(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    not_all_neg = y_true.any(axis=0)
    return y_true[:, not_all_neg], y_score[:, not_all_neg]


def one_error(y_true, y_score, pos_label=1):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    top_pred = np.full_like(y_score, fill_value=np.nan)
    top_pred[np.arange(len(y_score)), y_score.argmax(1)] = 1
    one_error = (y_true == top_pred).sum(axis=1)
    return one_error


def average_relevance_position(y_true, y_score, pos_label=1):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    ranks = (-y_score).argsort(axis=1).argsort(axis=1).astype(float)
    # print(y_score[1,ranks.astype(int)[1,0:15]])
    ranks[y_true != pos_label] = np.nan
    # print(ranks[0,:][~np.isnan(ranks[0,:])])
    return np.nanmean(ranks+1.0, axis=1)


def sample_averaged_ARP(y_true, y_score, *args, **kwargs):
    return average_relevance_position(y_true, y_score,
                                      *args, **kwargs).nanmean()


def precision_recall_at_K(y_true, y_score, k, pos_label=1,
                          rectified_precision=True,):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true) == pos_label
    arg_sorted_scores = (-y_score).argsort(axis=1)
    sorted_truth = np.take_along_axis(y_true, arg_sorted_scores, axis=1)
    summed_truth = np.cumsum(sorted_truth, axis=1, dtype=np.float64)
    n_codes = np.sum(y_true, axis=1).reshape([-1, 1])
    n_codes = np.where(n_codes == 0, np.nan, n_codes)
    recall = np.divide(summed_truth, n_codes)
    n_codes_seen = np.arange(
        1, y_true.shape[1]+1, 1, dtype=np.float32).reshape([1, -1])
    if rectified_precision:
        def get_last_non_zero_per_row(x):
            return np.where(np.count_nonzero(x, axis=1) == 0,
                            np.nan,
                            (x.shape[1]-1) - np.argmin(x[:, ::-1] == 0,
                                                       axis=1))
        last_relevant_rank = get_last_non_zero_per_row(
            sorted_truth).reshape([-1, 1])+1
        precision = summed_truth / \
            np.where(last_relevant_rank <= n_codes_seen,
                     last_relevant_rank, n_codes_seen)
    else:
        precision = summed_truth/n_codes_seen
    if k is None:
        return (precision, recall)
    else:
        return (precision[:, k], recall[:, k])


def rank_errors_at_K(y_true, y_score, k, pos_label=1):
    y_score = np.asarray(y_score)
    y_false = np.asarray(y_true) != pos_label
    arg_sorted_scores = (-y_score).argsort(axis=1)
    sorted_errors = np.take_along_axis(y_false, arg_sorted_scores, axis=1)
    rank_errors = np.cumsum(sorted_errors, axis=1, dtype=np.float64)
    return rank_errors


def precision_at_recall(recall, precision_at_k, recall_at_k):
    if not isinstance(recall, float) and recall > 1.0 or recall < 0.0:
        return ValueError(
            'recall value must be a numeric value between 0 and 1.'
            )
    # Get the index of first rank achieving the recall value
    indices = np.argmax(recall_at_k >= recall, axis=1)
    # Get the precision a the given recall
    p_at_r = np.squeeze(np.take_along_axis(arr=precision_at_k,
                                           indices=indices.reshape([-1, 1]),
                                           axis=1))
    p_at_r[np.any(np.isnan(recall_at_k), axis=1)] = np.nan
    return p_at_r


def get_positive_ranks(y_true, y_score, pos_label=1):
    y_score_arr = np.asarray(y_score)
    y_true_arr = np.asarray(y_true) == pos_label
    ranks = (-y_score_arr).argsort(axis=1).argsort(axis=1) + \
        1  # guarantees all ranks>0
    pos_ind = np.nonzero(y_true_arr*ranks)
    ranks = ranks[pos_ind]

    if isinstance(y_score, pd.DataFrame):
        return pd.DataFrame({'index': y_score.index[pos_ind[0]],
                             "column": y_score.columns[pos_ind[1]],
                             "prediction_rank": ranks
                             })
    elif isinstance(y_true, pd.DataFrame):
        return pd.DataFrame({'index': y_true.index[pos_ind[0]],
                             "column": y_true.columns[pos_ind[1]],
                             "prediction_rank": ranks
                             })
    else:
        return pd.DataFrame({'row_id': y_score.index[pos_ind[0]],
                             "col_id": y_score.columns[pos_ind[1]],
                             "prediction_rank": ranks
                             })


def compute_predictions_ranks(y_score):
    y_score_arr = np.asarray(y_score)
    ranks = (-y_score_arr).argsort(axis=1).argsort(axis=1) + \
        1  # guarantees all ranks>0
    if isinstance(y_score, pd.DataFrame):
        ranks = pd.DataFrame(
            ranks, columns=y_score.columns, index=y_score.index)
    return ranks


def dummy_ranker(y_true_learn, y_true_test, pos_label=1):
    y_true_learn = np.asarray(y_true_learn)
    freq = (y_true_learn == pos_label).mean(axis=0)
    freq = freq.reshape((1, -1))
    return np.concatenate([freq]*y_true_test.shape[0], axis=0)
