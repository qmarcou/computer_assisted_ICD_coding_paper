"""Some utility functions to perform hot encoding operations."""
import gc
import pandas as pd
import numpy as np
import dtypecheck as dc
from scipy.sparse import isspmatrix
import warnings


def _str_get_dummies_uint8(self, sep="|"):
    from pandas import Series
    import pandas._libs.lib as lib

    arr = Series(self).fillna("")
    try:
        arr = sep + arr + sep
    except TypeError:
        arr = sep + arr.astype(str) + sep

    tags: set[str] = set()
    for ts in Series(arr).str.split(sep):
        tags.update(ts)
    tags2 = sorted(tags - {""})

    dummies = np.empty((len(arr), len(tags2)), dtype=np.uint8)

    for i, t in enumerate(tags2):
        pat = sep + t + sep
        dummies[:, i] = lib.map_infer(arr.to_numpy(), lambda x: pat in x)
    return dummies, tags2


# Monkey patch pandas function to enforce use of 8bit integers instead of 64
(pd.core.strings.object_array
 .ObjectStringArrayMixin._str_get_dummies) = _str_get_dummies_uint8


def hot_encode_df_col(df: pd.DataFrame, colname: str, by,
                      keepCols=None, sep="|") -> pd.DataFrame:
    """Hot encode a DataFrame column."""
    # TODO:
    # - deal with possible multiindex/pre grouped data
    # - possibly keep other columns
    # - check existence of sep in encoded column
    # - check for NaN in the target aggregation column (otherwise the string
    # aggregation function will collapse) (add fillna param)
    if dc.is_str_or_strlist(by):
        if isinstance(by, str):
            by = [by]
    else:
        raise TypeError("'by' must be a str or a list[str]")
    # Perform string aggregation of the column
    agg_cols = [colname] + by
    str_agg_col_df = df.loc[:, agg_cols].groupby(
        by=by, as_index=True, axis=0).agg(sep.join)

    # One hot encode using get_dummies
    oh_df = str_agg_col_df[colname].str.get_dummies(sep=sep)

    # Re index DF to drop multilevel index
    oh_df.reset_index(inplace=True)

    return oh_df


def is_oh_df(oh_df: pd.DataFrame, extracols=None) -> bool:
    """Test if the provided DF is hot encoded appart from extracols."""
    return True


def threshold_oh_abs(oh_df: pd.DataFrame, n: int, extracols: list = None,
                     geq: bool = True):
    """Threshold on the absolute number of occurrences in a OneHot DF."""
    assert is_oh_df(oh_df, extracols=extracols)
    if extracols is None:
        extracols = []
    elif dc.is_str_or_strlist(extracols):
        if isinstance(extracols, str):
            extracols = [extracols]
    else:
        raise TypeError("Invalid type for extracols")

    if geq:
        mask = (oh_df.sum() < n)
    else:
        mask = (oh_df.sum() > n)

    dropped_cols = oh_df.columns[
        np.logical_and(mask, (~oh_df.columns.isin(extracols)))]
    return oh_df.drop(columns=dropped_cols)


def threshold_oh_freq(oh_df: pd.DataFrame, f: float, extracols=None,
                      geq: bool = True):
    """Threshold on the frequency of occurrences in a One Hot DF."""
    assert (f >= 0) & (f <= 1)
    n_cut = round(f * oh_df.shape[0])
    return threshold_oh_abs(oh_df, n_cut, extracols, geq)


def threshold_oh_topN(oh_df: pd.DataFrame, n: int, extracols: list = None):
    """Threshold to keep only N columns with max number of occurrences."""
    assert is_oh_df(oh_df, extracols=extracols)
    tmp = oh_df.drop(columns=extracols).sum().sort_values(axis=0,
                                                          ascending=False,
                                                          inplace=False)
    thresh = tmp[n - 1]
    if tmp[n - 1] == tmp[max(n - 1, 0)] or tmp[n - 1] == tmp[n]:
        warnings.warn("Ties found around Nth greatest element, the function "
                      "will return extra columns with ties in"
                      "threshold_oh_topN()",
                      RuntimeWarning)
    return threshold_oh_abs(oh_df, n=thresh, extracols=extracols)


def _oh_df_to_mat(oh_df: pd.DataFrame, negate=False, freq: bool = False,
                  sparse=None, precision: int = 64):
    # TODO:
    # - check if using sparse mat when negating still gives better perfs
    if sparse is not None:
        return_sparse = sparse
    else:
        return_sparse = False

    if negate:
        oh_df = (oh_df == 0).astype(np.uint8)
        if sparse is not None:
            # Negation of a sparse object is dense and vice versa unless the
            # initial object is not very sparse
            return_sparse = not sparse

    # Interpret precision argument
    float_precision = {16: np.float16, 32: np.float32,
                       64: np.float64}
    uint_precision = {8: np.uint8, 16: np.uint16, 32: np.uint32,
                      64: np.uint64}
    if freq:
        precision_dict = float_precision
    else:
        precision_dict = uint_precision

    if return_sparse:
        oh_mat = oh_df.astype(pd.SparseDtype(
            precision_dict[precision], 0)).sparse.to_coo().tocsr()
    else:
        oh_mat = oh_df.astype(precision_dict[precision]).to_numpy()

    return oh_mat


def _compute_occurrence_mat(oh_df: pd.DataFrame, freq: bool = False,
                            extracols=None, negate=False, discordance=False,
                            sparse=None, precision: int = 64) -> np.ndarray:
    # TODO perform dot product by chunk to alleviate memory issues
    # TODO try numba JIT on this function
    assert is_oh_df(oh_df, extracols)
    if negate and discordance:
        raise ValueError("Having both negate and discordance set to True is"
                         "ambiguous.")
    if extracols is not None:
        assert dc.is_str_or_strlist(extracols)
        oh_df = oh_df.drop(columns=extracols)
    if discordance:
        sp_oh_mat = _oh_df_to_mat(oh_df, negate=True, freq=freq,
                                  sparse=sparse, precision=precision)
        trans_oh_mat = _oh_df_to_mat(oh_df, negate=False, freq=freq,
                                     sparse=sparse,
                                     precision=precision).transpose()
    else:
        sp_oh_mat = _oh_df_to_mat(oh_df, negate=negate, freq=freq,
                                  sparse=sparse, precision=precision)
        trans_oh_mat = sp_oh_mat.transpose()

    # Compute dot product of the two matrices
    if discordance and sparse is not None:
        # In case of discordance with sparsity one matrix is sparse
        # the other is not. Care is needed in matrix operation to use sparse
        # matrix performance:
        # (sparsetype.dot(densetype) is OK but densetype.dot(sparsetype)
        # have undefined behavior according to Scipy doc.
        # Performance gains of using 2 sparse matrices even when one is not
        # sparse are huge (~25x in my tests)
        # Surprisingly: denselyfilled_sparse.dot(sparselyfilled_sparse) is also
        # several times faster than
        # sparselyfilled_sparse.dot(denselyfilled_sparse) BUT
        # it takes much more memory and is likely to blow it up and
        # be killed by the OS
        if sparse:
            # the negated is dense, the transpose is sparse
            co_occur_mat = trans_oh_mat.dot(sp_oh_mat)
        else:
            # the negated is sparse, the transpose is dense
            co_occur_mat = (sp_oh_mat.transpose()
                            .dot(trans_oh_mat.transpose())
                            .transpose())
    else:
        # matrices are both sparse or both dense if not discordance
        co_occur_mat = trans_oh_mat.dot(sp_oh_mat)

    if freq:
        co_occur_mat = co_occur_mat / float(oh_df.shape[0])

    if isspmatrix(co_occur_mat):
        co_occur_mat = co_occur_mat.todense()
    return co_occur_mat


def compute_cooccur_mat(oh_df: pd.DataFrame, freq: bool = False,
                        extracols=None, negate=False, sparse=None,
                        precision: int = 64) -> np.ndarray:
    """Compute the co-occurrence matrix from a One hot DF."""
    return _compute_occurrence_mat(oh_df, freq=freq, extracols=extracols,
                                   negate=negate, discordance=False,
                                   sparse=sparse, precision=precision)


def compute_discordance_mat(oh_df: pd.DataFrame, freq: bool = False,
                            extracols=None, sparse=None,
                            precision: int = 64) -> np.ndarray:
    """Compute the discordance matrix from a One Hot DF."""
    return _compute_occurrence_mat(oh_df, freq=freq,
                                   extracols=extracols, discordance=True,
                                   sparse=sparse, precision=precision)


def threshold_occur_mat(co_occur_mat, n: int):
    """Keep cols/rows where the diagonal is larger than threshold."""
    # TODO: fix diagonal with ndarray compatible code
    keep_indices = np.asarray(
        np.where(np.asarray(co_occur_mat.diagonal())[0, :] >= n))[0, :]
    return np.asarray(co_occur_mat)[keep_indices, :][:, keep_indices]


def norm_cooccur_mat(co_occur_mat: np.ndarray,
                     correct_diag=True) -> np.ndarray:
    """
    Normalize co-occurrence frequencies.

    Normalize co-occurrence frequencies f(A,B) by frequencies expected from
    independent occurrences f(A)f(B). The provided matrix must be a
    frequency matrix.
    """
    assert np.logical_or(co_occur_mat >= 0, co_occur_mat <= 1).all()
    # compute expected joint frequencies
    co_occur_diag = co_occur_mat.diagonal().reshape((1, -1))
    null_freq = np.matmul(co_occur_diag.transpose(),
                          co_occur_diag)

    # correct for the fact that the diagonal is not a joint
    # TODO: fix true divide by 0
    if correct_diag:
        null_freq[range(0, null_freq.shape[0]), range(
            0, null_freq.shape[0])] /= co_occur_diag[0, :]
    # return normalized matrix making sure there is no divide by zero issue
    return np.true_divide(co_occur_mat, null_freq,
                          out=np.zeros((co_occur_mat.shape[0],) * 2),
                          where=co_occur_mat != 0)


def norm_discordance_mat(discordance_freq_mat: np.ndarray,
                         co_occur_freq_mat: np.ndarray) -> np.matrix:
    """Normalize the discordance matrix"""
    assert np.logical_or(co_occur_freq_mat >= 0, co_occur_freq_mat <= 1).all()
    assert np.logical_or(discordance_freq_mat >= 0,
                         discordance_freq_mat <= 1).all()

    # Compute the expected frequency matrix of (i and !j)
    co_occur_diag = co_occur_freq_mat.diagonal().reshape((1, -1))
    null_freq = np.matmul(co_occur_diag.transpose(),
                          (1.0 - co_occur_diag))

    # No need to correct for the fact that the diagonal is not a joint freq
    # because it will be 0 by def of discordance matrix

    # Normalize and return the discordance matrix
    return np.true_divide(discordance_freq_mat, null_freq,
                          out=np.zeros((discordance_freq_mat.shape[0],) * 2),
                          where=discordance_freq_mat != 0)


def mutual_information_mat(oh_df, extracols=None, logbase=2, sparse=None,
                           precision=64):
    """
    Compute the pairwise mutual information matrix between variables.

    The provided matrix must be a frequency matrix.
    """
    sample_size = oh_df.shape[0]
    # Compute co-occurrence, negated co occurrences depending on sparsity
    # Avoid computing one of the matrices as the dot product is
    # computationally costly and simply subtract the frequencies of
    # the others
    if sparse is None or sparse:
        co_occur_mat = compute_cooccur_mat(oh_df, freq=True, negate=False,
                                           extracols=extracols, sparse=sparse,
                                           precision=precision)
        gc.collect()
    elif sparse is not None and not sparse:
        neg_co_occur_mat = compute_cooccur_mat(oh_df, freq=True, negate=True,
                                               extracols=extracols,
                                               sparse=sparse,
                                               precision=precision)
        gc.collect()

    # Compute discordance no matter what (it will be sparse whether sparse
    # is True or False and dense otherwise but needed.
    discordance_mat = compute_discordance_mat(oh_df, freq=True,
                                              extracols=extracols,
                                              sparse=sparse,
                                              precision=precision)
    gc.collect()
    discordance_mat_T = discordance_mat.transpose()

    if sparse is None or sparse:
        neg_co_occur_mat = (np.ones(shape=co_occur_mat.shape,
                                    dtype=np.float64)
                            - co_occur_mat
                            - discordance_mat
                            - discordance_mat_T)
    elif sparse is not None and not sparse:
        co_occur_mat = (np.ones(shape=neg_co_occur_mat.shape,
                                dtype=np.float64)
                        - neg_co_occur_mat
                        - discordance_mat
                        - discordance_mat_T)

    # Normalize the occurrence matrices
    # Do not correct the diagonal normalization such that I(X,X)=S(X)
    normed_co_occur_mat = np.asarray(norm_cooccur_mat(co_occur_mat,
                                                      correct_diag=False))
    normed_neg_co_occur_mat = np.asarray(norm_cooccur_mat(neg_co_occur_mat,
                                                          correct_diag=False))
    normed_discordance_mat = np.asarray(norm_discordance_mat(discordance_mat,
                                                             co_occur_mat))
    normed_discordance_mat_T = normed_discordance_mat.transpose()

    # Compute the mutual information subparts
    mutual_inf_contrib = np.empty(shape=normed_discordance_mat.shape + (4,),
                                  dtype=np.float64)
    mat_list = (co_occur_mat, neg_co_occur_mat,
                discordance_mat, discordance_mat.transpose())
    normed_mat_list = (normed_co_occur_mat, normed_neg_co_occur_mat,
                       normed_discordance_mat,
                       normed_discordance_mat_T)
    for i, (mat, normed_mat) in enumerate(zip(mat_list, normed_mat_list)):
        mutual_inf_contrib[:, :, i] = (np.asarray(mat)
                                       * np.log(normed_mat,
                                                out=np.zeros(
                                                    (mat.shape[0],) * 2),
                                                where=mat != 0))

    # Sum parts, adapt base and return
    return mutual_inf_contrib.sum(axis=2) / np.log(logbase)


def _entropy(freq_array: np.ndarray, logbase=2) -> np.ndarray:
    """Compute entropies from the frequency array for boolean outcome."""
    rev = 1 - freq_array
    return - (freq_array * np.log(freq_array) / np.log(logbase)
              + rev * np.log(rev) / np.log(logbase))


def uncertainty_coef_mat(mutual_info_matrix: np.ndarray):
    """
    Compute the pairwise uncertainty coefficient.

    The uncertainty coefficient U(X|Y)=I(X,Y)/H(X).
    U[x,y] = U(X|Y)
    """
    mi_diag = mutual_info_matrix.diagonal().reshape((-1, 1))

    uncert_coef = np.true_divide(mutual_info_matrix, mi_diag,
                                 out=np.zeros((mi_diag.shape[0],) * 2),
                                 where=mutual_info_matrix != 0)

    # reset diagonal UC to NaN as they are not properly defined
    uncert_coef[range(0, uncert_coef.shape[0]), range(
        0, uncert_coef.shape[0])] = np.nan
    return uncert_coef
