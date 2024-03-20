"""
Code extracted and adpted from:
https://github.com/euranova/estimating_eces/tree/main

Commit SHA: 9bfa81dd7a39ebe069c5b11b8e7a9bf9017e9350

MIT License

Copyright (c) 2021 EURA NOVA

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Original author: nicolas.posocco
"""

# from ...prototype.binning import ConfidenceConvexAllocationBinningPolicy
# from ...prototype.bin_boundaries import EqualAmountBinBoundariesPolicy
# from ..discrete_metrics import confidence_binned_metric
import numpy as np
from math import floor  # , ceil
# from .....utils import confidences_from_scores

from abc import ABC


class BinBoundariesPolicy(ABC):

    def __init__(self):
        """
        Initializes a bin boundaries policy.
        """

        pass

    def __call__(self, n_bins, segment, elements):
        """
        Returns the bin boundaries given the policy and context (an n_bins+1 array).
        Args:
            n_bins: int, number of bins to create.
            segment: list or tuple (length 2) containing the limits of the
            segment to subdivide, in increasing order.
            elements: np.ndarray (shape (n_elements, )) containing the elements
            to send to the bins.

        Returns:
        An np.ndarray (shape (n_bins+1, ) containing the boundaries defining the n_bins bins.
        """

        # n_bins
        assert type(n_bins) is int
        assert n_bins > 0

        # segment
        assert type(segment) is list or type(segment) is tuple
        assert len(segment) == 2
        assert np.isscalar(segment[0]) and np.isscalar(segment[1])
        assert segment[0] < segment[1]

        # elements
        assert type(
            elements) is np.ndarray, f"elements should be an np.ndarray, instead of {type(elements)}"
        assert elements.dtype == np.number

        raise NotImplementedError

    def affect(self, bin_boundaries, element):
        """
        Describes how element is sent to its bin.
        Args:
            bin_boundaries: output of the __call__ method above.
            element: numeric object in the range of the segment

        Returns:
        The index of the bin in which the element is being sent.
        """

        # bin_boundaries
        assert type(bin_boundaries) is np.ndarray

        # element
        assert isinstance(element, (int, float, np.number)), \
            "element = {} should be of a numeric type, not {}.".format(
                element, type(element))
        assert bin_boundaries[0] <= element <= bin_boundaries[-1]

        # For all bins, in increasing order
        for m in range(1, len(bin_boundaries)):

            # If the element is too small to get into the mth bin
            if element < bin_boundaries[m]:
                # Returning the index of the previous one
                return m - 1

        # Boundary case : element belongs to the last bin.
        return len(bin_boundaries) - 2

    def affect_all(self, bin_boundaries, elements):
        """
        Assigns all elements to their corresponding bins all at once.

        Uses vectorized operations, should be much faster as long as the number
        of bins is less than number of elements.
        Args:
            bin_boundaries: output of the __call__ method above.
            elements: numeric array object in the range of the segment

        Returns:
        A list of numpy array of indices of elements belonging to each bin.
        """
        # make sure elements is 1 dimensional
        elements = np.asarray(elements)
        assert len(elements.shape) == 1

        # Initialize with the array of elements for the first bin
        bins_elements = [(elements < bin_boundaries[0]).nonzero()[0]]
        # For all bins, in increasing order
        for b in range(1, len(bin_boundaries)):
            bins_elements.append(
                (
                    (elements >= bin_boundaries[b-1])
                    & (elements < bin_boundaries[b])
                ).nonzero()[0]
            )
        # Add elements of the final bin
        bins_elements.append((elements >= bin_boundaries[-1]).nonzero()[0])
        return bins_elements


class EqualBinsBinBoundariesPolicy(BinBoundariesPolicy):

    def __init__(self):
        """
        Initializes an equal bins bin boundaries policy,
        which splits given segment in n_bins equal bins
        """

        super().__init__()

    def __call__(self, n_bins, segment, elements):
        """
        Returns the bin boundaries corresponding to equal bins division.
        Args:
            n_bins: int, number of bins to create.
            segment: list or tuple (length 2) containing the limits of the
            segment to subdivide, in increasing order.
            elements: np.ndarray (shape (n_elements, )) containing the
            elements to send to the bins.

        Returns:
        An np.ndarray (shape (n_bins+1, ) containing the boundaries defining
        the n_bins bins.
        """

        # n_bins
        assert type(n_bins) is int
        assert n_bins > 0

        # segment
        assert type(segment) is list or type(segment) is tuple
        assert len(segment) == 2
        assert np.isscalar(segment[0]) and np.isscalar(segment[1])
        assert segment[0] < segment[1]

        # elements
        assert type(
            elements) is np.ndarray, f"elements should be an np.ndarray, instead of {type(elements)}"
        assert elements.dtype == np.number

        return np.array([segment[0] + i / n_bins * (segment[1] - segment[0])
                         for i in range(n_bins)]
                        + [float(segment[1])])

    def affect(self, bin_boundaries, element):
        """
        Describes how element is sent to its bin.
        Args:
            bin_boundaries: output of the __call__ method above.
            element: numeric object in the range of the segment

        Returns:
        The index of the bin in which the element is being sent.
        """

        # bin_boundaries
        assert type(bin_boundaries) is np.ndarray

        # element
        assert isinstance(element, (int, float, np.number)), \
            "element = {} should be of a numeric type, not {}.".format(
                element, type(element))
        assert bin_boundaries[0] <= element <= bin_boundaries[-1]

        n_bins = len(bin_boundaries) - 1
        m = floor(element * n_bins) if floor(element *
                                             n_bins) < n_bins else n_bins - 1

        return m


class EqualAmountBinBoundariesPolicy(BinBoundariesPolicy):

    def __init__(self):
        """
        Initializes a bin boundaries policy.
        """

        super().__init__()

    def __call__(self, n_bins, segment, elements):
        """
        Returns the bin boundaries corresponding to an adaptative binning
        policy
        (same amount of samples in each bin).
        Args:
            n_bins: int, number of bins to create.
            segment: list or tuple (length 2) containing the limits of the
            segment to subdivide, in increasing order.
            elements: np.ndarray (shape (n_elements, )) containing the elements
            to send to the bins.

        Returns:
        An np.ndarray (shape (n_bins+1, ) containing the boundaries defining
        the n_bins bins.
        """

        # n_bins
        assert type(n_bins) is int
        assert n_bins > 0

        # segment
        assert type(segment) is list or type(segment) is tuple
        assert len(segment) == 2
        assert np.isscalar(segment[0]) and np.isscalar(segment[1])
        assert segment[0] < segment[1]

        # elements
        assert type(
            elements) is np.ndarray, f"elements should be an np.ndarray, instead of {type(elements)}"
        assert elements.dtype == np.number

        sorted_elements = np.sort(elements)

        bin_card = int(floor(elements.shape[0]/n_bins))

        bin_boundaries = [segment[0]]

        for i in range(1, n_bins):
            boundary_l = sorted_elements[i*bin_card - 1]
            boundary_r = sorted_elements[i * bin_card]
            boundary = (boundary_l+boundary_r)/2

            bin_boundaries.append(boundary)

        bin_boundaries.append(segment[1])

        return np.array(bin_boundaries)


class BinningPolicy(ABC):

    def __init__(self):
        """
        Initializes the binning policy.
        """

        pass

    def __call__(self, model=None, X=None, scores=None):
        """
        Sends samples into bins depending on model scores.
        Args:
            model: ML model.
            X: np.ndarray (n_samples, dimensionality), data matrix.

        Returns:
        A list of list of 2-tuples (int, float) defining the binning
        affectation of samples, each tuple being a
        (sample_index, weight).
        """

        raise NotImplementedError


class FixedBinAmountBinningPolicy(BinningPolicy):

    def __init__(self, n_bins, bin_boundaries_policy):
        """
        Initializes a fixed bin amount binning policy,
        which sends samples into the n_bins bins created by the
        bin_boundaries_policy.
        Args:
            n_bins: int, number of bins used to divide the interval.
            bin_boundaries_policy: BinBoundariesPolicy, defining how to split
            the interval into bins.
        """

        # n_bins
        assert type(n_bins) is int or n_bins in ("sqrt",)
        assert n_bins > 0

        # bin_boundaries_policy
        assert isinstance(bin_boundaries_policy, BinBoundariesPolicy)

        self.n_bins = n_bins
        self.bin_boundaries_policy = bin_boundaries_policy

        super().__init__()

    def __call__(self, model=None, X=None, scores=None):
        """
        Sends samples into bins depending on model scores.
        Args:
            model: ML model.
            X: np.ndarray (n_samples, dimensionality), data matrix.

        Returns:
        A list of list of 2-tuples (int, float) defining the binning
        affectation of samples, each tuple being a
        (sample_index, weight).
        """

        # model
        assert hasattr(model, "predict_proba")

        # X
        assert type(X) is np.ndarray
        assert X.ndim == 2

        raise NotImplementedError


class ConfidenceBinningPolicy(FixedBinAmountBinningPolicy):

    def __init__(self, n_bins, bin_boundaries_policy):
        """
        Initializes a predicted class binning policy,
        which sends samples into the n_bins bins created by the
        bin_boundaries_policy,
        based on the score the model gives for the class provided at runtime.
        Args:
            n_bins: int, number of bins used to divide the interval.
            bin_boundaries_policy: BinBoundariesPolicy, defining how to split
            the interval into bins.
        """

        super().__init__(n_bins=n_bins,
                         bin_boundaries_policy=bin_boundaries_policy)

    def __call__(self, confidence_scores, n_classes):
        """

        Args:
            model:
            X:

        Returns:

        """

        # if confidence_scores is None:
        #     # model
        #     assert hasattr(model, "predict_proba"), "Model <model> must
        # implement the predict_proba method."

        #     # X
        #     assert type(X) is np.ndarray, "<X> must be a numpy array."
        #     assert X.ndim == 2, "Number of dimensions of array <X> must be 2."

        #     # Calculating model predictions and scores on X
        #     scores = model.predict_proba(X)
        #     predictions = model.predict(X)
        #     confidence_scores = confidences_from_scores(scores,
        #                                               predictions, model)

        # Grouping predictions into <bins> interval bins, each of size 1/M
        # bins = [[] for _ in range(self.n_bins)]
        low = 1 / int(n_classes)
        bin_boundaries = self.bin_boundaries_policy(
            n_bins=self.n_bins,
            segment=[low, 1],
            elements=confidence_scores)
        # for i in range(confidence_scores.shape[0]):
        #     # Getting corresponding bin
        #     m = self.bin_boundaries_policy.affect(bin_boundaries,
        #                                           confidence_scores[i])

        #     # Adding sample id to corresponding bin (no specific weight -> 1)
        #     # Perf fix: used to return tuples of (id,weight=1) with weight
        #     # never used
        #     bins[m].append(i)

        bins = self.bin_boundaries_policy.affect_all(bin_boundaries,
                                                     confidence_scores)

        return bins


class ConfidenceConvexAllocationBinningPolicy(FixedBinAmountBinningPolicy):

    def __init__(self, n_bins, bin_boundaries_policy):
        """
        TODO
        Args:
            n_bins: int, number of bins used to subdivide the segment [0,1].
            bin_boundaries_policy: BinBoundariesPolicy object, defining how
            the segment [0,1] is divided.
        """

        super().__init__(n_bins=n_bins,
                         bin_boundaries_policy=bin_boundaries_policy)

    def __call__(self, confidence_scores, n_classes):
        """
        TODO
        Args:
            model: ML model.
            X: np.ndarray (shape (n_samples, dimensionality)), data matrix.

        Returns:
        TODO
        """

        # if confidence_scores is None:

        #     # model
        #     assert hasattr(model, "predict_proba"), "Model must implement
        # the predict_proba method."

        #     # X
        #     assert type(X) is np.ndarray, "X must be a numpy array."
        #     assert X.ndim == 2, "Number of dimensions of array X must be 2."

        #     # Calculating model predictions and scores on X
        #     scores = model.predict_proba(X)
        #     predictions = model.predict(X)
        #     confidences = confidences_from_scores(scores, predictions, model)

        # Grouping predictions into <bins> interval bins, each of size 1/M
        bins = [[] for _ in range(self.n_bins)]
        low = 1 / int(n_classes)
        bin_boundaries = self.bin_boundaries_policy(
            n_bins=self.n_bins,
            segment=[low, 1],
            elements=confidence_scores)

        centroids = [(bin_boundaries[i] + bin_boundaries[i - 1]
                      ) / 2 for i in range(1, self.n_bins + 1)]

        for i in range(confidence_scores.shape[0]):

            # Getting corresponding predicted score
            target_score = confidence_scores[i]

            # Getting corresponding bin
            if target_score < centroids[0]:
                bins[0].append((i, 1))
                continue
            elif target_score > centroids[-1]:
                bins[-1].append((i, 1))
                continue

            # Usual case : item falls between two centroïds
            for b in range(self.n_bins):
                if centroids[b] < target_score:
                    continue
                else:
                    di = centroids[b] - target_score
                    dim1 = target_score - centroids[b - 1]
                    bins[b].append((i, di / (di + dim1)))
                    bins[b - 1].append((i, dim1 / (di + dim1)))
                    break

        return bins


def confidence_ece_ac(probas_pred=None, predictions=None,
                      confidence_scores=None, Y=None, n_bins=10,
                      backend=None):
    """
    Calculates the ECE (adaptative binning, convex allocation) of the model
    based on data (X,Y).
    Args:
        model: the model whose ECE we want.
        probas_pred: numpy.ndarray, variables of the calibration set.
        Y: numpy.ndarray, labels of the calibration set.
        n_bins: int, number of bins used to discretize score space [0,1].
        backend: string (default "accuracies_confidences"), name of the
        backend used.

    Returns:
    The expected calibration error (ECE) of the model.
    """

    if backend is None:
        backend = "accuracies_confidences"

    if backend == "accuracies_confidences":
        # Implementation in terms of accuracies and confidences

        if confidence_scores is None:
            # model
            if predictions is None:
                predictions = np.argmax(probas_pred, axis=1)
                confidence_scores = np.max(probas_pred, axis=1)
            # confidences_from_scores(model=model, predictions=predictions,
            # scores_matrix=probas_pred)

        if n_bins == "sqrt":
            n_bins = int(np.sqrt(len(confidence_scores)))

        bin_boundaries_policy = EqualAmountBinBoundariesPolicy()
        binning_policy = ConfidenceConvexAllocationBinningPolicy(
            bin_boundaries_policy=bin_boundaries_policy,
            n_bins=n_bins)

        bins_weights = binning_policy(confidence_scores=confidence_scores,
                                      n_classes=probas_pred.shape[1])

        card_dataset = confidence_scores.shape[0]

        result = 0
        for bin_weights in bins_weights:
            card_bin = np.sum(
                [sample_weight[1] for sample_weight in bin_weights]
            )

            if card_bin > 0:
                bin_acc_unorm = 0
                bin_conf_unorm = 0
                for i, w in bin_weights:
                    bin_acc_unorm += w * int(Y[i] == predictions[i])
                    bin_conf_unorm += w * confidence_scores[i]
                bin_contribution = (
                    np.abs(bin_acc_unorm - bin_conf_unorm) / card_bin
                )
                result += bin_contribution * card_bin / card_dataset

        return result

    elif backend == "contributions":
        raise NotImplementedError
        # Calculating bins allocation
        # bin_boundaries_policy = EqualAmountBinBoundariesPolicy()
        # binning_policy = ConfidenceConvexAllocationBinningPolicy(
        #       bin_boundaries_policy=bin_boundaries_policy,
        #     n_bins=n_bins)
        # bins_weights = binning_policy(model=model,  X=X)

        # return confidence_binned_metric(model, X, Y, bins_weights)

    elif backend == "prototype":
        raise NotImplementedError

    else:
        raise NotImplementedError


def confidence_ece_a(probas_pred=None, predictions=None,
                     confidence_scores=None, Y=None, n_bins=10,
                     backend=None, summarizing_function="average"):
    """
    Calculates the ECE (adaptative binning) of the model based on data (X,Y).
    Args:
        model: the model whose ECE we want.
        X: numpy.ndarray, variables of the calibration set.
        Y: numpy.ndarray, labels of the calibration set.
        n_bins: int, number of bins used to discretize score space [0,1].
        backend: string (default "prototype"), name of the backend used.
        summarizing_function: "average" or "max" or an array like containing 
        different summarizing function strings
        
    Returns:
    The calibration error (CE) of the model summarized by the provided function
    (default to average or expectation).

    Notes:
    If several summarizing functions are provided, results are provided as a
    with the same order.
    """
    probas_pred = np.asarray(probas_pred)
    # Assert the given array is an array of probabilities with tolerance
    assert (np.abs(probas_pred.sum(axis=1)-1.0) < 1e-6).all() 

    if backend is None:
        backend = "accuracies_confidences"

    if backend == "contributions":
        raise NotImplementedError
        # # Calculating bins allocation
        # bin_boundaries_policy = EqualAmountBinBoundariesPolicy()
        # binning_policy = ConfidenceBinningPolicy(
        #       bin_boundaries_policy=bin_boundaries_policy,
        #       n_bins=n_bins)
        # bins_weights = binning_policy(model=model,  X=X)

        # return confidence_binned_metric(model, X, Y, bins_weights)

    elif backend == "accuracies_confidences":
        # Implementation in terms of accuracies and confidences

        if confidence_scores is None:
            # model
            if predictions is None:
                predictions = np.argmax(probas_pred, axis=1)
                confidence_scores = np.max(probas_pred, axis=1)
            # confidences_from_scores(model=model, predictions=predictions,
            # scores_matrix=probas_pred)

        if n_bins == "sqrt":
            n_bins = int(np.sqrt(len(confidence_scores)))

        bin_boundaries_policy = EqualAmountBinBoundariesPolicy()
        binning_policy = ConfidenceBinningPolicy(
            bin_boundaries_policy=bin_boundaries_policy,
            n_bins=n_bins
        )

        bins_elements_ind = binning_policy(
            confidence_scores=confidence_scores,
            n_classes=probas_pred.shape[1]
        )

        card_dataset = confidence_scores.shape[0]

        bins_weights = []
        bins_ce = []
        for bin_elements_ind in bins_elements_ind:
            sample_indices = np.asarray(bin_elements_ind)
            card_bin = sample_indices.shape[0]
            bin_acc_unorm = (
                Y[sample_indices] == predictions[sample_indices]
            ).astype(dtype=int).sum()
            bin_conf_unorm = confidence_scores[sample_indices].sum()
            if card_bin > 0:
                bin_ce = np.abs(
                    bin_acc_unorm - bin_conf_unorm) / float(card_bin)
            else:
                # Cannot compute bin's CE
                bin_ce = np.NaN
            bins_weights.append(float(card_bin)/float(card_dataset))
            bins_ce.append(bin_ce)
        # assert np.abs(np.asarray(bins_weights).sum()-1) < 1e-5
        bins_ce = np.ma.array(bins_ce, mask=np.isnan(bins_ce))
        returns = []
        if isinstance(summarizing_function, str):
            summarizing_function = [summarizing_function]
        for func_str in tuple(summarizing_function):
            if func_str == "average":
                # Ignore NaNs using a masked array for averaging
                ece = np.ma.average(a=bins_ce, weights=bins_weights)
                returns.append(ece)
            elif func_str == "max":
                returns.append(np.ma.max(bins_ce))
            else:
                raise NotImplementedError("Unknown summarizing function '"
                                          + func_str + "'")
        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)
