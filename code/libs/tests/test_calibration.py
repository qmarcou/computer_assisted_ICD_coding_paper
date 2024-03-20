from unittest import TestCase
import calibration
import numpy as np


class TestConfidenceECE(TestCase):
    Y = np.asarray(([1]*50)+([0]*50))
    
    def test_confidence_ece_a(self):
        np.testing.assert_almost_equal(
            actual=calibration.confidence_ece_a(
                probas_pred=np.asarray([[0.5, 0.5], ]*100),
                Y=self.Y,
                n_bins=1,
                backend="accuracies_confidences",
                summarizing_function="average"),
            desired=0.0
        )
        
        np.testing.assert_almost_equal(
            actual=calibration.confidence_ece_a(
                probas_pred=np.asarray([[1.0, 0.0], ]*100),
                Y=self.Y,
                n_bins=1,
                backend="accuracies_confidences",
                summarizing_function="average"),
            desired=0.5
        )
        
        np.testing.assert_almost_equal(
            actual=calibration.confidence_ece_a(
                probas_pred=np.asarray([[.90, 0.10], ]*100),
                Y=self.Y,
                n_bins=1,
                backend="accuracies_confidences",
                summarizing_function="average"),
            desired=0.4
        )
        
        # Test max calibration error
        np.testing.assert_almost_equal(
            actual=calibration.confidence_ece_a(
                probas_pred=np.asarray([[.10, 0.90], ]*50
                                       + [[.30, 0.70], ]*50),
                Y=self.Y,
                n_bins=2,
                backend="accuracies_confidences",
                summarizing_function="max"),
            desired=0.7
        )
              
        # Test dual return of average and max
        np.testing.assert_almost_equal(
            actual=np.asarray(calibration.confidence_ece_a(
                probas_pred=np.asarray([[.10, 0.90], ]*50
                                       + [[.30, 0.70], ]*50),
                Y=self.Y,
                n_bins=2,
                backend="accuracies_confidences",
                summarizing_function=["average", "max"])),
            desired=np.asarray((0.4, 0.7))
        )

        # Check that it also works the other way around
        np.testing.assert_almost_equal(
            actual=np.asarray(calibration.confidence_ece_a(
                probas_pred=np.asarray([[.10, 0.90], ]*50
                                       + [[.30, 0.70], ]*50),
                Y=self.Y,
                n_bins=2,
                backend="accuracies_confidences",
                summarizing_function=["max", "average"])),
            desired=np.asarray((0.7, 0.4))
        )
