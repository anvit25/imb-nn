from sklearn.base import ClassifierMixin
from sklearn.neighbors._base import NeighborsBase, KNeighborsMixin
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.estimator_checks import check_estimator
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.preprocessing import LabelEncoder

from scipy.stats import nbinom


class Proposed(KNeighborsMixin, ClassifierMixin, NeighborsBase):
    def __init__(
        self, n_neighbors=5, algorithm="auto",
        leaf_size=30, p=2, metric="minkowski",
        metric_params=None, n_jobs=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm, leaf_size=leaf_size,
            metric=metric, p=p, metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def fit(self, X, y):
        # self._fit(X, y)
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.le_ = LabelEncoder()
        y = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        _, counts = np.unique(y, return_counts=True)
        self.n_features_in_ = X.shape[1]
        self.counts_ = counts

        if X.shape[0] < 1:
            raise ValueError("Atleast 1 sample reqruied.")

        if len(counts) == 1:
            raise ValueError(f"Unknown label type: Proposed estimator only \
                supports 2 classes, got 1 class.")

        if len(counts) != 2:
            raise ValueError(f"Unknown label type: Proposed estimator only \
                supports 2 classes, got {len(counts)} classes.")

        # self.X_ = X
        # self.y_ = y

        self.minority_class_ = counts.argmin()
        self.minority_ = KNeighborsClassifier(**self.get_params())
        self.minority_.fit(X[y == self.minority_class_],
                           y[y == self.minority_class_])
        self.majority_ = KNeighborsClassifier(**self.get_params())
        self.majority_.fit(X[y != self.minority_class_],
                           y[y != self.minority_class_])
        self.p_ = (y == self.minority_class_).mean()

        # Return the classifier
        return self

    def predict_proba(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        minor_distances, _ = self.minority_.kneighbors(X)

        n_pts = self.majority_.n_samples_fit_
        major_distances, _ = self.majority_.kneighbors(X, n_neighbors=n_pts)
        fails = self.compute_fails(major_distances, minor_distances)

        evidence_1 = nbinom.sf(fails, 1 + np.arange(self.n_neighbors), self.p_) + \
            0.5*nbinom.pmf(fails, 1 + np.arange(self.n_neighbors), self.p_)
        cols = np.abs(evidence_1 - 1/2).argmax(1)
        rows = np.arange(evidence_1.shape[0])
        prob_1 = evidence_1[rows, cols]
        if not self.minority_class_:
            prob_1 = 1 - prob_1

        return np.stack((1-prob_1, prob_1), axis=1)

    def compute_fails(self, major_distances, minor_distances):
        q1, t = major_distances.shape
        q2, k = minor_distances.shape
        assert q1 == q2
        a = major_distances.reshape((q1, t, 1))
        b = minor_distances.reshape((q1, 1, k))
        fails = (a < b).astype(int).sum(1)
        return fails

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.le_.inverse_transform(probs.argmax(1))

    def _more_tags(self):
        return {'binary_only': True, 'multioutput': False}


if __name__ == "__main__":
    print("Checking Estimator")
    check_estimator(Proposed())
    # check_estimator(Proposed2())
    print("Estimator Check Passed")
