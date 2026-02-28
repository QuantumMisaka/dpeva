"""Clustering methods."""

from __future__ import annotations

import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import Birch

logger = logging.getLogger(__name__)


class BirchClustering(BaseEstimator, TransformerMixin):
    """Birch Clustering as one step of the DIRECT pipeline."""

    def __init__(self, n=None, threshold_init=0.5, max_iter=50, min_threshold=1e-3, **kwargs):
        """
        Args:
            n: Clustering the PCs into n clusters. When n is None, the number of clusters
                is dependent on threshold_init and other kwargs, and the final
                (global) clustering step is skipped. Default to None.
            threshold_init: The initial radius of the subcluster obtained by merging
                a new sample and the closest subcluster should be lesser than
                the threshold. Otherwise, a new subcluster is started. See details in:
                https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html.
                Users may tune this value for desired performance of birch, while 0.5
                is generally a good starting point, and some automatic tuning is done
                with our built-in codes to achieve n clusters if given.
            max_iter: Maximum number of iterations for threshold adjustment. Default to 50.
            min_threshold: Minimum threshold value to prevent underflow/infinite loops. Default to 1e-3.
            **kwargs: Pass to BIRCH.
        """
        self.n = n
        self.threshold_init = threshold_init
        self.max_iter = max_iter
        self.min_threshold = min_threshold
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """
        Place holder for fit API.

        Args:
            X: Any inputs
            y: Any outputs

        Returns: self
        """
        return self

    def transform(self, PCAfeatures):
        """
        Perform Birch Clustering to an array of input PCA features.

        Args:
            PCAfeatures: An array of PCA features.

        Returns:
            A dict of Birch Clustering results, including labels of each
            PCA feature, centroid positions of each cluster in PCA feature s
            pace, and the array of input PCA features.
        """
        model = Birch(n_clusters=self.n, threshold=self.threshold_init, **self.kwargs).fit(PCAfeatures)
        
        if self.n is not None:
            iteration = 0
            while (
                len(set(model.subcluster_labels_)) < self.n
                and iteration < self.max_iter
            ):  # decrease threshold until desired n clusters is achieved
                current_clusters = len(set(model.subcluster_labels_))
                logger.info(
                    f"Iteration {iteration+1}/{self.max_iter}: "
                    f"BirchClustering with threshold={self.threshold_init:.6f} and n={self.n} "
                    f"gives {current_clusters} subclusters.",
                )
                
                # Check minimum threshold safety
                if self.threshold_init < self.min_threshold:
                    logger.warning(
                        f"Threshold {self.threshold_init:.6e} dropped below min_threshold {self.min_threshold}. "
                        "Stopping iteration to prevent underflow."
                    )
                    break

                # Update threshold
                # Safety: Ensure we don't multiply by 0 if current_clusters is 0 (though unlikely)
                ratio = current_clusters / self.n if self.n > 0 else 0
                if ratio == 0:
                     ratio = 0.1 # Fallback to avoid zeroing out threshold instantly
                
                self.threshold_init = self.threshold_init * ratio
                
                model = Birch(n_clusters=self.n, threshold=self.threshold_init, **self.kwargs).fit(PCAfeatures)
                iteration += 1

            # Final check and warning
            final_subclusters = len(set(model.subcluster_labels_))
            if final_subclusters < self.n:
                logger.warning(
                    f"BirchClustering failed to reach target n={self.n} clusters after {iteration} iterations. "
                    f"Final subclusters: {final_subclusters}. Proceeding with best effort."
                )

        labels = model.predict(PCAfeatures)
        self.model = model
        logger.info(
            f"BirchClustering with threshold_init={self.threshold_init} and n={self.n} "
            f"gives {len(set(model.subcluster_labels_))} clusters.",
        )
        label_centers = dict(zip(model.subcluster_labels_, model.subcluster_centers_))
        return {
            "labels": labels,
            "label_centers": label_centers,
            "PCAfeatures": PCAfeatures,
        }
