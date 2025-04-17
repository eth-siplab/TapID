"""
Data preprocessing utilities for the TapID project.

This module provides transformers and utilities for preprocessing sensor data,
including normalization, scaling, and other data transformations.
"""

import numpy as np
from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin


class DataNormaliser(BaseEstimator, TransformerMixin):
    """
    Normalizes time-series sensor data using standardization and optional jerk (derivative) computation.
    
    This transformer works on data shaped (batch_size, channels, time).
    
    Parameters
    ----------
    jerk : bool, default=True
        Whether to compute the temporal derivative (jerk) before normalization.
    
    Attributes
    ----------
    _mean : np.ndarray
        Mean computed during fit, shape (1, channels, 1).
    _std : np.ndarray
        Standard deviation computed during fit, shape (1, channels, 1).
    """

    def __init__(self, jerk: bool = True):
        self._jerk = jerk
        self._mean = None
        self._std = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'NormaliseScaler':
        """
        Fit the scaler to the input data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (batch_size, channels, time).
        y : Optional[np.ndarray]
            Ignored.

        Returns
        -------
        NormaliseScaler
            Fitted instance.
        """
        X_ = self._apply_jerk(X)
        self._mean = np.mean(X_, axis=(0, 2), keepdims=True)  # mean over batch & time
        self._std = np.std(X_, axis=(0, 2), keepdims=True)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply normalization to the input data.

        Parameters
        ----------
        X : np.ndarray
            Data to normalize, shape (batch_size, channels, time).

        Returns
        -------
        np.ndarray
            Normalized data with same shape.
        """
        X_ = self._apply_jerk(X)
        X_ = (X_ - self._mean) / self._std
        return X_.astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the scaler and transform the input data.

        Parameters
        ----------
        X : np.ndarray
            Data to fit and transform, shape (batch_size, channels, time).
        y : Optional[np.ndarray]
            Ignored.

        Returns
        -------
        np.ndarray
            Normalized data.
        """
        return self.fit(X, y).transform(X)

    def _apply_jerk(self, X: np.ndarray) -> np.ndarray:
        """
        Apply jerk (temporal diff) across the time axis if enabled.

        Parameters
        ----------
        X : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Preprocessed data.
        """
        if self._jerk:
            return np.diff(X, axis=-1, append=0)  # diff along time (last axis)
        return X

    @property
    def is_fit(self) -> bool:
        """
        Check if the scaler has been fitted.

        Returns
        -------
        bool
        """
        return self._mean is not None and self._std is not None

    def __getstate__(self):
        """
        For pickling.
        """
        return self.__dict__.copy()

    def __setstate__(self, state):
        """
        For unpickling.
        """
        self.__dict__ = state

