#!/usr/bin/env python
# Created by "Thieu" at 13:58, 09/05/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


class LabelEncoder:
    """
    Encode categorical features as integer labels.
    """

    def __init__(self):
        self.unique_labels = None
        self.label_to_index = {}

    def fit(self, y):
        """
        Fit label encoder to a given set of labels.

        Parameters:
        -----------
        y : array-like
            Labels to encode.
        """
        self.unique_labels = np.unique(y)
        self.label_to_index = {label: i for i, label in enumerate(self.unique_labels)}

    def transform(self, y):
        """
        Transform labels to encoded integer labels.

        Parameters:
        -----------
        y : array-like
            Labels to encode.

        Returns:
        --------
        encoded_labels : array-like
            Encoded integer labels.
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        return np.array([self.label_to_index[label] for label in y])

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Encoded labels.
        """
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        """
        Transform integer labels to original labels.

        Parameters:
        -----------
        y : array-like
            Encoded integer labels.

        Returns:
        --------
        original_labels : array-like
            Original labels.
        """
        if self.unique_labels is None:
            raise ValueError("Label encoder has not been fit yet.")
        return np.array([self.unique_labels[i] if i in self.label_to_index.values() else "unknown" for i in y])
