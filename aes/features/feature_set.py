#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Automated Essay Scoring: A Data-First Deep Learning Approach                        #
# Version    : 0.1.0                                                                               #
# Python     : 3.9.12                                                                              #
# Filename   : /feature_set.py                                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 9th 2022 07:49:32 pm                                                 #
# Modified   : Friday August 12th 2022 08:43:33 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Feature Extraction Module."""
import pandas as pd
import logging
import logging.config

# ------------------------------------------------------------------------------------------------ #
from aes.utils.config import LogConfig
from aes.features.base import Feature
from aes.features import FEATURES

# ------------------------------------------------------------------------------------------------ #
logging.config.dictConfig(LogConfig().config)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


class FeatureSet:
    """Collection of Feature objects that can be extracted, summarized, and reported.

    Args:
        data (np.array): Numpy array containing the text.

    """

    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data  # The training data
        self._columns = data.columns.to.list()
        self._features = {}  # Dictionary of feature objects.
        self._extracted = False

    def extract(self) -> None:
        """Extracts and updates the data with length, word, syntactic, semantic and readability features."""

    def add_feature(self, feature: Feature) -> None:
        """Adds a feature to the FeatureSet

        Args:
            feature (Feature): Feature object
        """
        if self._features.get(feature.name, None):
            logger.warn("Feature {} already added to FeatureSet".format(feature.name))
        else:
            self._features[feature.name] = feature

    def remove_feature(self, name: str) -> None:
        """Removes a feature from the FeatureSet

        Args:
            name (str): The canonical name for the feature.
        """
        try:
            del self._features[name]
        except KeyError as e:
            logger.error("Feature {} does not belong to this FeatureSet object.".format(name))
            raise KeyError(e)

    def list_features(self) -> list:
        """Returns a list of features in the FeatureSet"""
        return [name for name in self._features.keys()]

    def get_feature(self, name: str) -> Feature:
        """Returns a feature object.

        Args:
            name (str): The name of the feature
        """
        try:
            return self._features[name]
        except KeyError as e:
            logging.error("Feature {} does not belong to this FeatureSet object.".format(name))

    def get_features(self, category: str = None) -> pd.DataFrame:
        """Returns the data with features added.

        Args:
            category (str): I grouping of features in  ['length', 'word', 'semantic', 'syntactic', 'readability']

        """
        if category and category in FEATURES.keys():
            features = FEATURES[category]
            columns = self._columns.axtend(features)
            return self._data[columns]
