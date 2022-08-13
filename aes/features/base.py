#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Automated Essay Scoring: A Data-First Deep Learning Approach                        #
# Version    : 0.1.0                                                                               #
# Python     : 3.9.12                                                                              #
# Filename   : /base.py                                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday August 10th 2022 04:30:42 am                                              #
# Modified   : Friday August 12th 2022 08:43:33 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Defines Base classes for Feature classes throughout the package."""
import pandas as pd
import matplotlib.pyplot as plt
import logging
import logging.config
from typing import Union

# ------------------------------------------------------------------------------------------------ #
from aes.utils.config import LogConfig
from aes.visualization.visualize import Histogram, Boxplot
from aes.features.extraction.base import FeatureExtractorFactory, FeatureExtractor

# ------------------------------------------------------------------------------------------------ #
logging.config.dictConfig(LogConfig().config)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
# ------------------------------------------------------------------------------------------------ #


class Feature:
    """Base class for all Feature objects. Defines the basic interface and behaviors.

    Args:
        name (str): The name of the feature e.g. 'word_count'
        category (str): The category of feature, e.g. length feature.


    """

    def __init__(self, name: str, category: str) -> None:
        self._name = name
        self._category = category
        self._values = None  # DataFrame containing the id_var and feature values.

    @property
    def values(self) -> pd.DataFrame:
        return self._values

    @property
    def name(self) -> None:
        return self._name

    @property
    def category(self) -> None:
        self._category

    def extract(self, data: pd.DataFrame, **kwargs) -> None:
        self._values = data[self._idvars].to_frame()
        extractor = self._extractor_factory()
        self._values[self._name] = extractor.extract(data, kwargs)

    def describe(self, by: str = None) -> pd.DataFrame:
        """Returns a DataFrame with descriptive statistics for the feature at the 'by' level of aggregation"""
        return self._values.describe[self._name].describe().to_frame().T

    def hist(
        self, by: str = None, title: str = None, xlab: str = None, ylab: str = None
    ) -> Union[plt.figure, plt.axes]:
        """Histogram displaying the distribution of the feature counts."""
        title = (
            "Distribution of {}\nFeedback Prize - Predicting Effective Arguments Dataset".format(
                self._name
            )
        )
        xlab = self._name
        ylab = "Counts"
        visualizer = Histogram()
        fig, ax = visualizer.plot(
            data=self._values, x=self._name, title=title, xlab=xlab, ylab=ylab
        )
        plt.tight_layout()
        plt.show()

    def boxplot(
        self, by: str = None, title: str = None, xlab: str = None, ylab: str = None
    ) -> Union[plt.figure, plt.axes]:
        """Boxplot presenting the distribution of the feature counts."""
        title = (
            "Distribution of {}\nFeedback Prize - Predicting Effective Arguments Dataset".format(
                self._name
            )
        )
        xlab = self._name
        visualizer = Boxplot()
        fig, ax = visualizer.plot(data=self._values, x=self._name, title=title, xlab=xlab)
        plt.tight_layout()
        plt.show()

    def _extractor_factory(self) -> FeatureExtractor:
        factory = FeatureExtractorFactory()
        return factory.create_extractor(category=self._category, name=self._name)
