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
# Created    : Wednesday August 10th 2022 05:03:22 pm                                              #
# Modified   : Friday August 12th 2022 08:43:33 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Base Feature Extraction Module"""
from abc import ABC, abstractmethod
import os
import importlib
import pandas as pd
import logging
import logging.config

# ------------------------------------------------------------------------------------------------ #
from aes.utils.config import LogConfig
from aes.utils.config import Config
from aes.utils.metacode import class_list_from_file

# ------------------------------------------------------------------------------------------------ #
logging.config.dictConfig(LogConfig().config)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


class FeatureExtractor(ABC):
    def __init__(self) -> None:
        self._config = Config(filepath="aes/data/config.yaml")
        config = self._config.read()
        self._idvar = config["columns"]["idvar"]  # The idvar in the data.
        self._text_col = config["columns"]["text"]  # The name of the text column in the data.
        self._name = None  # The canonical name for the feature assigned in subclasses.
        self._category = None  # The feature category assigned by subclasses.

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return self._category

    @abstractmethod
    def extract(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Extracts the features from the dataset and returns a Feature object.

        Args:
            data (pd.DataFrame): DataFrame containing the idvar and the text data (only) to be analyzed.

        Returns:
            pd.DataFrame containing the idvar and the feature values.
        """
        pass


# ------------------------------------------------------------------------------------------------ #
class FeatureExtractorFactory:
    """Constructs a feature extractor using the canonical name of the feature."""

    __modules = [
        "aes.features.extraction.length",
        "aes.features.extraction.word",
        "aes.features.extraction.semantic",
        "aes.features.extraction.syntactic",
        "aes.features.extraction.readability",
    ]

    def __init__(self) -> None:

        self._extractor_index = {}
        self._build_index()

    @property
    def categories(self) -> list:
        return [category for category in self._extractor_index.keys()]

    def list_extractors(self) -> list:
        return [e for e in self._extractor_index.keys()]

    def create_extractor(self, name: str) -> FeatureExtractor:
        try:
            return self._extractor_index[name]
        except KeyError as e:
            logger.error(
                "Invalid category or name. Check the category and name properties for valid options."
            )
            raise KeyError(e)

    def _build_index(self) -> None:

        for module_name in FeatureExtractorFactory.__modules:
            module = importlib.import_module(name=module_name)  # Get the module object.
            module_name_parts = module_name.split(".")
            module_path = os.path.join(*module_name_parts) + ".py"
            classlist = class_list_from_file(
                module_path
            )  # Obtain a list of class names in the module
            for klass in classlist:
                extractor = getattr(module, klass)
                extractor = extractor()
                self._extractor_index[extractor.name] = extractor
