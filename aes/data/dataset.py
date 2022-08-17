#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Automated Essay Scoring: A Data-First Deep Learning Approach                        #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /dataset.py                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday August 12th 2022 02:28:43 pm                                                 #
# Modified   : Monday August 15th 2022 04:59:41 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Dataset Module"""
import os
import pandas as pd
import logging
import logging.config
from aes.utils.io import IO
from aes.utils.config import LogConfig

# ------------------------------------------------------------------------------------------------ #
logging.config.dictConfig(LogConfig().config)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# ------------------------------------------------------------------------------------------------ #


class Dataset:
    """Encapsulates dataset with io capability

    Args:
        name (str): Name for this instantiation of the dataset.
        stage (str): Stage of data processing, e.g.  'raw'. Optional, as None means pending acquisition.
        filepath (str): Path to file
        version (int): Numeric version number
    """

    __feature_names = [
        "discourse_id",
        "essay_id",
        "discourse_text",
        "discourse_type",
    ]
    __primary_key = "discourse_id"
    __target_var = "discourse_effectiveness"
    __text_var = "discourse_text"

    def __init__(
        self,
        name: str,
        stage: str = None,
        filepath: str = None,
        version: int = 1,
    ) -> None:
        self._name = name
        self._stage = stage
        self._filepath = filepath
        self._fileformat = filepath.splitext()[1].replace(".", "")
        self._version = version

        self._io = IO(self._fileformat)

        self._feature_names = Dataset.__feature_names
        self._primary_key = Dataset.__primary_key
        self._target_var = Dataset.__target_var
        self._text_var = Dataset.__text_var

        self._data = None

        if os.path.exists(self._filepath):
            self._load()

    @property
    def name(self) -> str:
        return self._name

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def filepath(self) -> str:
        return self._filepath

    @property
    def fileformat(self) -> str:
        return self._fileformat

    @property
    def version(self) -> str:
        return self._version

    @property
    def columns(self) -> list:
        return self._columns

    @property
    def primary_key(self) -> list:
        return self._primary_key

    @property
    def target_var(self) -> list:
        return self._target_var

    @property
    def text_var(self) -> list:
        return self._text_var

    @property
    def features(self) -> pd.DataFrame:
        return self._data[self._feature_names]

    @property
    def target(self) -> pd.DataFrame:
        return self._data[self._primary_key, self._target_var]

    @property
    def texts(self) -> pd.DataFrame:
        return self._data[[self._primary_key, self._text_var]]

    def _load(self) -> None:
        self._data = self._io.read(self._filepath)
