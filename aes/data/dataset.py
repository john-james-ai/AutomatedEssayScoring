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
# Modified   : Friday August 12th 2022 08:43:33 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Dataset Module"""
import os
import pandas as pd
import logging
import logging.config
from aes.workflow.operators import LoadCSV, SaveCSV
from aes.utils.config import LogConfig

# ------------------------------------------------------------------------------------------------ #
logging.config.dictConfig(LogConfig().config)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# ------------------------------------------------------------------------------------------------ #


class Dataset:
    """Encapsulates dataset metadata with io capability

    Args:
        id (str): Dataset identifier, e.g. 'fp2022'
        name (str): Dataset name
        description (str): Dataset description
        stage (str): Stage of data processing, e.g. 'raw'
        filepath (str): Path to file
        version (int): Numeric version number
    """

    __columns = [
        "discourse_id",
        "essay_id",
        "discourse_text",
        "discourse_type",
        "discourse_effectiveness",
    ]
    __primary_key = "discourse_id"
    __target_var = "discourse_effectiveness"
    __text_var = "discourse_text"

    def __init__(
        self, id: str, name: str, description: str, stage: str, filepath: str, version: int = 1
    ) -> None:
        self._id = id
        self._name = name
        self._description = description
        self._stage = stage
        self._filepath = filepath
        self._fileformat = filepath.splitext()[1].replace(".", "")
        self._version

        self._columns = Dataset.__columns
        self._primary_key = Dataset.__primary_key
        self._target_var = Dataset.__target_var
        self._text_var = Dataset.__text_var

        self._data = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

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
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.load()
        return self._data

    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        self._data = data

    def get_texts(self, with_id: bool = True) -> pd.DataFrame:
        self.load()
        if with_id:
            return self._data[[self._primary_key, self._text_var]]
        else:
            return self._data[self._text_var]

    def load(self, force: bool = False) -> None:
        if self._data is None or force:
            params = {"filepath": self._filepath}
            io = LoadCSV(name=self._name, params=params)
            self._data = io.execute()
        else:
            logger.info("Data already loaded. Set force=True to reload.")

    def save(self, force: bool = False) -> None:
        if not os.path.exists(self._filepath) or force:
            params = {"filepath": self._filepath}
            io = SaveCSV(name=self._name, params=params)
            io.execute()
        else:
            logger.info("Data already exiss. Set force=True to overwrite.")
