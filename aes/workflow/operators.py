#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Automated Essay Scoring: A Data-First Deep Learning Approach                        #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /operators.py                                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday August 11th 2022 09:43:52 pm                                               #
# Modified   : Friday August 12th 2022 08:43:33 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Operator Module"""
import os
import shlex
import subprocess
import zipfile
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from typing import Any
import logging
import logging.config

from aes.utils.config import LogConfig

# ------------------------------------------------------------------------------------------------ #
logging.config.dictConfig(LogConfig().config)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
#                                OPERATOR BASE CLASS                                               #
# ------------------------------------------------------------------------------------------------ #


class Operator(ABC):
    """Abstract class for operator classes

    Args:
        seq (int): Sequence number of operation in a pipeline.
        params (Any): Parameters for the operation.

    Class Variables:
        __name (str): The human-reedable name for the operator
        __desc (str): String describing what the operator does.

    """

    __name = "operator_base_class"
    __desc = "Describes the interface for all operator subclasses."

    def __init__(self, name: str = None, params: dict = {}) -> None:
        self._name = name or Operator.__name
        self._params = params

        self._desc = Operator.__desc

        self._created = datetime.now()
        self._started = None
        self._stopped = None
        self._duration = None

    def __str__(self) -> str:
        return str(
            "Sequence #: {}\tOperator: {}\t{}\tParams: {}".format(
                self._seq, Operator.__name, Operator.__desc, self._params
            )
        )

    def run(self, data: Any = None, context: dict = {}) -> Any:
        self._setup()
        data = self.execute(data=data, context=context)
        self._teardown()
        return data

    @abstractmethod
    def execute(self, data: Any = None, context: dict = {}) -> Any:
        pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> Any:
        return self._params

    @property
    def created(self) -> datetime:
        return self._created

    @property
    def started(self) -> datetime:
        return self._started

    @property
    def stopped(self) -> datetime:
        return self._stopped

    @property
    def duration(self) -> datetime:
        return self._duration

    def _setup(self) -> None:
        self._started = datetime.now()

    def _teardown(self) -> None:
        self._stopped = datetime.now()
        self._duration = round((self._stopped - self._started).total_seconds(), 4)


# ------------------------------------------------------------------------------------------------ #
#                                   FILE OPERATORS                                                 #
# ------------------------------------------------------------------------------------------------ #


class DownloaderAPI(Operator):
    def __init__(self, name: str = None, params: dict = {}) -> None:
        super(DownloaderAPI, self).__init__(name=name, params=params)
        self._command = params.get("command", None)
        self._filename = params.get("filename", None)
        self._destination = params.get("destination", None)
        self._force = params.get("force", None)

    def execute(self, data: Any = None, context: dict = {}) -> Any:
        """Downloads compressed data via an API using bash

        Args:
            data: Not used
            context: not used.
        """
        if self._force or not os.path.exists(self._destination):
            try:
                os.makedirs(self._destination, exist_ok=True)
                subprocess.run(shlex.split(self._command), check=True, text=True, shell=False)
            except subprocess.CalledProcessError as e:
                logger.error(e.output)


# ------------------------------------------------------------------------------------------------ #


class ExtractZip(Operator):
    def __init__(self, name: str = None, params: dict = {}) -> None:
        super(ExtractZip, self).__init__(name=name, params=params)
        self._source = params.get("source", None)
        self._destination = params.get("destination", None)
        self._force = params.get("force", None)

    def execute(self, data: Any = None, context: dict = {}) -> Any:
        """Decompresses a zipfile from source and stores contents at destination.

        Args:
            data: Not used
            context: not used.
        """
        if self._force or not os.path.exists(self._destination):
            os.makedirs(self._destination, exist_ok=True)

            with zipfile.ZipFile(self._source, "r") as zf:
                zf.extractall(self._destination)


# ------------------------------------------------------------------------------------------------ #


class LoadCSV(Operator):
    def __init__(self, name: str = None, params: dict = {}) -> None:
        super(LoadCSV, self).__init__(name=name, params=params)
        self._filepath = params.get("filepath", None)
        self._encoding_errors = params.get("encoding_errors", "strict")
        self._retry_ignore_encoding_errors = params.get("retry_ignore_encoding_errors", True)

    def execute(self, data: Any = None, context: dict = {}) -> Any:
        """Loads data from a csv file into a DataFrame

        Args:
            data: Not used
            context: not used.
        Returns:
            DataFrame
        """
        try:
            return pd.read_csv(self._filepath, encoding_errors=self._encoding_errors)

        except UnicodeError as e:
            logger.error("Encoding error with {} error handling".format(self._encoding_errors))
            if self._retry_ignore_encoding_errors:
                logger.info("Retrying with encoding_errors='ignore'")
                return pd.read_csv(self._filepath, encoding_errors="ignore")
            else:
                raise (e)

        except FileNotFoundError as e:
            logger.error("File {} not found.".format(self._filepath))
            raise (e)


# ------------------------------------------------------------------------------------------------ #


class SaveCSV(Operator):
    def __init__(self, name: str = None, params: dict = {}) -> None:
        super(SaveCSV, self).__init__(name=name, params=params)
        self._filepath = params.get("filepath", None)

    def execute(self, data: Any = None, context: dict = {}) -> Any:
        """Loads data from a csv file into a DataFrame

        Args:
            data (DataFrame): Contains data to be saved.
            context: not used.

        """

        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)
        data.to_csv(self._filepath, header=True)


# ------------------------------------------------------------------------------------------------ #
#                                  DATA OPERATORS                                                  #
# ------------------------------------------------------------------------------------------------ #


class Encoder(Operator):
    def __init__(self, name: str = None, params: dict = {}) -> None:
        super(Encoder, self).__init__(name=name, params=params)
        self._encoding = params.get("encoding", "utf-8")
        self._column = params.get("column", None)

    def execute(self, data: Any = None, context: dict = {}) -> Any:
        """Loads data from a csv file into a DataFrame

        Args:
            data (DataFrame): Contains data to be encoded.
            context: not used.

        """
        try:
            data[self._column] = map(lambda x: x.encode("utf-8", "strict"), data[self._column])
            return data
        except UnicodeEncodeError as e:
            logger.error("Error encoding data")
            raise (e)
