#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Automated Essay Scoring: A Data-First Deep Learning Approach                        #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /acquisition.py                                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday August 12th 2022 02:14:38 am                                                 #
# Modified   : Friday August 12th 2022 03:05:06 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Data Acquisition Module"""
import os
import shlex
import subprocess
import zipfile
import logging
import logging.config
from typing import Any
from operations.workflow import Operator
from aes.utils.log_config import LOG_CONFIG

# ------------------------------------------------------------------------------------------------ #
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)
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
                subprocess.run(shlex.split(self._command), check=True, text=True, shell=False)
                destination_filepath = os.path.join(self._destination, self._filename)
                os.makedirs(self._destination, exist_ok=True)
                os.rename(self._filename, destination_filepath)
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
