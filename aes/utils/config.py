#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Automated Essay Scoring: A Data-First Deep Learning Approach                        #
# Version    : 0.1.0                                                                               #
# Filename   : /config.py                                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday July 29th 2022 12:41:04 am                                                   #
# Modified   : Monday August 15th 2022 05:42:39 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Configuration Module"""
from abc import ABC
import os
from dotenv import load_dotenv

from aes.utils.io import IOFactory

# ------------------------------------------------------------------------------------------------ #


class Config(ABC):
    """Base class defining read / write access to configuration files."""

    def __init__(self, name: str) -> None:
        self._filepath = None
        self._io = None
        self._config = None
        self._initialize(name)

    @property
    def config(self) -> dict:
        return self._config

    @config.setter
    def config(self, config: dict) -> None:
        self._config = config
        self.save()

    def load(self) -> list:
        """Reads data from a yaml file."""
        self._config = self._io.read(self._filepath)

    def save(self) -> None:
        """Writes config data to a yaml file."""
        self._io.write(data=self._config, filepath=self._filepath)

    def _initialize(self, name: str) -> None:
        """Initializes the Config object with an io object, a filepath, and the config data."""

        # Config filepaths are stored in the environment variables.
        load_dotenv()
        self._filepath = os.getenv(name)

        # Extract the file format from the filepath
        fileformat = os.path.splitext(self._filepath)[1].replace(".", " ")

        # Use the fileformat to obtain an io object.
        self._io = IOFactory().io(fileformat=fileformat)

        # Load the configuration data
        self._config = self._io.read(self._filepath)


# ------------------------------------------------------------------------------------------------ #
class FP2021Config(Config):

    __CONFIG_NAME = "CONFIG_DATA_FP2021"

    def __init__(self) -> None:
        name = FP2021Config.__CONFIG_NAME
        super(FP2021Config, self).__init__(name=name)


# ------------------------------------------------------------------------------------------------ #


class FP2022Config(Config):

    __CONFIG_NAME = "CONFIG_DATA_FP2022"

    def __init__(self) -> None:
        name = FP2022Config.__CONFIG_NAME
        super(FP2022Config, self).__init__(name=name)


# ------------------------------------------------------------------------------------------------ #


class LogConfig(Config):

    __CONFIG_NAME = "CONFIG_LOG"

    def __init__(self) -> None:
        config = LogConfig.__CONFIG_NAME
        super(LogConfig, self).__init__(config)


# ------------------------------------------------------------------------------------------------ #


class SpacyConfig(Config):

    __CONFIG_NAME = "CONFIG_SPACY"

    def __init__(self) -> None:
        self._config_name = SpacyConfig.__CONFIG_NAME
