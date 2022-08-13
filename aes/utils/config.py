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
# Modified   : Friday August 12th 2022 08:42:27 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Configuration Module"""
from abc import ABC
import os
from dotenv import load_dotenv
import yaml

# ------------------------------------------------------------------------------------------------ #


class Config(ABC):
    """Base class defining read / write access to configuration files."""

    __CONFIG_NAME = None

    def __init__(self) -> None:
        self._config = None

    @property
    def config(self) -> dict:
        return self.read()

    def get_config_filepath(self) -> str:
        load_dotenv()
        return os.getenv(self._config_name)

    def read(self) -> list:
        """Reads data from a yaml file.

        Args:
            key (str): The (optional) key for the corresponding value to return.
        """
        filepath = self.get_config_filepath()

        with open(filepath) as file:
            return yaml.load(file, Loader=yaml.SafeLoader)

    def write(self, data: dict) -> None:
        """Writes data to a yaml file.

        Args:
            data (dict): The data to be written to the file.
        """
        filepath = self.get_config_filepath()

        with open(filepath, "w") as file:
            yaml.dump(data, file, sort_keys=False, default_flow_style=False)


# ------------------------------------------------------------------------------------------------ #
class DataConfig(Config):

    __CONFIG_NAME = "CONFIG_DATA"

    def __init__(self) -> None:
        self._config_name = DataConfig.__CONFIG_NAME


# ------------------------------------------------------------------------------------------------ #
class LogConfig(Config):

    __CONFIG_NAME = "CONFIG_LOG"

    def __init__(self) -> None:
        self._config_name = LogConfig.__CONFIG_NAME


# ------------------------------------------------------------------------------------------------ #
class SpacyConfig(Config):

    __CONFIG_NAME = "CONFIG_SPACY"

    def __init__(self) -> None:
        self._config_name = SpacyConfig.__CONFIG_NAME
