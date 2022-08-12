#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Automated Neural Network Essay Scoring and Evaluation (DANNESE)                #
# Version    : 0.1.0                                                                               #
# Filename   : /config.py                                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday July 29th 2022 12:41:04 am                                                   #
# Modified   : Wednesday August 10th 2022 03:23:00 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import yaml

# ------------------------------------------------------------------------------------------------ #


class Config:
    """Provides basic io for Yaml files.

    Args:
        filepath (str): The configuration filepath (optional). Defaults
            to the class member variable.

    """

    def __init__(self, filepath: str) -> None:
        self._filepath = filepath

    def read(self) -> list:
        """Reads data from a yaml file.

        Args:
            key (str): The (optional) key for the corresponding value to return.
        """

        with open(self._filepath) as file:
            return yaml.load(file, Loader=yaml.SafeLoader)

    def write(self, data: dict) -> None:
        """Writes data to a yaml file.

        Args:
            data (dict): The data to be written to the file.
        """
        with open(self._filepath, "w") as file:
            yaml.dump(data, file, sort_keys=False, default_flow_style=False)
