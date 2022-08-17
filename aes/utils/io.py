#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Automated Essay Scoring: A Data-First Deep Learning Approach                        #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /io.py                                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday August 14th 2022 01:56:55 am                                                 #
# Modified   : Monday August 15th 2022 05:44:54 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
import os
import pickle
import pandas as pd
import yaml
from typing import Union

# ------------------------------------------------------------------------------------------------ #


class IO(ABC):
    @abstractmethod
    def read(self, filepath: str, **kwargs) -> Union[pd.DataFrame, dict]:
        pass

    @abstractmethod
    def write(self, data: Union[pd.DataFrame, dict], filepath: str, **kwargs) -> None:
        pass


# ------------------------------------------------------------------------------------------------ #


class CsvIO(IO):
    def read(self, filepath: str, **kwargs) -> Union[pd.DataFrame, dict]:

        sep = kwargs.get("sep", ",")
        encoding_errors = kwargs.get("encoding_errors", "strict")
        header = kwargs.get("header", "infer")
        names = kwargs.get("names", None)
        usecols = kwargs.get("usecols", None)
        nrows = kwargs.get("nrows", None)
        thousands = kwargs.get("thousands", ",")

        return pd.read_csv(
            filepath,
            encoding_errors=encoding_errors,
            sep=sep,
            header=header,
            names=names,
            usecols=usecols,
            nrows=nrows,
            thousands=thousands,
        )

    def write(self, data: Union[pd.DataFrame, dict], filepath: str, **kwargs) -> None:

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        sep = kwargs.get("sep", ",")
        header = kwargs.get("header", True)
        columns = kwargs.get("columns", None)
        index = kwargs.get("index", True)
        errors = kwargs.get("errors", "strict")
        encoding = kwargs.get("encoding", "utf-8")

        data.to_csv(
            filepath,
            sep=sep,
            header=header,
            columns=columns,
            index=index,
            errors=errors,
            encoding=encoding,
        )


# ------------------------------------------------------------------------------------------------ #


class YamlIO(IO):
    def read(self, filepath: str, **kwargs) -> Union[pd.DataFrame, dict]:
        with open(filepath, "r") as file:
            return yaml.SafeLoader(stream=file)

    def write(self, data: Union[pd.DataFrame, dict], filepath: str, **kwargs) -> None:

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as file:
            yaml.dump(data, file)


# ------------------------------------------------------------------------------------------------ #


class PickleIO(IO):
    def read(self, filepath: str, **kwargs) -> Union[pd.DataFrame, dict]:

        with open(filepath, "rb") as file:
            return pickle.load(file)

    def write(self, data: Union[pd.DataFrame, dict], filepath: str, **kwargs) -> None:

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "wb") as file:
            pickle.dump(data, file)


# ------------------------------------------------------------------------------------------------ #


class IOFactory:
    """IO Factory"""

    __io = {"csv": CsvIO(), "yml": YamlIO(), "pickle": PickleIO()}

    def io(self, fileformat: str) -> IO:
        try:
            return IOFactory.__io[fileformat]
        except KeyError as e:
            print("Format {} is not supported".format(fileformat))
            raise e
