#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Automated Essay Scoring: A Data-First Deep Learning Approach                        #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /test_io.py                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 15th 2022 05:23:48 pm                                                 #
# Modified   : Monday August 15th 2022 05:43:16 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import pytest
import pandas as pd

# Enter imports for modules and classes being tested here
from aes.utils.io import IOFactory, CsvIO, YamlIO, PickleIO

# ------------------------------------------------------------------------------------------------ #


@pytest.mark.io
class TestIO:
    def test_csv_io(self, caplog, data):
        fileformat = "csv"
        output_filepath = "tests/output/io/test.csv"
        io = IOFactory.io(fileformat=fileformat)
        assert isinstance(io, CsvIO)

        io.write(data=data, filepath=output_filepath)
        df = io.read(filepath=output_filepath)
        assert isinstance(df, pd.DataFrame)

    def test_yml_io(self, caplog, data):
        fileformat = "yml"
        output_filepath = "tests/output/io/test.yml"
        io = IOFactory.io(fileformat=fileformat)
        assert isinstance(io, YamlIO)

        io.write(data=data, filepath=output_filepath)
        df = io.read(filepath=output_filepath)
        assert isinstance(df, pd.DataFrame)

    def test_pickle_io(self, caplog, data):
        fileformat = "pickle"
        output_filepath = "tests/output/io/test.pickle"
        io = IOFactory.io(fileformat=fileformat)
        assert isinstance(io, PickleIO)

        io.write(data=data, filepath=output_filepath)
        df = io.read(filepath=output_filepath)
        assert isinstance(df, pd.DataFrame)
