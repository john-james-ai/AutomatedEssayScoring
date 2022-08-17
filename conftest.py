#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Automated Essay Scoring: A Data-First Deep Learning Approach                        #
# Version    : 0.1.0                                                                               #
# Filename   : /conftest.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday July 29th 2022 02:25:41 am                                                   #
# Modified   : Wednesday August 17th 2022 12:40:50 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Includes fixtures, classes and functions supporting testing."""
import pytest
import pandas as pd

# ------------------------------------------------------------------------------------------------ #
TEST_DATA_FILEPATH = "tests/data/train.csv"
# ------------------------------------------------------------------------------------------------ #
#                                        IGNORE                                                    #
# ------------------------------------------------------------------------------------------------ #
collect_ignore_glob = ["tests/old_tests/**/*.py"]
# ------------------------------------------------------------------------------------------------ #
#                                          SOURCE                                                  #
# ------------------------------------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(TEST_DATA_FILEPATH)


@pytest.fixture(scope="module")
def test_data_foldere():
    return "tests/testdata"
