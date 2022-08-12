#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Automated Neural Network Essay Scoring and Evaluation (DANNESE)                #
# Version    : 0.1.0                                                                               #
# Filename   : /test_config.py                                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday August 3rd 2022 03:24:48 am                                               #
# Modified   : Wednesday August 10th 2022 03:32:40 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #

import inspect
import pytest
import logging
import logging.config

# Enter imports for modules and classes being tested here
from aes.utils.config import Config
from aes.utils.log_config import LOG_CONFIG

# ------------------------------------------------------------------------------------------------ #
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #

# ================================================================================================ #
#                                    TEST SOMETHING                                                #
# ================================================================================================ #


@pytest.mark.config
class TestConfig:
    # def test_data_config(self, caplog):
    #     logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    #     raw_train_essays = "./data/fpe2021/0_raw/train"
    #     raw_test_essays = "./data/fpe2021/0_raw/test"

    #     raw_train_discourse = "./data/fpe2021/0_raw/train.csv"
    #     raw_test_discourse = "./data/fpe2021/0_raw/test.csv"

    #     config = DataConfig()
    #     assert raw_train_essays == config.raw_train_essays
    #     assert raw_test_essays == config.raw_test_essays

    #     assert raw_train_discourse == config.raw_train_discourse
    #     assert raw_test_discourse == config.raw_test_discourse

    #     logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_visual_config(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        filepath = "tests/test_utils/config.yaml"

        config = Config(filepath=filepath)
        data = config.read()
        assert isinstance(data, dict)
        data["visual"]["palette"] = "Glues_d"
        config.write(data)
        data = config.read()
        data["visual"]["palette"] == "Glues_d"

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

    def test_data_config(self, caplog):
        logger.info("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        filepath = "aes/data/config.yaml"

        config = Config(filepath=filepath)
        data = config.read()
        assert isinstance(data, dict)
        assert data["columns"]["idvar"] == "discourse_id"

        logger.info("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
