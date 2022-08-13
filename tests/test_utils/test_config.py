#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Automated Essay Scoring: A Data-First Deep Learning Approach                        #
# Version    : 0.1.0                                                                               #
# Filename   : /test_config.py                                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday August 3rd 2022 03:24:48 am                                               #
# Modified   : Friday August 12th 2022 08:44:40 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #

import inspect
import pytest
import logging
import logging.config

# Enter imports for modules and classes being tested here
from aes.utils.config import LogConfig

# ------------------------------------------------------------------------------------------------ #
logging.config.dictConfig(LogConfig().config)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# ------------------------------------------------------------------------------------------------ #

# ================================================================================================ #
#                                    TEST SOMETHING                                                #
# ================================================================================================ #


@pytest.mark.config
class TestConfig:
    def test_config(self, caplog):
        logger.info("\nStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        config = LogConfig()
        data = config.read()
        assert isinstance(data, dict)

        logger.info("Completed {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
