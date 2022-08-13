#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Automated Essay Scoring: A Data-First Deep Learning Approach                        #
# Version    : 0.1.0                                                                               #
# Python     : 3.9.12                                                                              #
# Filename   : /test_length.py                                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday August 10th 2022 04:15:45 pm                                              #
# Modified   : Friday August 12th 2022 08:43:33 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import inspect
import pytest
import logging
import logging.config

from aes.utils.config import LogConfig
from aes.features.extraction.base import FeatureExtractorFactory

# ------------------------------------------------------------------------------------------------ #
logging.config.dictConfig(LogConfig().config)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# ------------------------------------------------------------------------------------------------ #

# ================================================================================================ #
#                                    TEST SOMETHING                                                #
# ================================================================================================ #


@pytest.mark.length
class TestLengthFeatureExtractors:
    def test_length_features(self, caplog, data):
        logger.debug("\tStarted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))

        f = FeatureExtractorFactory()
        extractors = f.list_extractors()
        for e in extractors:
            logger.debug("Extracting {}".format(e))
            extractor = f.create_extractor(name=e)
            assert extractor.name == e
            features = extractor.extract(data=data)
            assert features.shape[0] == 1000

        logger.debug("\tCompleted {} {}".format(self.__class__.__name__, inspect.stack()[0][3]))
