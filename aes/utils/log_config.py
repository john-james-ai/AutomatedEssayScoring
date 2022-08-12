#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Automated Neural Network Essay Scoring and Evaluation (DANNESE)                #
# Version    : 0.1.0                                                                               #
# Filename   : /log_config.py                                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday August 3rd 2022 03:25:47 am                                               #
# Modified   : Thursday August 11th 2022 07:57:55 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "multiprocess": {
            "format": "%(levelname)s | %(asctime)s | %(module)s | %(process)d  | %(thread)d  | %(message)s"
        },
        "verbose": {"format": "%(levelname)s | %(asctime)s | %(module)s | %(message)s"},
        "standard": {"format": "%(levelname)s | %(asctime)s | %(message)s"},
        "simple": {"format": "%(message)s"},
    },
    "handlers": {
        "console": {"level": "INFO", "class": "logging.StreamHandler", "formatter": "simple"},
        "logfile": {
            "level": "DEBUG",
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "verbose",
            "filename": "logs/aes.log",
            "when": "D",
            "backupCount": 3,
        },
        "eventfile": {
            "level": "INFO",
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "verbose",
            "filename": "logs/event.log",
            "when": "D",
            "backupCount": 3,
        },
    },
    "loggers": {
        "root": {"handlers": ["console", "logfile"], "propagate": False, "level": "DEBUG"},
        "event": {"handlers": ["eventfile"], "propagate": False, "level": "INFO"},
    },
}
