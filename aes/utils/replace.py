#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Argument Effectiveness                                                              #
# Version    : 0.1.0                                                                               #
# Python     : 3.9.12                                                                              #
# Filename   : /replace.py                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/ArgumentEffectiveness                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 9th 2022 01:48:18 pm                                                 #
# Modified   : Tuesday August 9th 2022 01:52:54 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import fileinput
import sys

# ------------------------------------------------------------------------------------------------ #


def replacement(file, previousw, nextw):
    """Replaces a previous word with the next word in a file."""

    for line in fileinput.input(file, inplace=1):
        line = line.replace(previousw, nextw)
        sys.stdout.write(line)
