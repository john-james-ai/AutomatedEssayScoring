#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Automated Essay Scoring                                                             #
# Version    : 0.1.0                                                                               #
# Filename   : /frint.py                                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday August 3rd 2022 01:20:44 pm                                               #
# Modified   : Wednesday August 10th 2022 12:42:57 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Module for handling formatting and printing issues."""


def print_int(n: int) -> str:
    return f"{n:,}"


def print_float(n: float) -> str:
    return f"{n:,}"


def print_str(s: str) -> str:
    return s.replace("_", " ").title()
