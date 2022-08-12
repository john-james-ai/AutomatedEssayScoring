#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Automated Neural Network Essay Scoring and Evaluation (DANNESE)                #
# Version    : 0.1.0                                                                               #
# Filename   : /__init__.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 2nd 2022 08:08:36 pm                                                 #
# Modified   : Wednesday August 10th 2022 08:57:32 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import re

specials = {
    r"tilde": "~",
    r"accent_grave": "`",
    r"ampersat": "@",
    r"hash": "#",
    r"dollar": "\\$",
    r"percent": "%",
    r"caret": "^",
    r"ampersand": "&",
    r"asterisk": "\\*",
    r"underscore": "_",
    r"plus": "\\+",
    r"equal": "=",
    r"pipe": "\\|",
    "backslash": re.escape("\\"),
    r"less_than": "<",
    r"greater_than": ">",
    r"forward_slash": "/",
}

punctuation = {
    r"exclamation point": "!",
    r"question mark": "\\?",
    r"period": "\\.",
    r"comma": ",",
    r"semicolon": ";",
    r"colon": ":",
    r"hypen": "-",
    r"open parenthesis": "\\(",
    r"close parenthesis": "\\)",
    r"open brace": "\\{",
    r"close brace": "\\}",
    r"open bracket": "\\[",
    r"close bracket": "\\]",
    r"quotation mark": '"',
    r"apostrophe": "'",
    r"elipses": "...",
}

controls = {"newline": "\n", "tab": "\t", "carriage return": "\r"}
