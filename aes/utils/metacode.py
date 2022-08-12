#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Automated Neural Network Essay Scoring and Evaluation (DANNESE)                #
# Version    : 0.1.0                                                                               #
# Python     : 3.9.12                                                                              #
# Filename   : /metacode.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday August 10th 2022 06:19:33 pm                                              #
# Modified   : Wednesday August 10th 2022 06:26:15 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import re
import os

# ------------------------------------------------------------------------------------------------ #


def class_list_from_file(filepath) -> list:
    """Returns a list of classes in a file designated by its filepath

    Args:
        filepath: The path to the file.

    Returns:
        list containing class names in the file.

    """

    with open(filepath) as file:
        text = file.read()
    return re.findall(r"(?<=\bclass\s)(\w+)", text)


def main():
    print(os.getcwd())
    filepath = "../../aes/features/extraction/length.py"
    print(class_list_from_file(filepath))


if __name__ == "__main__":
    main()
