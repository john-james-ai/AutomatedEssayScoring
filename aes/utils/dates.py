#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Tool Shack                                                                          #
# Version    : 0.1.0                                                                               #
# Filename   : /dates.py                                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/ToolShack                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 9th 2022 01:03:19 am                                                 #
# Modified   : Tuesday August 9th 2022 04:50:53 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Module for converting dates."""
import datetime
import re
from dateutil.tz import tzutc, tzoffset

# ------------------------------------------------------------------------------------------------ #

pdf_date_pattern = re.compile(
    "".join(
        [
            r"(D:)?",
            r"(?P<year>\d\d\d\d)",
            r"(?P<month>\d\d)",
            r"(?P<day>\d\d)",
            r"(?P<hour>\d\d)",
            r"(?P<minute>\d\d)",
            r"(?P<second>\d\d)",
            r"(?P<tz_offset>[+-zZ])?",
            r"(?P<tz_hour>\d\d)?",
            r"'?(?P<tz_minute>\d\d)?'?",
        ]
    )
)


def pdfdatetime(date_str):
    """Convert a pdf date such as "D:20120321183444+07'00'" i.e. '(D:YYYYMMDDHHmmSSOHH'mm')'into a usable datetime object.
    Sources: https://stackoverflow.com/questions/16503075/convert-creationtime-of-pdf-to-a-readable-format-in-python
             http://www.verypdf.com/pdfinfoeditor/pdf-date-format.htm

    Args:
        param (str): pdf date string

    Returns:
        datetime object
    """
    global pdf_date_pattern
    match = re.match(pdf_date_pattern, date_str)
    if match:
        date_info = match.groupdict()

        for k, v in date_info.items():  # transform values
            if v is None:
                pass
            elif k == "tz_offset":
                date_info[k] = v.lower()  # so we can treat Z as z
            else:
                date_info[k] = int(v)

        if date_info["tz_offset"] in ("z", None):  # UTC
            date_info["tzinfo"] = tzutc()
        else:
            multiplier = 1 if date_info["tz_offset"] == "+" else -1
            date_info["tzinfo"] = tzoffset(
                None, multiplier * (3600 * date_info["tz_hour"] + 60 * date_info["tz_minute"])
            )

        for k in ("tz_offset", "tz_hour", "tz_minute"):  # no longer needed
            del date_info[k]

        return datetime.datetime(**date_info)
