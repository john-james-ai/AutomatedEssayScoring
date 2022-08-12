#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Argument Effectiveness                                                              #
# Version    : 0.1.0                                                                               #
# Filename   : /analysis.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/ArgumentEffectiveness                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 8th 2022 08:05:59 pm                                                  #
# Modified   : Tuesday August 9th 2022 02:16:42 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import os
import re
from PyPDF2 import PdfReader
import fitz
import logging
import pandas as pd
from tqdm import tqdm
from aes.utils.dates import pdfdatetime

# ------------------------------------------------------------------------------------------------ #
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# ------------------------------------------------------------------------------------------------ #


class PDFCollection:
    """PDF Collection for analysis

    Args:
        folder (str): The directory containing the documents to analyze
    """

    def __init__(self, folder: str) -> None:
        self._folder = folder
        self._data = {}
        self._analysis = pd.DataFrame()

    @property
    def analysis(self) -> pd.DataFrame:
        return self._analysis

    def analyze(self, keyword_file: str = None) -> pd.DataFrame:

        self._keywords = pd.read_csv(keyword_file, sep=",", header=0)["keywords"].to_list()

        papers = os.listdir(self._folder)
        docbar = tqdm(papers)
        for title in docbar:
            self._format_doc_data()
            docbar.set_description("Processing: {}".format(title))
            filepath = os.path.join(self._folder, title)
            try:
                df = self._pypdf2(filepath, title)
            except TypeError as e:
                df = self._pymupdf(filepath, title)
                logger.error("Error processing {}\n{}".format(title, e))
            self._analysis = pd.concat([self._analysis, df], axis=0)

    def save(self, filepath: str, what: str = "analysis") -> None:
        if what.startswith("analysis"):
            if self._analysis.shape[0] > 0:
                self._analysis.to_csv(filepath, sep=",", header=True)
            else:
                logger.info("Nothing {} document to save.".format(what))
        else:
            raise ValueError("{} is an unrecognized document.".format(what))

    def load(self, filepath: str, what: str = "analysis") -> None:
        if what.startswith("analysis"):
            self._analysis = pd.read_csv(filepath, sep=",")
        else:
            raise ValueError("{} is an unrecognized document.".format(what))

    def _format_doc_data(self) -> None:
        self._data = {}
        self._data["titles"] = []
        self._data["created"] = []
        self._data["pages"] = []
        for keyword in self._keywords:
            self._data[keyword] = 0

    def _pypdf2(self, filepath: str, title: str):

        reader = PdfReader(filepath)
        pagebar = tqdm(reader.pages)
        metadata = reader.metadata
        self._data["titles"].append(title)
        self._data["created"].append(pdfdatetime(metadata["/CreationDate"]))
        self._data["pages"].append(len(reader.pages))

        for page in pagebar:
            pagebar.update(1)
            text = page.extract_text() + "\n"
            for keyword in self._keywords:
                pattern = r"\b" + keyword
                self._data[keyword] += len(re.findall(pattern, text))
        df = pd.DataFrame(self._data, index=[0])
        return df

    def _pymupdf(self, filepath: str, title: str):

        with fitz.open(filepath) as doc:

            self._data["titles"].append(title)
            self._data["created"].append(pdfdatetime(doc.metadata["creationDate"]))
            self._data["pages"].append(doc.page_count)
            pagebar = tqdm(doc)
            for page in pagebar:
                pagebar.update(1)
                text = page.get_text() + "\n"
                for keyword in self._keywords:
                    pattern = r"\b" + keyword
                    self._data[keyword] += len(re.findall(pattern, text))
            df = pd.DataFrame(self._data, index=[0])
            return df
