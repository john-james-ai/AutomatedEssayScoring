#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Automated Neural Network Essay Scoring and Evaluation (DANNESE)                #
# Version    : 0.1.0                                                                               #
# Python     : 3.9.12                                                                              #
# Filename   : /corpus.py                                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday August 10th 2022 11:24:50 pm                                              #
# Modified   : Thursday August 11th 2022 07:13:10 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Corpus Module: Collection of Documents, e.g. Essays"""
from abc import ABC, abstractmethod
import spacy
from spacy.tokens import DocBin
from datetime import datetime

# ------------------------------------------------------------------------------------------------ #


class Corpus(ABC):
    """Base class for a collection of documents in text or token form."""

    def __init__(self, name: str, stage: str, filepath: str) -> None:
        self._name = name
        self._stage = stage
        self._filepath = filepath
        self._format = None
        self._documents = None
        self._feature_space = None
        self._created = datetime.now()
        self._modified = datetime.now()
        self._saved = None

    @abstractmethod
    @property
    def document_count(self) -> int:
        pass

    def add_documents(self, documents: Union[pd.DataFrame, dict]) -> None:
        self._documents = documents

    def get_documents(self) -> Union[pd.DataFrame, dict]:
        return self._documents

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass


# ------------------------------------------------------------------------------------------------ #
class TextCorpus(Corpus):
    """Corpus in which the unit is the document and storage format is text."""

    def __init__(self, name: str, stage: str, filepath: str) -> None:
        super(TextCorpus, self).__init__(name=name, stage=stage, filepath=filepath)
        self._format = "csv"

    @property
    def document_count(self) -> int:
        return self._documents.shape[0]

    def load(self) -> None:
        """Loads the corpus from csv file format."""
        self._documents = pd.read_csv(self._filepath)

    def save(self) -> None:
        """Saves the corpus in csv file format."""
        self._documents.to_csv(self._filepath, index=False)
