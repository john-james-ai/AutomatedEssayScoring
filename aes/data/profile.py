#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Automated Essay Scoring: A Data-First Deep Learning Approach                        #
# Version    : 0.1.0                                                                               #
# Filename   : /profile.py                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday August 3rd 2022 05:21:41 am                                               #
# Modified   : Friday August 12th 2022 11:39:06 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
import os
from datetime import datetime
from abc import ABC, abstractmethod
import re
import pandas as pd
import numpy as np
import logging
import logging.config
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy
import spacy
from spacy.tokens import Doc

# ------------------------------------------------------------------------------------------------ #
from aes.utils.config import LogConfig, SpacyConfig
from aes.data.dataset import Dataset

# ------------------------------------------------------------------------------------------------ #
logging.config.dictConfig(LogConfig().config)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


class Profile:
    """Profiles a Corpus

    Args:
        dataset (Dataset): The Dataset object the profile represents
        metadata (pd.DataFrame): Token level metadata

    """

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._token_data = None
        self._created = datetime.now()

    @property
    def dataset_name(self) -> str:
        return self._dataset.name

    @property
    def filepath(self) -> str:
        return self._dataset.filepath


# ------------------------------------------------------------------------------------------------ #


class ProfileBuilder:
    """Constructs a Profile for a dataset"""

    def __init__(self) -> None:


        self.reset()
        self._get_config()

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._reset()

    @property
    def profile(self) -> Profile:
        return self._profile

    def build(self) -> None:
        """Obtains data from dataset and orchestrates build process."""

        # Convert texts to a list of tuples of the format (text,{'discourse_id': discourse_id}).
        # The second tuple element will be added to the spacy document as context.
        texts = self._get_texts_with_metadata()

        # Run the pipeline and add 'discourse_id' to document object as custom attribute.
        docs = self._run_pipeline(texts)

        # Create token_level metadata from data extracted from the doc objects.
        token_data = self._extract_token_data(docs)

    def reset(self) -> None:
        """Resets the profile object."""
        self._dataset = None
        self._profile = None
        self._model = None
        self._pipeline = None
        self._token_attributes = None

        self._token_data = None
        self._sentence_data = None

    def _get_config(self) -> None:
        """Obtains spaCy model and pipeline configurations."""
        config = SpacyConfig().config
        try:
            self._model = config["models"]["trained"]
            self._pipeline = config["pipelines"]["profile"]['components']
            self._token_attributes = config['pipelines']['profile']['token_attributes']
        except KeyError as e:
            logger.error("The required trained model not found in spaCy configuration file.")
            raise (e)

    def _get_texts_with_metadata(self) -> list:
        """Returns text as a list of tuples including text and 'discourse_id'.

        Each document is uniquely identified by a 'discourse_id'. To add this identifier
        to the spaCy document object created by the pipe object, we must provide
        the data in (text, {'discourse_id': discourse_id}) format. This will
        add the identifier information to the document object context.

        Source:https://spacy.io/usage/processing-pipelines

        """
        texts = self._dataset.get_texts(with_id=True)
        recs = texts.to_dict(orient="records")
        texts = []
        for d in recs:
            text = (d["discourse_text"], {"discourse_id": d["discourse_id"]})
            texts.append(text)
        return texts

    def _run_pipeline(self, texts: list) -> None:
        """Executes a spaCy pipeline."""

        # Load the trained model and run the pipeline
        nlp = spacy.load(self._model)
        doc_tuples = nlp.pipe(texts, as_tuples=True)

        # Add the 'discourse_id' from context to the document object.
        docs = []
        for doc, context in doc_tuples:
            doc._.discourse_id = context[
                "discourse_id"
            ]  # The underscore is required for the addition of custom attributes.
            docs.append(doc)

        return docs

    def _extract_token_data(self, docs: list) -> pd.DataFrame:
        """Extracts token data from each document and creates token level data frame."""
        doc_arrays = []
        for doc in docs:
            doc.to_array(attr_ids = self._token_attributes)
        return [
        (d.doc._.discourse_id,
        d.i,
        d.vocab,
        d.text,
        d.sent,
        d.is_alpha,
        d.is_ascii,
        d.is_digit,
        d.is_lower,
        d.is_upper,
        d.is_title,
        d.is_punct,
        d.is_left_punct,
        d.is_right_punct,
        d.is_sent_start,
        d.is_sent_end,
        d.is_space,
        d.is_bracket,
        d.is_quote,
        d.is_currency,
        d.like_url,
        d.like_num,
        d.like_email,
        d.is_oov,
        d.is_stop) for d in doc:
