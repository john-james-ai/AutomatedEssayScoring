#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Automated Essay Scoring: A Data-First Deep Learning Approach                        #
# Version    : 0.1.0                                                                               #
# Python     : 3.9.12                                                                              #
# Filename   : /length.py                                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 9th 2022 08:44:14 pm                                                 #
# Modified   : Monday August 15th 2022 04:59:41 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Defines Length Textual Features and Behaviors for Extraction and Summarization"""
#%%
import re
import pandas as pd
import numpy as np
import statistics
import logging
import logging.config
from nltk.tokenize import sent_tokenize, word_tokenize

# ------------------------------------------------------------------------------------------------ #
from aes.utils.config import LogConfig
from aes.data import specials, punctuation
from aes.features.extraction.base import FeatureExtractor

# ------------------------------------------------------------------------------------------------ #
logging.config.dictConfig(LogConfig().config)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
#                                  ALPHABETIC CHARACTERS                                           #
# ------------------------------------------------------------------------------------------------ #


class AlphaCharacters(FeatureExtractor):
    """Counts the number of alphabetic characters"""

    def __init__(self) -> None:
        super(AlphaCharacters, self).__init__()
        self._name = "alphabetic_character_count"
        self._category = "length"

    def extract(self, data: pd.DataFrame, idvar: str = "discourse_id") -> None:
        """Extracts the feature from the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the text data to be analyzed.
        """
        return data[self._text_col].apply(lambda x: len([i for i in x if i.isalpha()]))


# ------------------------------------------------------------------------------------------------ #
#                                    NUMBER CHARACTERS                                             #
# ------------------------------------------------------------------------------------------------ #
class NumberCharacters(FeatureExtractor):
    """Counts the number of numeric characters"""

    def __init__(self) -> None:
        super(NumberCharacters, self).__init__()
        self._name = "number_character_count"
        self._category = "length"

    def extract(self, data: pd.DataFrame, **kwargs) -> None:
        """Extracts the feature from the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the text data to be analyzed.
        """
        return data[self._text_col].apply(lambda x: len([x for x in x.split() if x.isdigit()]))


# ------------------------------------------------------------------------------------------------ #
#                                    SPECIAL CHARACTERS                                            #
# ------------------------------------------------------------------------------------------------ #
class SpecialCharacters(FeatureExtractor):
    """Counts the number of special characters"""

    def __init__(self) -> None:
        super(SpecialCharacters, self).__init__()
        self._name = "special_character_count"
        self._category = "length"
        self._counts_dict = {}

    def extract(self, data: pd.DataFrame, **kwargs) -> None:
        """Extracts the feature from the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the text data to be analyzed.
        """
        self._initialize_counts()
        return data[self._text_col].apply(lambda x: self._count_feature(x))
        self._counts = pd.DataFrame.from_dict(
            data=self._counts_dict, orient="index", columns=["Count"]
        ).sort_values(by="Count", ascending=False)

    def _initialize_counts(self) -> None:
        for k, v in specials.items():
            self._counts_dict[k] = 0

    def _count_feature(self, discourse: str) -> int:
        total = 0
        for k, v in specials.items():
            count = len(re.findall(v, discourse))
            self._counts_dict[k] += count
            total += count
        return total


# ------------------------------------------------------------------------------------------------ #
#                                      PUNCTUATION                                                 #
# ------------------------------------------------------------------------------------------------ #
class Punctuation(FeatureExtractor):
    """Counts punctuation marks"""

    def __init__(self) -> None:
        super(Punctuation, self).__init__()
        self._name = "punctuation_count"
        self._category = "length"
        self._counts_dict = {}

    def extract(self, data: pd.DataFrame, **kwargs) -> None:
        """Extracts the feature from the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the text data to be analyzed.
        """
        self._initialize_counts()
        return data[self._text_col].apply(lambda x: self._count_feature(x))
        self._counts = pd.DataFrame.from_dict(
            data=self._counts_dict, orient="index", columns=["Count"]
        ).sort_values(by="Count", ascending=False)

    def _initialize_counts(self) -> None:
        for k, v in punctuation.items():
            self._counts_dict[k] = 0

    def _count_feature(self, discourse: str) -> int:
        total = 0
        for k, v in punctuation.items():
            count = len(re.findall(v, discourse))
            self._counts_dict[k] += count
            total += count
        return total


# ------------------------------------------------------------------------------------------------ #
#                                        COMMAS                                                    #
# ------------------------------------------------------------------------------------------------ #
class Commas(FeatureExtractor):
    """Counts commas"""

    def __init__(self) -> None:
        super(Commas, self).__init__()
        self._name = "commas_count"
        self._category = "length"

    def extract(self, data: pd.DataFrame, **kwargs) -> None:
        """Extracts the feature from the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the text data to be analyzed.
        """
        return data[self._text_col].apply(lambda x: self._count_feature(x))

    def _count_feature(self, discourse: str) -> int:
        pattern = punctuation["comma"]
        return len(re.findall(pattern, discourse))


# ------------------------------------------------------------------------------------------------ #
#                                     QUESTION MARKS                                               #
# ------------------------------------------------------------------------------------------------ #
class QuestionMarks(FeatureExtractor):
    """Counts commas"""

    def __init__(self) -> None:
        super(QuestionMarks, self).__init__()
        self._name = "question_mark_count"
        self._category = "length"

    def extract(self, data: pd.DataFrame, **kwargs) -> None:
        """Extracts the feature from the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the text data to be analyzed.
        """
        return data[self._text_col].apply(lambda x: self._count_feature(x))

    def _count_feature(self, discourse: str) -> int:
        pattern = punctuation["question mark"]
        return len(re.findall(pattern, discourse))


# ------------------------------------------------------------------------------------------------ #
#                                        WORD COUNT                                                #
# ------------------------------------------------------------------------------------------------ #
class WordCount(FeatureExtractor):
    """Counts Words. Also provides counts for words exceeding a minimum length."""

    def __init__(self) -> None:
        super(WordCount, self).__init__()
        self._name = "word_count"
        self._category = "length"

    def extract(self, data: pd.DataFrame, min_length: int = 0) -> None:
        """Extracts the feature from the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the text data to be analyzed.
        """
        return data[self._text_col].apply(lambda x: self._count_features(x, min_length))

    def _count_features(self, x: str, min_length: int = 0) -> int:
        tokens = str(x).split(" ")
        return len([token for token in tokens if len(token) > min_length])


# ------------------------------------------------------------------------------------------------ #
#                                      AVG WORD LENGTH                                             #
# ------------------------------------------------------------------------------------------------ #
class AvgWordLength(FeatureExtractor):
    """Counts Words. Also provides counts for words exceeding a minimum length."""

    def __init__(self) -> None:
        super(AvgWordLength, self).__init__()
        self._name = "avg_word_length"
        self._category = "length"

    def extract(self, data: pd.DataFrame, min_length: int = 0) -> None:
        """Extracts the feature from the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the text data to be analyzed.
        """
        return data[self._text_col].apply(lambda x: self._average_word_length(x))

    def _average_word_length(self, discourse: str) -> float:
        words = discourse.split(" ")
        return sum(len(word) for word in words) / len(words)


# ------------------------------------------------------------------------------------------------ #
#                                      STD WORD LENGTH                                             #
# ------------------------------------------------------------------------------------------------ #
class StdWordLength(FeatureExtractor):
    """Counts Words. Also provides counts for words exceeding a minimum length."""

    def __init__(self) -> None:
        super(StdWordLength, self).__init__()
        self._name = "std_word_length"
        self._category = "length"

    def extract(self, data: pd.DataFrame, min_length: int = 0) -> None:
        """Extracts the feature from the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the text data to be analyzed.
        """
        return data[self._text_col].apply(lambda x: np.std(self._word_length(x)))

    def _word_length(self, discourse: str) -> float:
        words = discourse.split(" ")
        return [len(word) for word in words]


# ------------------------------------------------------------------------------------------------ #
#                                       VOCABULARY SIZE                                            #
# ------------------------------------------------------------------------------------------------ #
class VocabularySize(FeatureExtractor):
    """Counts words in the vocabulary."""

    def __init__(self) -> None:
        super(VocabularySize, self).__init__()
        self._name = "vocabulary_size"
        self._category = "length"

    def extract(self, data: pd.DataFrame, **kwargs) -> None:
        """Extracts the feature from the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the text data to be analyzed.
        """
        return data[self._text_col].apply(lambda x: self._count_vocabulary(x))

    def _count_vocabulary(self, discourse: str) -> int:
        tokens = discourse.split(" ")
        return len(set(tokens))


# ------------------------------------------------------------------------------------------------ #
#                                       SENTENCE COUNT                                             #
# ------------------------------------------------------------------------------------------------ #
class SentenceCount(FeatureExtractor):
    """Counts sentences in the discourse."""

    def __init__(self) -> None:
        super(SentenceCount, self).__init__()
        self._name = "sentence_count"
        self._category = "length"

    def extract(self, data: pd.DataFrame, **kwargs) -> None:
        """Extracts the feature from the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the text data to be analyzed.
        """
        return data[self._text_col].apply(lambda x: len(sent_tokenize(x)))


# ------------------------------------------------------------------------------------------------ #
#                                    AVG SENTENCE LENGTH                                           #
# ------------------------------------------------------------------------------------------------ #
class AvgSentenceLength(FeatureExtractor):
    """Counts sentences in the discourse."""

    def __init__(self) -> None:
        super(AvgSentenceLength, self).__init__()
        self._name = "avg_sentence_length"
        self._category = "length"

    def extract(self, data: pd.DataFrame, **kwargs) -> None:
        """Extracts the feature from the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the text data to be analyzed.
        """
        return data[self._text_col].apply(lambda x: self._ave_sentence_length(x))

    def _ave_sentence_length(self, discourse: str) -> float:
        sentences = sent_tokenize(discourse)
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        if len(sentence_lengths) < 2:
            return sentence_lengths[0]
        else:
            return statistics.stdev(sentence_lengths)


# ------------------------------------------------------------------------------------------------ #
#                                      STD WORD LENGTH                                             #
# ------------------------------------------------------------------------------------------------ #
class StdSentenceLength(FeatureExtractor):
    """Measures variation in sentence length over the discourse"""

    def __init__(self) -> None:
        super(StdSentenceLength, self).__init__()
        self._name = "std_sentence_length"
        self._category = "length"

    def extract(self, data: pd.DataFrame, min_length: int = 0) -> None:
        """Extracts the feature from the dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the text data to be analyzed.
        """
        return data[self._text_col].apply(lambda x: self._std_sentence_length(x))

    def _std_sentence_length(self, discourse: str) -> float:
        sentences = sent_tokenize(discourse)
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        if len(sentence_lengths) < 2:
            return 0.0
        else:
            return statistics.stdev(sentence_lengths)


# %%
