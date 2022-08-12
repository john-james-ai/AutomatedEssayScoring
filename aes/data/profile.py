#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Deep Automated Neural Network Essay Scoring and Evaluation (DANNESE)                #
# Version    : 0.1.0                                                                               #
# Filename   : /profile.py                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday August 3rd 2022 05:21:41 am                                               #
# Modified   : Wednesday August 10th 2022 06:36:24 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
import re
import pandas as pd
import numpy as np
import logging
import logging.config
import matplotlib.pyplot as plt
import seaborn as sns
from copy import copy
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# ------------------------------------------------------------------------------------------------ #
from fpe.utils.log_config import LOG_CONFIG
from fpe.data import specials, punctuation, controls

# ------------------------------------------------------------------------------------------------ #
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


class TextProfiler:
    """Provides an overview of the basic properties and characteristics of tabular and text data.

    Args:
        data (pd.DataFrame): The data to be profiled in a pandas DataFrame
        text_col (str): The column containing the text data.
    """

    def __init__(self, data: pd.DataFrame, text_col: str) -> None:
        self._data = data
        self._text_col = text_col

    @property
    def rows(self) -> int:
        """Returns the number of rows in the dataset."""
        return self._data.shape[0]

    @property
    def columns(self) -> int:
        """Returns the number of columns in the dataset."""
        return self._data.shape[1]

    @property
    def memory(self) -> pd.DataFrame:
        df = self._data.memory_usage(deep=True, index=False).to_frame().reset_index()
        df = df.rename(columns={"index": "Column", 0: "Memory (Bytes)"})
        d = {"Column": "Total", "Memory (Bytes)": df["Memory (Bytes)"].sum()}
        total = pd.DataFrame(data=d, index=[df.shape[0] + 1])
        df = pd.concat([df, total], axis=0)
        return df.style.format(thousands=",")

    @property
    def cardinality(self) -> pd.DataFrame:
        """Returns the cardinality of each column in a Dataframe."""
        return pd.DataFrame(data=self._data.nunique(), columns=["Cardinality"])

    @property
    def duplicate_text(self) -> pd.DataFrame:
        """Returns the count, and fraction of duplication in each column and duplicated data."""

        duplicates = self._data[self._data.duplicated(subset="discourse_text", keep=False)]

        stats = {
            "Column": "discourse_text",
            "Num Duplicates": len(duplicates),
            "Pct Duplicates": len(duplicates) / self._data.shape[0] * 100,
        }

        d = {}
        d["stats"] = pd.DataFrame(data=stats, index=[0]).set_index(["Column"])
        d["data"] = duplicates["discourse_text"]
        return d

    def value_counts(self, col: str) -> pd.DataFrame:
        """Returns the value counts and normalized proportions."""
        df1 = self._data[col].value_counts().to_frame()
        df2 = round(self._data[col].value_counts(normalize=True).to_frame(), 2)
        df1 = df1.merge(df2, left_index=True, right_index=True, suffixes=("_count", "_ratio"))
        return df1


# ------------------------------------------------------------------------------------------------ #


class FeatureExtractor:
    """Performs character, word, and sentence feature extraction via delegation to FeatureExtractor Classes"""

    def __init__(self, data: pd.DataFrame, text_col: str = "discourse_text") -> None:
        self._data = data
        self._columns = data.columns.tolist()
        self._text_col = text_col
        self._feature_extractor = None
        self._feature = None

    @property
    def feature(self) -> str:
        return self._feature

    @feature.setter
    def feature(self, feature: str) -> None:
        self._feature = feature
        if feature.lower().startswith("alpha"):
            self._feature_extractor = AlphabeticFeatures(
                data=copy(self._data), text_col=self._text_col
            )
        elif feature.lower().startswith("numeric"):
            self._feature_extractor = NumericFeatures(
                data=copy(self._data), text_col=self._text_col
            )
        elif feature.lower().startswith("special"):
            self._feature_extractor = SpecialCharacterFeatures(
                data=copy(self._data), text_col=self._text_col
            )
        elif feature.lower().startswith("control"):
            self._feature_extractor = ControlCharacterFeatures(
                data=copy(self._data), text_col=self._text_col
            )
        elif feature.lower().startswith("punctuation"):
            self._feature_extractor = PunctuationFeatures(
                data=copy(self._data), text_col=self._text_col
            )
        elif feature.lower().startswith("word") and "length" not in feature.lower():
            self._feature_extractor = WordFeatures(data=copy(self._data), text_col=self._text_col)
        elif feature.lower().startswith("word") and "length" in feature.lower():
            self._feature_extractor = WordLengthFeatures(
                data=copy(self._data), text_col=self._text_col
            )
        elif feature.lower().startswith("sentence") and "length" not in feature.lower():
            self._feature_extractor = SentenceFeatures(
                data=copy(self._data), text_col=self._text_col
            )
        elif feature.lower().startswith("sentence") and "length" in feature.lower():
            self._feature_extractor = SentenceLengthFeatures(
                data=copy(self._data), text_col=self._text_col
            )

        elif feature.lower().startswith("upper"):
            self._feature_extractor = UpperCaseFeatures(
                data=copy(self._data), text_col=self._text_col
            )

        elif feature.lower().startswith("stop"):
            self._feature_extractor = StopWordFeatures(
                data=copy(self._data), text_col=self._text_col
            )

        else:
            raise ValueError("Invalid feature")

    @property
    def total(self) -> int:
        return self._feature_extractor.total

    @property
    def features(self) -> pd.DataFrame:
        return self._feature_extractor.features

    @property
    def statistics(self) -> int:
        return self._feature_extractor.statistics

    @property
    def random_discourse(self) -> pd.DataFrame:
        columns = self._columns
        columns.append(self._feature)
        return self._feature_extractor.random_discourse

    def get_extractor(self) -> int:
        return self._feature_extractor

    def plot(
        self, fig: plt.figure, title: str = None, xlab: str = None, ylab: str = None
    ) -> plt.figure:
        return self._feature_extractor.plot(fig=fig, title=title, xlab=xlab, ylab=ylab)


# ------------------------------------------------------------------------------------------------ #
class FeatureExtractorBase(ABC):
    """Base class defining API for text feature extractors.

    Args:
        data (np.array): Numpy array containing the text.
        text_col (str): The column name to be used for the text in the feature extraction output

    """

    def __init__(self, data: pd.DataFrame, text_col: str = "discourse_text") -> None:
        self._data = data  # The training data
        self._text_col = text_col  # Column name containing the text
        self._total = None  # The total count for the feature in the text
        self._statistics = None  # The descriptive statistics for the feature by row of text
        self._name = None  # The name of the feature extracted by the subclass.
        self._extracted = False

    @property
    def total(self) -> int:
        """Returns the total count for the feature in the text."""
        self._check_extracted()
        return self._data[self._name].values.sum()

    @property
    def features(self) -> pd.DataFrame:
        """Returns a dataframe with the text and the associated feature counts by observation."""
        self._check_extracted()
        return self._data.sort_values(by=self._name, ascending=False, axis=0)

    @property
    def statistics(self) -> pd.DataFrame:
        self._check_extracted()
        return self._data[self._name].describe().to_frame().T

    @property
    def random_discourse(self) -> pd.DataFrame:
        idx = self._data[self._data[self._name] > 0].index.values.tolist()
        return self._data.loc[np.random.choice(idx)].to_frame()

    @abstractmethod
    def _extract(self) -> None:
        """Computes the feature counts by text observation."""
        pass

    def plot(
        self, fig: plt.figure, title: str = None, xlab: str = None, ylab: str = None
    ) -> plt.figure:
        """Histogram displaying the distribution of the feature counts."""
        self._check_extracted()
        ax = fig.add_subplot()
        ax = sns.histplot(data=self._data, x=self._name)
        ax.set_xlabel(xlabel=xlab)
        ax.set_ylabel(ylabel=ylab)
        ax.set_title(label=title, size=16)
        return fig

    def _check_extracted(self) -> None:
        if not self._extracted:
            self._extract()
            self._extracted = True


# ------------------------------------------------------------------------------------------------ #


class AlphabeticFeatures(FeatureExtractorBase):
    def __init__(self, data: pd.DataFrame, text_col: str = "discourse_text") -> None:
        super(AlphabeticFeatures, self).__init__(data=data, text_col=text_col)
        self._name = "alpha"

    def _extract(self) -> None:
        self._data[self._name] = self._data[self._text_col].apply(
            lambda x: len([i for i in x if i.isalpha()])
        )


# ------------------------------------------------------------------------------------------------ #


class NumericFeatures(FeatureExtractorBase):
    def __init__(self, data: pd.DataFrame, text_col: str = "discourse_text") -> None:
        super(NumericFeatures, self).__init__(data=data, text_col=text_col)
        self._name = "numeric"

    def _extract(self) -> None:
        self._data[self._name] = self._data[self._text_col].apply(
            lambda x: len([x for x in x.split() if x.isdigit()])
        )


# ------------------------------------------------------------------------------------------------ #


class SpecialCharacterFeatures(FeatureExtractorBase):
    def __init__(self, data: pd.DataFrame, text_col: str = "discourse_text") -> None:
        super(SpecialCharacterFeatures, self).__init__(data=data, text_col=text_col)
        self._name = "special_characters"
        self._counts_dict = {}
        self._counts = None

    @property
    def counts(self) -> pd.DataFrame:
        return self._counts

    def _extract(self) -> None:
        self._initialize_counts()
        self._data[self._name] = self._data[self._text_col].apply(lambda x: self._count_specials(x))
        self._counts = pd.DataFrame.from_dict(
            data=self._counts_dict, orient="index", columns=["Count"]
        ).sort_values(by="Count", ascending=False)

    def _initialize_counts(self) -> None:
        for k, v in specials.items():
            self._counts_dict[k] = 0

    def _count_specials(self, discourse: str) -> int:
        total = 0
        for k, v in specials.items():
            count = len(re.findall(v, discourse))
            self._counts_dict[k] += count
            total += count
        return total


# ------------------------------------------------------------------------------------------------ #


class PunctuationFeatures(FeatureExtractorBase):
    def __init__(self, data: pd.DataFrame, text_col: str = "discourse_text") -> None:
        super(PunctuationFeatures, self).__init__(data=data, text_col=text_col)
        self._name = "punctuation"
        self._counts_dict = {}
        self._counts = None

    @property
    def counts(self) -> pd.DataFrame:
        return self._counts

    def _extract(self) -> None:
        self._initialize_counts()
        self._data[self._name] = self._data[self._text_col].apply(
            lambda x: self._count_punctuation(x)
        )
        self._counts = pd.DataFrame.from_dict(
            data=self._counts_dict, orient="index", columns=["Count"]
        ).sort_values(by="Count", ascending=False)

    def _initialize_counts(self) -> None:
        for k, v in punctuation.items():
            self._counts_dict[k] = 0

    def _count_punctuation(self, discourse: str) -> int:
        total = 0
        for k, v in punctuation.items():
            count = len(re.findall(v, discourse))
            self._counts_dict[k] += count
            total += count
        return total


# ------------------------------------------------------------------------------------------------ #


class ControlCharacterFeatures(FeatureExtractorBase):
    def __init__(self, data: pd.DataFrame, text_col: str = "discourse_text") -> None:
        super(ControlCharacterFeatures, self).__init__(data=data, text_col=text_col)
        self._name = "control"

    def _extract(self) -> None:
        self._data[self._name] = self._data[self._text_col].apply(lambda x: self._count_controls(x))

    def _count_controls(self, discourse: str) -> int:
        total = 0
        for k, v in controls.items():
            total += len(re.findall(v, discourse))
        return total


# ------------------------------------------------------------------------------------------------ #


class WordFeatures(FeatureExtractorBase):
    def __init__(self, data: pd.DataFrame, text_col: str = "discourse_text") -> None:
        super(WordFeatures, self).__init__(data=data, text_col=text_col)
        self._name = "words"

    def _extract(self) -> None:
        self._data[self._name] = self._data[self._text_col].apply(lambda x: len(str(x).split(" ")))


# ------------------------------------------------------------------------------------------------ #


class WordLengthFeatures(FeatureExtractorBase):
    def __init__(self, data: pd.DataFrame, text_col: str = "discourse_text") -> None:
        super(WordLengthFeatures, self).__init__(data=data, text_col=text_col)
        self._name = "word_length"

    def _extract(self) -> None:
        self._data[self._name] = self._data[self._text_col].apply(
            lambda x: self._average_word_length(x)
        )

    def _average_word_length(self, discourse: str) -> float:
        words = discourse.split()
        return sum(len(word) for word in words) / len(words)


# ------------------------------------------------------------------------------------------------ #


class SentenceFeatures(FeatureExtractorBase):
    def __init__(self, data: pd.DataFrame, text_col: str = "discourse_text") -> None:
        super(SentenceFeatures, self).__init__(data=data, text_col=text_col)
        self._name = "sentences"

    def _extract(self) -> None:
        self._data[self._name] = self._data[self._text_col].apply(lambda x: len(sent_tokenize(x)))


# ------------------------------------------------------------------------------------------------ #


class SentenceLengthFeatures(FeatureExtractorBase):
    def __init__(self, data: pd.DataFrame, text_col: str = "discourse_text") -> None:
        super(SentenceLengthFeatures, self).__init__(data=data, text_col=text_col)
        self._name = "sentence_length"

    def _extract(self) -> None:
        self._data[self._name] = self._data[self._text_col].apply(
            lambda x: self._average_sentence_length(x)
        )

    def _average_sentence_length(self, discourse: str) -> float:
        sentences = list(sent_tokenize(discourse))
        return sum(len(sentence.split()) for sentence in sentences) / len(sentences)


# ------------------------------------------------------------------------------------------------ #


class UpperCaseFeatures(FeatureExtractorBase):
    def __init__(self, data: pd.DataFrame, text_col: str = "discourse_text") -> None:
        super(UpperCaseFeatures, self).__init__(data=data, text_col=text_col)
        self._name = "upper_case"

    def _extract(self) -> None:
        self._data[self._name] = self._data[self._text_col].apply(
            lambda x: self._count_upper_case_words(x)
        )

    def _count_upper_case_words(self, discourse: str) -> float:
        words = discourse.split()
        return sum(word.isupper() for word in words) / len(words)


# ------------------------------------------------------------------------------------------------ #


class StopWordFeatures(FeatureExtractorBase):
    def __init__(self, data: pd.DataFrame, text_col: str = "discourse_text") -> None:
        super(StopWordFeatures, self).__init__(data=data, text_col=text_col)
        self._name = "stopwords"

    def _extract(self) -> None:
        stop = stopwords.words("english")
        self._data[self._name] = self._data[self._text_col].apply(
            lambda x: len([x for x in x.split() if x in stop]) / len(x.split())
        )
