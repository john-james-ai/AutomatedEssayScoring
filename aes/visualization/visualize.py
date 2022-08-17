#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Automated Essay Scoring: A Data-First Deep Learning Approach                        #
# Version    : 0.1.0                                                                               #
# Python     : 3.9.12                                                                              #
# Filename   : /visualize.py                                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 2nd 2022 08:08:36 pm                                                 #
# Modified   : Sunday August 14th 2022 01:55:35 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
from aes.utils.pprint import print_str
from aes.utils.config import VisualConfig

# ------------------------------------------------------------------------------------------------ #
sns.set_style("whitegrid")
sns.set_palette("Blues_d")


class Visualizer(ABC):
    def __init__(self, config: dict = None) -> None:
        self._config = VisualConfig()
        self._project = "Automated Essay Scoring"

    @abstractmethod
    def plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        group_by: str = None,
        title: str = None,
        xlab: str = None,
        ylab: str = None,
    ) -> plt.figure:
        pass

    def _format_axis(self, ax: plt.axes, title: str, xlab: str, ylab: str) -> plt.axes:
        """Formats labels and titles on axis object."""
        ax.set_xlabel(xlabel=print_str(xlab))
        ax.set_ylabel(xlabel=print_str(ylab))
        ax.set_title(label=print_str(title))
        return ax


# ------------------------------------------------------------------------------------------------ #


class Histogram(Visualizer):
    def __init__(self, config: dict = None) -> None:
        super(Histogram, self).__init__(config=config)

    def plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        group_by: str = None,
        title: str = None,
        xlab: str = None,
        ylab: str = None,
    ) -> plt.figure:

        if not group_by:
            return self._plot(data, x, y, title, xlab, ylab)
        elif group_by.startswith("discourse_effectiveness"):
            return self._plot_ratings(data, x, y, group_by, title, xlab, ylab)
        else:
            self._plot_discourse(data, x, y, group_by, title, xlab, ylab)

    def _plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        title: str = None,
        xlab: str = None,
        ylab: str = None,
    ) -> Union(plt.figure, plt.axes):
        """Plots all the data."""
        fig, ax = plt.subplots(1, 1, figsize=self._config["visual"]["figsize"]["medium"])
        fig = sns.distplot(data, x=x)
        ax = self._format_axis(ax=ax, title=title, xlab=xlab, ylab=ylab)
        return fig, ax

    def _plot_ratings(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        group_by: str = None,
        title: str = None,
        xlab: str = None,
        ylab: str = None,
    ) -> Union[plt.figure, plt.axes]:
        """Plot ratings level groupings."""

        fig, ax = plt.subplots(1, 1, figsize=self._config["visual"]["figsize"]["large"])
        fig = sns.distplot(data, x=x, hue=group_by)
        ax = self._format_axis(ax=ax, title=title, xlab=xlab, ylab=ylab)
        return fig, ax

    def _plot_discourse(
        self,
        data: pd.DataFrame,
        fig: plt.figure,
        x: str,
        y: str = None,
        group_by: str = None,
        title: str = None,
        xlab: str = None,
        ylab: str = None,
    ) -> Union[plt.figure, plt.axes]:
        """Plot discourse level groupings."""

        fig, ax = plt.subplots(2, 4, figsize=self._config["visual"]["figsize"]["vlarge"])
        fig = sns.distplot(data=data, x=x, col=group_by)
        ax = self._format_axis(ax=ax, title=title, xlab=xlab, ylab=ylab)
        return fig, ax


# ------------------------------------------------------------------------------------------------ #


class Boxplot(Visualizer):
    # TODO: This code has not been reviewed.
    def __init__(self, config: dict = None) -> None:
        super(Histogram, self).__init__(config=config)

    def plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        group_by: str = None,
        title: str = None,
        xlab: str = None,
        ylab: str = None,
    ) -> plt.figure:

        if not group_by:
            return self._plot(data, x, y, title, xlab, ylab)
        elif group_by.startswith("discourse_effectiveness"):
            return self._plot_ratings(data, x, y, group_by, title, xlab, ylab)
        else:
            self._plot_discourse(data, x, y, group_by, title, xlab, ylab)

    def _plot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        title: str = None,
        xlab: str = None,
        ylab: str = None,
    ) -> Union(plt.figure, plt.axes):
        """Plots all the data."""
        fig, ax = plt.subplots(1, 1, figsize=self._config["visual"]["figsize"]["medium"])
        fig = sns.distplot(data, x=x)
        ax = self._format_axis(ax=ax, title=title, xlab=xlab, ylab=ylab)
        return fig, ax

    def _plot_ratings(
        self,
        data: pd.DataFrame,
        x: str,
        y: str = None,
        group_by: str = None,
        title: str = None,
        xlab: str = None,
        ylab: str = None,
    ) -> Union[plt.figure, plt.axes]:
        """Plot ratings level groupings."""

        fig, ax = plt.subplots(1, 1, figsize=self._config["visual"]["figsize"]["large"])
        fig = sns.distplot(data, x=x, hue=group_by)
        ax = self._format_axis(ax=ax, title=title, xlab=xlab, ylab=ylab)
        return fig, ax

    def _plot_discourse(
        self,
        data: pd.DataFrame,
        fig: plt.figure,
        x: str,
        y: str = None,
        group_by: str = None,
        title: str = None,
        xlab: str = None,
        ylab: str = None,
    ) -> Union[plt.figure, plt.axes]:
        """Plot discourse level groupings."""

        fig, ax = plt.subplots(2, 4, figsize=self._config["visual"]["figsize"]["vlarge"])
        fig = sns.distplot(data=data, x=x, col=group_by)
        ax = self._format_axis(ax=ax, title=title, xlab=xlab, ylab=ylab)
        return fig, ax
