#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Automated Essay Scoring: A Data-First Deep Learning Approach                        #
# Version    : 0.1.0                                                                               #
# Python     : 3.10.4                                                                              #
# Filename   : /pipeline.py                                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/AutomatedEssayScoring                              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday August 11th 2022 09:43:52 pm                                               #
# Modified   : Friday August 12th 2022 08:43:33 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : BSD 3-clause "New" or "Revised" License                                             #
# Copyright  : (c) 2022 John James                                                                 #
# ================================================================================================ #
"""Pipeline Module"""
import importlib
from datetime import datetime
import pandas as pd
import mlflow
import logging
import logging.config

from aes.utils.config import Config
from aes.utils.config import LogConfig
from aes.workflow.operators import Operator

# ------------------------------------------------------------------------------------------------ #
logging.config.dictConfig(LogConfig().config)
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


class Pipeline:
    """Collection of operators and methods to execute and track Pipeline runs.

    Args:
        name (str): Human readable name for the pipeline run.
        context (dict): Data required by all operators in the pipeline. Optional.
    """

    def __init__(self, name: str, version: str = "0.1.0", context: dict = {}) -> None:
        self._name = name
        self._version = version
        self._context = context
        self._steps = []
        self._active_run = None
        self._run_id = None

        self._created = datetime.now()
        self._started = None
        self._stopped = None
        self._duration = None

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def created(self) -> datetime:
        return self._created

    @property
    def started(self) -> datetime:
        return self._started

    @property
    def stopped(self) -> datetime:
        return self._stopped

    @property
    def duration(self) -> datetime:
        return self._duration

    def add_step(self, step: Operator) -> None:
        """Adds a operator step to the pipeline.

        Args:
            step (Operator): Operator object
        """
        self._steps.append(step)

    def add_steps(self, steps: []) -> None:
        """Adds a list of steps to the Pipeline.

        Args:
            steps (list): List of pipeline steps
        """
        self._steps.extend(steps)

    def remove_step(self, name: str) -> None:
        """Removes a step, referenced by name, from the pipeline

        Args:
            name (str): Name assigned to the operator object.
        """
        self._steps = [step for step in self._steps if step.name != name]

    def get_step(self, name: str) -> None:
        """Returns a Operator object by name."""
        return [step for step in self._steps if step.name == name][0]

    def print_steps(self) -> None:
        """Prints the step names in order in which they are added."""
        seq = range(1, len(self._steps) + 1)
        steps = {"Seq": seq, "Step": [step.name for step in self._steps]}
        df = pd.DataFrame(steps)
        print(df)

    def run(self, start_step: int = 0, stop_step: float = float("inf")) -> None:
        """Runs the pipeline

        Args:
            start_step (int): First step to execute in the run sequence.
            stop_step (int): Last step to execute in the run sequence.
        """
        self._setup()
        self._execute(start_step=start_step, stop_step=stop_step, context=self._context)
        self._teardown()

    def _execute(
        self, start_step: int = 0, stop_step: float = float("inf"), context: dict = {}
    ) -> None:
        """Iterates through the sequence of steps.

        Args:
            start_step (int): First step to execute in the run sequence.
            stop_step (int): Last step to execute in the run sequence.
            context (dict): Dictionary of parameters shared across steps.
        """

        data = None
        for seq, task in enumerate(self._steps, 1):
            if task.seq >= start_step and task.seq <= stop_step:
                result = task.run(data=data, context=context)
                data = result if result is not None else data

    def _setup(self) -> None:
        """Executes setup for pipeline."""
        mlflow.start_run()
        self._active_run = mlflow.active_run()
        self._run_id = self._active_run.info.run_id
        self._started = datetime.now()

    def _teardown(self) -> None:
        """Completes the pipeline process."""
        mlflow.end_run()
        self._stopped = datetime.now()
        self._duration = round((self._stopped - self._started).total_seconds(), 4)


# ------------------------------------------------------------------------------------------------ #


class PipelineBuilder:
    """Constructs Configuration file based Pipeline objects"""

    def __init__(self, config_filepath: str) -> None:
        self._config_filepath = config_filepath
        self.reset()

    def reset(self) -> None:
        self._pipeline = None

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    def build(self, config: dict) -> None:
        """Constructs a Pipeline object.

        Args:
            config (dict): Pipeline configuration
        """
        config = self._get_config()
        pipeline = self._build_pipeline(config)
        steps = self._build_steps(config.get("steps", None))
        pipeline.add_steps(steps)
        self._pipeline = pipeline

    def _get_config(self) -> dict:
        io = Config(self._config_filepath)
        return io.read()

    def _build_pipeline(self, config: dict) -> Pipeline:
        return Pipeline(name=config.get("pipeline_name", None), version=config.get("version", None))

    def _build_steps(self, config: dict) -> list:
        """Iterates through task and returns a list of task objects."""

        steps = []

        for _, step_config in config.items():

            try:

                # Create task object from string using importlib
                module = importlib.import_module(name=step_config["module"])
                step = getattr(module, step_config["operator"])

                operator = step(
                    name=step_config["name"],
                    params=step_config["params"],
                )

                steps.append(operator)

            except KeyError as e:
                logger.error("Configuration File is missing operator configuration data")
                raise (e)

        return steps
