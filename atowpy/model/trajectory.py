import pickle
from pathlib import Path
from loguru import logger

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error

from atowpy.model.simple import SimpleModel
from atowpy.paths import get_models_path
from atowpy.read import read_challenge_set, read_submission_set
from atowpy.version import MODEL_FILE, ENCODER_FILE


class TrajectoryModel(SimpleModel):
    """ Advanced machine learning model which utilizes
    """

    def __init__(self, model: str = "load", apply_validation: bool = True):
        super().__init__(model, apply_validation)

        # Use different features than simple model
        pass

    @staticmethod
    def load_data_for_model_fit(folder_with_files: Path):
        # TODO add here merging
        return read_challenge_set(folder_with_files)

    @staticmethod
    def load_data_for_submission(folder_with_files: Path):
        # TODO add here merging
        return read_submission_set(folder_with_files)
