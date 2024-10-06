import pickle
from pathlib import Path
from typing import Union
from loguru import logger

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from atowpy.paths import results_folder, get_models_path
from atowpy.read import read_challenge_set, read_submission_set
from atowpy.version import MODEL_FILE


class SimpleModel:
    """
    Class for training simple regression models without using
    trajectories data
    """

    model_by_name = {"rfr": RandomForestRegressor}

    def __init__(self, model: Union[str, Path] = "rfr"):
        if isinstance(model, str):
            # Name of the model passed - need to initialize class
            self.model = self.model_by_name[model]()
            logger.info("Simple model. Core model was successfully initialized")
        else:
            # Path to the binary file passed - load it
            self.model = self.load(model)
            logger.info("Simple model. Core model was successfully loaded from "
                        "file")

        self.num_features = ["month", "day_of_week", "flight_duration", "taxiout_time", "flown_distance"]
        self.categorical_features = ["month", "day_of_week", "adep", "name_adep",
                                     "ades", "name_ades", "actual_offblock_time",
                                     "arrival_time", "aircraft_type", "wtc",
                                     "airline"]
        self.target = "tow"
        self.all_columns = self.num_features + self.categorical_features + [self.target]

    def fit(self, folder_with_files: Path):
        logger.debug("Features preprocessing for fit. Starting ...")
        features_df = read_challenge_set(folder_with_files)
        features_df = self._preprocess_features(features_df)
        logger.debug("Features preprocessing for fit. Successfully finished...")

        # Fit model on the data
        logger.debug("Model fit. Starting ...")
        self.model.fit(np.array(features_df[self.num_features]),
                       np.array(features_df[self.target]))
        logger.debug("Model fit. Successfully finished...")
        return self.model

    def predict(self, folder_with_files: Path):
        logger.debug("Features preprocessing for predict. Starting ...")
        features_df = read_submission_set(folder_with_files)
        features_df = self._preprocess_features(features_df)
        logger.debug("Features preprocessing for predict. Successfully finished...")

        predicted = self.model.predict(np.array(features_df[self.num_features]))
        features_df[self.target] = predicted
        return features_df

    def save(self):
        """ Save fitted model """
        # In addition to model, scaler and encoder will be saved
        with open(Path(get_models_path(), MODEL_FILE), "wb") as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load(model_path: Path):
        """ Load the model from file """
        with open(Path(get_models_path(), MODEL_FILE), "rb") as f:
            model = pickle.load(f)

        return model

    @staticmethod
    def _preprocess_features(features_df: pd.DataFrame):
        features_df["month"] = features_df["date"].dt.month
        features_df["day_of_week"] = features_df["date"].dt.dayofweek

        features_df = features_df.drop(columns=["date"])
        return features_df
