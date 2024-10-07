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

from atowpy.paths import get_models_path
from atowpy.read import read_challenge_set, read_submission_set
from atowpy.version import MODEL_FILE, ENCODER_FILE

RANDOM_STATE = 2024


class SimpleModel:
    """
    Class for training simple regression models without using
    trajectories data
    """

    model_by_name = {"rfr": RandomForestRegressor}

    def __init__(self, model: str = "load"):
        self.model_name = model
        if model == "load":
            # Path to the binary file passed - load it
            self.model, self.encoder = self.load()
            logger.info("Simple model. Core model was successfully loaded from "
                        "file")
        else:
            # Name of the model passed - need to initialize class
            self.model = self.model_by_name[model]()
            self.encoder = None
            logger.info("Simple model. Core model was successfully initialized")

        self.num_features = ["actual_offblock_hour", "arrival_hour",
                             "flight_duration", "taxiout_time", "flown_distance"]
        self.categorical_features = ["month", "day_of_week",
                                     "aircraft_type", "wtc",
                                     "airline"]
        self.target = "tow"
        self.all_columns = self.num_features + self.categorical_features + [self.target]

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def fit(self, folder_with_files: Path):

        def objective(trial):
            # Split for train and validation
            train_features, test_features, train_target, test_target = train_test_split(
                self.x_train,
                self.y_train,
                test_size=0.3,
                random_state=RANDOM_STATE,
                shuffle=True)

            params = {"n_estimators": trial.suggest_categorical("n_estimators",
                                                                [10, 50, 100]),
                      "min_samples_split": trial.suggest_int("min_samples_split", 2, 32),
                      "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 32),
                      "max_features": trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                      "criterion": trial.suggest_categorical("criterion",
                                                            ["squared_error",
                                                             "absolute_error",
                                                             "friedman_mse",
                                                             "poisson"]),
                      "max_depth": trial.suggest_int("max_depth", 3, 100, step=2)}

            model = self.model_by_name[self.model_name](**params)
            model = model.fit(train_features, train_target)
            predicted = model.predict(test_features)
            rmse_metric = root_mean_squared_error(y_true=test_target,
                                                  y_pred=predicted)
            return rmse_metric

        logger.debug("Features preprocessing for fit. Starting ...")
        features_df = read_challenge_set(folder_with_files)
        features_df = self._preprocess_features(features_df)
        logger.debug("Features preprocessing for fit. Successfully finished...")

        # Fit model on the data
        logger.debug("Model fit. Starting ...")
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        categorical_features = self.encoder.fit_transform(
            np.array(features_df[self.categorical_features])).toarray()
        numerical_features = np.array(features_df[self.num_features])

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            np.hstack([numerical_features, categorical_features]),
            np.array(features_df[self.target]),
            test_size=0.2,
            random_state=RANDOM_STATE,
            shuffle=True)

        study = optuna.create_study(direction="minimize",
                                    study_name="simple model fit")
        study.optimize(objective, n_trials=15, timeout=3600000)
        best_trial = study.best_trial

        # Re-create the model
        self.model = self.model_by_name[self.model_name](**best_trial.params)
        self.model.fit(self.x_train, self.y_train)
        self._validate(self.x_test, self.y_test)
        logger.debug("Model fit. Successfully finished...")
        return self.model

    def _validate(self, x_test: np.array, y_test: np.array):
        logger.info("--- MODEL VALIDATION ---")
        logger.debug(f"Validation sample size: {len(x_test)}")
        predicted = self.model.predict(x_test)

        mae_metric = mean_absolute_error(y_true=y_test, y_pred=predicted)
        logger.info(f'MAE metric: {mae_metric:.2f}')

        mape_metric = mean_absolute_percentage_error(y_true=y_test,
                                                     y_pred=predicted) * 100
        logger.info(f'MAPE metric: {mape_metric:.2f}')

        rmse_metric = root_mean_squared_error(y_true=y_test, y_pred=predicted)
        logger.info(f'RMSE metric: {rmse_metric:.2f}')
        logger.info("--- MODEL VALIDATION ---")

    def predict(self, folder_with_files: Path):
        logger.debug("Features preprocessing for predict. Starting ...")
        features_df = read_submission_set(folder_with_files)
        features_df = self._preprocess_features(features_df)
        logger.debug("Features preprocessing for predict. Successfully finished...")

        logger.debug("Model predict. Starting ...")
        categorical_features = self.encoder.transform(np.array(features_df[self.categorical_features])).toarray()
        numerical_features = np.array(features_df[self.num_features])
        all_features = np.hstack([numerical_features, categorical_features])

        predicted = self.model.predict(all_features)
        features_df[self.target] = predicted
        logger.debug("Model predict. Prediction generation was successfully finished")
        return features_df

    def save(self):
        """ Save fitted model """
        logger.debug("Start saving model ... ")
        # In addition to model, scaler and encoder will be saved
        with open(Path(get_models_path(), MODEL_FILE), "wb") as f:
            pickle.dump(self.model, f)

        with open(Path(get_models_path(), ENCODER_FILE), "wb") as f:
            pickle.dump(self.encoder, f)
        logger.debug("Model was successfully saved.")

    @staticmethod
    def load():
        """ Load the model from file """
        with open(Path(get_models_path(), MODEL_FILE), "rb") as f:
            model = pickle.load(f)

        with open(Path(get_models_path(), ENCODER_FILE), "rb") as f:
            encoder = pickle.load(f)

        return model, encoder

    @staticmethod
    def _preprocess_features(features_df: pd.DataFrame):
        features_df["month"] = features_df["date"].dt.month
        features_df["day_of_week"] = features_df["date"].dt.dayofweek
        features_df["actual_offblock_hour"] = features_df["actual_offblock_time"].dt.hour
        features_df["arrival_hour"] = features_df["arrival_time"].dt.hour

        features_df = features_df.drop(columns=["date"])
        return features_df
