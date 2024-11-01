import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

from loguru import logger

import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from atowpy.dask_wrappers import close_dask_client, convert_into_dask_dataframe, DaskClientWrapper
from atowpy.model.xgboost_wrappers import DaskXGBRegressor
from atowpy.paths import get_models_path
from atowpy.read import read_challenge_set, read_submission_set
from atowpy.version import MODEL_FILE, ENCODER_FILE, SCALER_FILE, VERSION, PCA_FILE

RANDOM_STATE = 2024


class SimpleModel:
    """
    Class for training simple regression models without using
    trajectories data
    """

    model_by_name = {"rfr": RandomForestRegressor,
                     "ridge": Ridge,
                     "knn": KNeighborsRegressor,
                     "xgb": XGBRegressor,
                     "xgb_dask": DaskXGBRegressor}

    def __init__(self, model: str = "load", apply_validation: bool = True):
        self.model_name = model
        if model == "load":
            # Path to the binary file passed - load it
            self.model, self.encoder, self.scaler, self.pca = self.load()
            logger.info("Simple model. Core model was successfully loaded from "
                        "file")
        else:
            # Name of the model passed - need to initialize class
            self.model = self.model_by_name[model]()
            self.encoder = None
            self.scaler = None
            self.pca = None
            logger.info("Simple model. Core model was successfully initialized")

        self.num_features = ["actual_offblock_hour", "arrival_hour",
                             "flight_duration", "taxiout_time",
                             "flown_distance"]
        self.categorical_features = ["month", "day_of_week",
                                     "aircraft_type", "wtc",
                                     "airline"]
        self.target = "tow"
        self.features_columns = self.num_features + self.categorical_features
        self.all_columns = self.num_features + self.categorical_features + [self.target]
        self.apply_validation = apply_validation

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.dask_handler = None

        self.was_prediction_data_full: bool = True
        self.path_to_backup_submission = None
        self.optimization_results = []

    def fit(self, folder_with_files: Path):
        if "dask" in self.model_name:
            self._fit_with_dask(folder_with_files)
        else:
            self._fit_with_numpy(folder_with_files)

    def _fit_with_dask(self, folder_with_files: Path):

        def objective(trial):
            if self.model_name == "xgb_dask":
                # See https://xgboost.readthedocs.io/en/stable/parameter.html for details
                params = {"n_estimators": trial.suggest_int('n_estimators', 800,
                                                            1200),
                          "booster": trial.suggest_categorical("booster",
                                                               ["gbtree",
                                                                "dart"]),
                          "eta": trial.suggest_float("eta", 0.01, 0.99),
                          "max_depth": trial.suggest_int("max_depth", 3, 10,
                                                         step=2),
                          "gamma": trial.suggest_float("gamma", 0.0, 10.00)}
            else:
                raise ValueError(f"Model {self.model_name} is not supported")

            logger.debug(f"Objective function. Current params: {params}")
            # train, test = self.x_train.random_split([0.6, 0.4], random_state=RANDOM_STATE)

            model = self.model_by_name[self.model_name](**params)
            output = model.fit(self.x_train[self.features_columns].values, self.x_train[self.target].values,
                               self.dask_handler)
            # Predict and calculate
            rmse_on_validation_set = output['history']['valid']['rmse'][-1]
            predicted = model.predict(self.x_test[self.features_columns].values, self.dask_handler)
            rmse_metric = root_mean_squared_error(y_true=self.x_test[self.target].values.compute(),
                                                  y_pred=predicted.compute())
            # Combination on metric on validation set and difference between test and validation
            logger.debug(f"Objective function. Validation RMSE: {rmse_metric:.2f}. "
                         f"Training: {rmse_on_validation_set:.2f}")

            self.optimization_results.append({"n_estimators": params["n_estimators"],
                                              "booster": params["booster"],
                                              "eta": params["eta"],
                                              "max_depth": params["max_depth"],
                                              "gamma": params["gamma"],
                                              "metric": rmse_metric,
                                              "validation_metric": rmse_on_validation_set})
            return rmse_metric

        features_df = self.load_data_for_model_fit(folder_with_files)
        features_df = self._preprocess_features(features_df)
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        categorical_features = self.encoder.fit_transform(np.array(features_df[self.categorical_features])).toarray()
        # self.scaler = StandardScaler()
        numerical_features = np.array(features_df[self.num_features])

        # Save result into pandas dataframe
        all_features = np.hstack([numerical_features, categorical_features])
        logger.debug(f"Features numbers: {all_features.shape[-1]}")

        self.features_columns = [f"{i}" for i in range(all_features.shape[-1])]
        # cat_columns = [f"{i}" for i in range(numerical_features.shape[-1], all_features.shape[-1])]
        df_for_fit = pd.DataFrame(all_features, columns=self.features_columns)
        df_for_fit["tow"] = np.array(features_df["tow"], dtype=float)

        with self.close_dask_client_after_execution():
            df_for_fit = convert_into_dask_dataframe(df_for_fit)

            # for feature in cat_columns:
            #     # Mark columns as categories
            #     df_for_fit[feature] = df_for_fit[feature].astype('category')
            #     # df_for_fit[feature] = df_for_fit[feature].cat.as_known()

            if self.apply_validation:
                # There will be validation
                train_ratio = 0.8
                test_ratio = round(1 - train_ratio, 1)
                self.x_train, self.x_test = df_for_fit.random_split([train_ratio, test_ratio],
                                                                    random_state=RANDOM_STATE)
            else:
                # No validation needed
                self.x_train = df_for_fit

            try:
                study = optuna.create_study(direction="minimize",
                                            study_name="dask model fit")
                study.optimize(objective, n_trials=30, timeout=None)
                best_trial = study.best_trial
            finally:
                optimization_results = pd.DataFrame(self.optimization_results)
                optimization_results.to_csv(Path(get_models_path(),
                                                 f"optimization_track_for_model_{VERSION}.csv"), index=False)

            self.model = self.model_by_name[self.model_name](**best_trial.params)
            self.model.fit(self.x_train[self.features_columns].values, self.x_train[self.target].values,
                           self.dask_handler)

            if self.apply_validation:
                # Validate the model
                predicted = self.model.predict(self.x_test[self.features_columns].values, self.dask_handler)
                predicted = predicted.compute()
                y_test = self.x_test[self.target].values.compute()

                logger.info("--- DASK MODEL VALIDATION ---")
                logger.debug(f"Validation sample size: {len(y_test)}")

                mae_metric = mean_absolute_error(y_true=y_test, y_pred=predicted)
                logger.info(f'MAE metric: {mae_metric:.2f}')

                mape_metric = mean_absolute_percentage_error(y_true=y_test,
                                                             y_pred=predicted) * 100
                logger.info(f'MAPE metric: {mape_metric:.2f}')

                rmse_metric = root_mean_squared_error(y_true=y_test, y_pred=predicted)
                logger.info(f'RMSE metric: {rmse_metric:.2f}')
                logger.info("--- DASK MODEL VALIDATION ---")

            # Fit one more time on all data TODO - it freezes somehow
            # self.model = self.model_by_name[self.model_name](**best_trial.params)
            # self.model.fit(df_for_fit[self.features_columns].values,
            #                df_for_fit[self.target].values,
            #                self.dask_handler)
        return self.model

    def _fit_with_numpy(self, folder_with_files: Path):

        def objective(trial):
            # Split for train and validation
            train_features, test_features, train_target, test_target = train_test_split(
                self.x_train,
                self.y_train,
                test_size=0.3,
                random_state=RANDOM_STATE,
                shuffle=True)

            if self.model_name == "rfr":
                params = {"n_estimators": trial.suggest_categorical("n_estimators",
                                                                    [10, 25, 50, 100]),
                          "min_samples_split": trial.suggest_int("min_samples_split", 2, 32),
                          "min_samples_leaf": trial.suggest_int('min_samples_leaf', 1, 32),
                          "max_features": trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                          "criterion": trial.suggest_categorical("criterion",
                                                                ["squared_error",
                                                                 "friedman_mse",
                                                                 "poisson"]),
                          "max_depth": trial.suggest_int("max_depth", 3, 200, step=2)}
            elif self.model_name == "xgb":
                params = {"n_estimators": trial.suggest_categorical("n_estimators",
                                                                    [10, 25, 50, 100]),
                          "max_depth": trial.suggest_int('max_depth', 10, 1000)}
            elif self.model_name == "ridge":
                params = {"alpha": trial.suggest_float("alpha", 0, 100.0),
                          "tol": trial.suggest_float("tol", 0.0001, 0.005)}
            else:
                params = {"n_neighbors": trial.suggest_int("n_neighbors", 2, 50),
                          "leaf_size": trial.suggest_int("leaf_size", 2, 50),
                          "weights": trial.suggest_categorical('weights', ['uniform', 'distance'])}

            logger.debug(f"Current params: {params}")
            model = self.model_by_name[self.model_name](**params)
            model = model.fit(train_features, train_target)
            predicted = model.predict(test_features)
            rmse_metric = root_mean_squared_error(y_true=test_target,
                                                  y_pred=predicted)
            return rmse_metric

        logger.debug("Features preprocessing for fit. Starting ...")
        features_df = self.load_data_for_model_fit(folder_with_files)
        features_df = self._preprocess_features(features_df)
        logger.debug("Features preprocessing for fit. Successfully finished...")

        # Fit model on the data
        logger.debug("Model fit. Starting ...")
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        categorical_features = self.encoder.fit_transform(
            np.array(features_df[self.categorical_features])).toarray()
        self.scaler = StandardScaler()
        numerical_features = np.array(features_df[self.num_features])

        if self.apply_validation:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                np.hstack([numerical_features, categorical_features]),
                np.array(features_df[self.target]),
                test_size=0.2,
                random_state=RANDOM_STATE,
                shuffle=True)
        else:
            # There will be no validation provided
            self.x_train = np.hstack([numerical_features, categorical_features])
            self.y_train = np.array(features_df[self.target])

        study = optuna.create_study(direction="minimize",
                                    study_name="model fit")
        study.optimize(objective, n_trials=30, timeout=None)
        best_trial = study.best_trial

        # Re-create the model
        self.model = self.model_by_name[self.model_name](**best_trial.params)
        self.model.fit(self.x_train, self.y_train)
        self._validate(self.x_test, self.y_test)
        logger.debug("Model fit. Successfully finished...")
        return self.model

    def _validate(self, x_test: np.array, y_test: np.array):
        if self.apply_validation is False:
            logger.debug(f"Validation is not provided for current model")
            return None

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

    def fit_one_model(self, folder_with_files: Path, params: Dict):
        with self.close_dask_client_after_execution():

            features_df = self.load_data_for_model_fit(folder_with_files)
            features_df = self._preprocess_features(features_df)
            self.encoder = OneHotEncoder(handle_unknown='ignore')
            categorical_features = self.encoder.fit_transform(
                np.array(features_df[self.categorical_features])).toarray()
            # self.scaler = StandardScaler()
            numerical_features = np.array(features_df[self.num_features])

            # Save result into pandas dataframe
            all_features = np.hstack([numerical_features, categorical_features])
            self.features_columns = [f"{i}" for i in range(all_features.shape[-1])]
            cat_columns = [f"{i}" for i in range(numerical_features.shape[-1], all_features.shape[-1])]
            df_for_fit = pd.DataFrame(all_features, columns=self.features_columns)
            df_for_fit["tow"] = np.array(features_df["tow"], dtype=float)

            df_for_fit = convert_into_dask_dataframe(df_for_fit)

            for feature in cat_columns:
                # Mark columns as categories
                df_for_fit[feature] = df_for_fit[feature].astype('category')

            if self.apply_validation:
                # There will be validation
                train_ratio = 0.95
                test_ratio = round(1 - train_ratio, 2)
                self.x_train, self.x_test = df_for_fit.random_split([train_ratio, test_ratio],
                                                                    random_state=RANDOM_STATE)
            else:
                # No validation needed
                self.x_train = df_for_fit

            self.model = self.model_by_name[self.model_name](**params)
            self.model.fit(self.x_train[self.features_columns].values, self.x_train[self.target].values,
                           self.dask_handler)

            if self.apply_validation:
                # Validate the model
                predicted = self.model.predict(self.x_test[self.features_columns].values, self.dask_handler)
                predicted = predicted.compute()
                y_test = self.x_test[self.target].values.compute()

                logger.info("--- DASK MODEL VALIDATION ---")
                logger.debug(f"Validation sample size: {len(y_test)}")

                mae_metric = mean_absolute_error(y_true=y_test, y_pred=predicted)
                logger.info(f'MAE metric: {mae_metric:.2f}')

                mape_metric = mean_absolute_percentage_error(y_true=y_test,
                                                             y_pred=predicted) * 100
                logger.info(f'MAPE metric: {mape_metric:.2f}')

                rmse_metric = root_mean_squared_error(y_true=y_test, y_pred=predicted)
                logger.info(f'RMSE metric: {rmse_metric:.2f}')
                logger.info("--- DASK MODEL VALIDATION ---")

            # Fit one more time on all data TODO - it freezes somehow
            # self.model = self.model_by_name[self.model_name](**best_trial.params)
            # self.model.fit(df_for_fit[self.features_columns].values,
            #                df_for_fit[self.target].values,
            #                self.dask_handler)
        return self.model

    def predict(self, folder_with_files: Path):
        logger.debug("Features preprocessing for predict. Starting ...")
        features_df = self.load_data_for_submission(folder_with_files)
        features_df = self._preprocess_features(features_df)
        logger.debug("Features preprocessing for predict. Successfully finished...")

        logger.debug("Model predict. Starting ...")
        categorical_features = self.encoder.transform(np.array(features_df[self.categorical_features])).toarray()
        if self.scaler is not None:
            numerical_features = self.scaler.transform(np.array(features_df[self.num_features]))
        else:
            numerical_features = np.array(features_df[self.num_features])
        all_features = np.hstack([numerical_features, categorical_features])

        if self.pca is not None:
            all_features = self.pca.fit_transform(all_features)

        if isinstance(self.model, DaskXGBRegressor):
            # Dask related model
            self.features_columns = [f"{i}" for i in range(all_features.shape[-1])]
            # cat_columns = [f"{i}" for i in range(numerical_features.shape[-1], all_features.shape[-1])]
            df_for_predict = pd.DataFrame(all_features, columns=self.features_columns)

            with self.close_dask_client_after_execution():
                df_for_predict = convert_into_dask_dataframe(df_for_predict)

                # for feature in cat_columns:
                #     # Mark columns as categories
                #     df_for_predict[feature] = df_for_predict[feature].astype('category')
                #     # df_for_predict[feature] = df_for_predict[feature].cat.as_known()
                predicted = self.model.predict(df_for_predict[self.features_columns].values, self.dask_handler)
                predicted = predicted.compute()
        else:
            # Simple numpy-based model
            predicted = self.model.predict(all_features)

        features_df[self.target] = predicted
        if self.was_prediction_data_full is False:
            # Need to replace missing values with predicted values using old model which is not using features
            # extracted from trajectories
            backup_predictions = pd.read_csv(self.path_to_backup_submission)
            backup_predictions = backup_predictions.rename(columns={"tow": "tow_old"})
            merged_df = backup_predictions.merge(features_df[["flight_id", "tow"]], how="outer", on="flight_id")
            merged_df['tow'] = merged_df['tow'].fillna(merged_df['tow_old'])
            return merged_df[["flight_id", "tow"]]

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

        if self.scaler is not None:
            with open(Path(get_models_path(), SCALER_FILE), "wb") as f:
                pickle.dump(self.scaler, f)

        if self.pca is not None:
            with open(Path(get_models_path(), PCA_FILE), "wb") as f:
                pickle.dump(self.pca, f)
        logger.debug("Model was successfully saved.")

    @staticmethod
    def load():
        """ Load the model from file """
        with open(Path(get_models_path(), MODEL_FILE), "rb") as f:
            model = pickle.load(f)

        with open(Path(get_models_path(), ENCODER_FILE), "rb") as f:
            encoder = pickle.load(f)

        scaler = None
        if Path(get_models_path(), SCALER_FILE).exists():
            with open(Path(get_models_path(), SCALER_FILE), "rb") as f:
                scaler = pickle.load(f)

        pca = None
        if Path(get_models_path(), PCA_FILE).exists():
            with open(Path(get_models_path(), PCA_FILE), "rb") as f:
                pca = pickle.load(f)

        return model, encoder, scaler, pca

    @staticmethod
    def _preprocess_features(features_df: pd.DataFrame):
        features_df["month"] = features_df["date"].dt.month
        features_df["day_of_week"] = features_df["date"].dt.dayofweek
        features_df["actual_offblock_hour"] = features_df["actual_offblock_time"].dt.hour
        features_df["arrival_hour"] = features_df["arrival_time"].dt.hour

        features_df['route'] = features_df[['adep', 'ades']].agg('-'.join, axis=1)
        features_df = features_df.drop(columns=["date"])
        return features_df

    def load_data_for_model_fit(self, folder_with_files: Path):
        return read_challenge_set(folder_with_files)

    def load_data_for_submission(self, folder_with_files: Path):
        return read_submission_set(folder_with_files)

    @contextmanager
    def close_dask_client_after_execution(self):
        self.dask_handler = DaskClientWrapper()
        try:
            yield
        finally:
            close_dask_client()
