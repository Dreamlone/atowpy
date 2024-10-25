import xgboost as xgb
from xgboost import XGBRegressor
from loguru import logger
import numpy as np

from dask.array.core import Array
from dask_ml.model_selection import train_test_split

from atowpy.dask_wrappers import DaskClientWrapper, convert_into_dask_array


class DaskXGBRegressor:
    """
    Class wrapper for launching train and predict process for xgboost model
    with dask
    """

    def __init__(self, **params):
        self.params = params
        self.output = {}
        self.n_estimators = None
        if self.params is not None and 'n_estimators' in self.params.keys():
            self.n_estimators = self.params['n_estimators']

        self.booster = "gbtree"
        if self.params is not None and 'booster' in self.params.keys():
            self.booster = self.params['booster']

        self.eta = 0.3
        if self.params is not None and 'eta' in self.params.keys():
            self.eta = self.params['eta']

        self.max_depth = 6
        if self.params is not None and 'max_depth' in self.params.keys():
            self.max_depth = self.params['max_depth']

        self.gamma = 0
        if self.params is not None and 'gamma' in self.params.keys():
            self.gamma = self.params['gamma']

    def fit(self, features: Array, target: Array, dask_handler):
        logger.debug(f"DaskXGBRegressor. Start model fitting")
        logger.debug(f"n_estimators: {self.n_estimators}. max_depth: {self.max_depth}")

        x_train, x_test, y_train, y_test = train_test_split(features.compute_chunk_sizes(),
                                                            target.compute_chunk_sizes(),
                                                            test_size=0.2,
                                                            random_state=2)
        # Heuristic: recommend to use 5% of total iterations for early stopping
        early_stopping_rounds = round(self.n_estimators * 0.05)
        if early_stopping_rounds < 5:
            early_stopping_rounds = 5

        dtrain = xgb.dask.DaskDMatrix(dask_handler.client, features, target)
        xgb_params = {'verbosity': 0, 'tree_method': 'hist', 'objective': 'reg:squarederror',
                      'booster': self.booster, "eta": self.eta, "max_depth": self.max_depth,
                      "gamma": self.gamma}
        dval = xgb.dask.DaskDMatrix(dask_handler.client, x_test, y_test)

        self.output = xgb.dask.train(dask_handler.client, xgb_params, dtrain,
                                     num_boost_round=self.n_estimators,
                                     evals=[(dval, 'valid')], xgb_model=self.output.get('booster'),
                                     early_stopping_rounds=early_stopping_rounds)
        del dtrain
        return self.output

    def predict(self, features: Array, dask_handler):
        """ Generate predict on new data """
        prediction = xgb.dask.predict(dask_handler.client, self.output, features)
        return prediction
