from atowpy.model.simple import SimpleModel
from atowpy.model.trajectory import TrajectoryModel
from atowpy.paths import get_data_path


import warnings
warnings.filterwarnings('ignore')


def fit_trajectory_model():
    """
    Example how to fit trajectory model and serialize it (save)
    """
    data_path = get_data_path()

    # Fit random forest model
    model = TrajectoryModel("xgb_dask", apply_validation=True, vis=True)
    # model.fit(data_path)
    model.fit_one_model(data_path, {"n_estimators": 2000, "booster": "gbtree", "eta": 0.06719124193529334,
                                           "max_depth": 7, "gamma": 1.9932175141782538})

    # Save fitted model
    model.save()


if __name__ == '__main__':
    fit_trajectory_model()
