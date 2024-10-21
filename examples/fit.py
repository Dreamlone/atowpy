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
    model = TrajectoryModel("xgb_dask", apply_validation=True)
    model.fit(data_path)

    # Save fitted model
    model.save()


if __name__ == '__main__':
    fit_trajectory_model()
