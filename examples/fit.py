from atowpy.model.simple import SimpleModel
from atowpy.paths import get_data_path


def fit_simple_model():
    """
    Example how to fit simple model and serialize it (save)
    """
    data_path = get_data_path()

    # Fit random forest model
    model = SimpleModel("rfr")
    model.fit(data_path)

    # Save fitted model
    model.save()


if __name__ == '__main__':
    fit_simple_model()
