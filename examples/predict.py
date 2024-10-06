from pathlib import Path

from atowpy.model.simple import SimpleModel
from atowpy.paths import get_data_path, get_models_path
from atowpy.submit import save_prediction_dataframe_as_file
from atowpy.version import MODEL_FILE


def make_prediction():
    """ Use serialized model to generate prediction """
    data_path = get_data_path()

    model = SimpleModel("load")

    predicted_dataframe = model.predict(data_path)
    save_prediction_dataframe_as_file(predicted_dataframe)


if __name__ == '__main__':
    make_prediction()
