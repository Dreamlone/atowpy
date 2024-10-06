from pathlib import Path

import pandas as pd

from atowpy.paths import get_submissions_path
from atowpy.version import SUBMISSION_FILE


def save_prediction_dataframe_as_file(predicted_df: pd.DataFrame):
    file_path = Path(get_submissions_path(), SUBMISSION_FILE)
    predicted_df[["flight_id", "tow"]].to_csv(file_path, index=False)
