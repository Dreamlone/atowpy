from pathlib import Path

import pandas as pd


def read_challenge_set(working_directory: Path):
    file_name = "challenge_set.csv"
    df = pd.read_csv(Path(working_directory, file_name),
                     parse_dates=['date', 'actual_offblock_time',
                                  'arrival_time'])
    return df


def read_submission_set(working_directory: Path):
    file_name = "submission_set.csv"
    df = pd.read_csv(Path(working_directory, file_name),
                     parse_dates=['date'])
    return df
