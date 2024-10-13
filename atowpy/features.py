from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger
from traffic.core import Traffic

from atowpy.paths import get_data_path


def _is_file_was_assimilated_already(parquet_file: Path, extraction_file_path: Path, reference_df: pd.DataFrame) -> bool:
    """ Checking whether the file was assimilated already """
    if extraction_file_path.exists() is False:
        # File with results was not created yet
        return False

    extracted_features = pd.read_csv(extraction_file_path)
    assimilated_flights = set(extracted_features["flight_id"].unique())


def _assimilate_file(parquet_file: Path, extraction_file_path: Path, reference_df: pd.DataFrame):
    """ Start process of feature extraction """
    if extraction_file_path.exists():
        updated_results = [pd.read_csv(extraction_file_path)]
    else:
        updated_results = []

    date = parquet_file.name.split('.parquet')[0]
    date = datetime.strptime(date, '%Y-%m-%d')

    date_df = reference_df[reference_df['date'] == date]

    t = (Traffic.from_file(parquet_file)
         .filter()
         .resample('1s')
         .eval())
    for _, row in date_df.iterrows():
        flight_info = t.query(f'flight_id == {row.flight_id}')

        start = row.actual_offblock_time.strftime("%Y-%m-%d %H:%M")
        end = row.arrival_time.strftime("%Y-%m-%d %H:%M")

        dataframe_for_analysis = flight_info.between(start, end).data


def trajectory_features_preparation(reference_file: Union[Path, str]):
    """ Iterate through .parquet files and generate features for machine learning models

    :param reference_file:
    """
    if isinstance(reference_file, str):
        reference_file = Path(reference_file)

    # Define the place where to save the results
    file_base_name = reference_file.name.split('.csv')[0]
    file_with_results = f"extracted_trajectory_features_for{file_base_name}.csv"
    extraction_file_path = Path(get_data_path(), file_with_results)

    parquet_files = list(filter(lambda x: x.name.endswith('.parquet'), list(get_data_path().iterdir())))
    reference_df = pd.read_csv(reference_file, parse_dates=['date', 'actual_offblock_time', 'arrival_time'])

    for file in parquet_files:
        logger.info(f"Reference file '{reference_file.name}'. Parquet file {file.name} is processing ...")
        if _is_file_was_assimilated_already(parquet_file=file,
                                            extraction_file_path=extraction_file_path,
                                            reference_df=reference_df):
            logger.debug(f'Finished processing. File was already successfully assimilated')
            continue

        starting_time = datetime.now()
        _assimilate_file(file, extraction_file_path, reference_df)
        spend_time = datetime.now() - starting_time
        logger.debug(f'Finished processing. Spend seconds: {spend_time.total_seconds()}')

