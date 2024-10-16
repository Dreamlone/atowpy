from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from loguru import logger
from traffic.core import Traffic

from atowpy.paths import get_data_path


FEATURES_FOR_AGGREGATION = ['latitude', 'longitude', 'altitude', 'groundspeed', 'track', 'vertical_rate',
                            'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity',
                            'track_unwrapped']


def _is_file_was_assimilated_already(parquet_file: Path, extraction_file_path: Path, reference_df: pd.DataFrame) -> bool:
    """ Checking whether the file was assimilated already """
    if extraction_file_path.exists() is False:
        # File with results was not created yet
        return False

    date_str = parquet_file.name.split('.parquet')[0]

    extracted_features = pd.read_csv(extraction_file_path)
    processed_date_df = extracted_features[extracted_features["date"] == date_str]

    # Moving with overlapping
    if processed_date_df is None or len(processed_date_df) < 5:
        return False

    return True


def _assimilate_file(parquet_file: Path, extraction_file_path: Path, reference_df: pd.DataFrame):
    """
    Start process of feature extraction from the parquet file
    """
    date_str = parquet_file.name.split('.parquet')[0]
    date = datetime.strptime(date_str, '%Y-%m-%d')
    date_previous_day = date - timedelta(days=1)
    date_next_day = date + timedelta(days=1)

    previous_day = reference_df[reference_df['date'] == date_previous_day]
    date_df = reference_df[reference_df['date'] == date]
    next_df = reference_df[reference_df['date'] == date_next_day]

    batch_features = []
    t = (Traffic.from_file(parquet_file)
         .filter()
         .resample('1s')
         .eval())
    processed_flights = []
    processed_dates = []
    for df, label in zip([previous_day, date_df, next_df], ['past', 'present', 'future']):
        if len(df) < 1:
            continue

        # Iterate with overlapping
        for _, row in df.iterrows():
            # Iterate through each row
            flight_info = t.query(f'flight_id == {row.flight_id}')
            start = row.actual_offblock_time.strftime("%Y-%m-%d %H:%M")
            end = row.arrival_time.strftime("%Y-%m-%d %H:%M")

            if flight_info is None:
                logger.debug(f"Skip {row.flight_id}. Start {start}. End {end}."
                             f"Because there is no that flight in the parquet file")
                continue

            logger.debug(f"Start extracting features for flight {row.flight_id}")
            try:
                dataframe_for_analysis = flight_info.between(start, end).data
                # Take first 20 elements to save as features
                dataframe_for_analysis = dataframe_for_analysis.head(20)
                extracted_features = []
                extracted_features_names = []
                for feature_name in FEATURES_FOR_AGGREGATION:
                    feature = np.array(dataframe_for_analysis[feature_name])
                    extracted_features.append(np.ravel(feature))

                    new_features_names = [f"{feature_name}_lag_{i}" for i in range(len(feature))]
                    extracted_features_names.extend(new_features_names)

                extracted_features = pd.DataFrame(np.hstack(extracted_features).reshape((1, -1)),
                                                  columns=extracted_features_names)
                batch_features.append(extracted_features)
                processed_flights.append(row.flight_id)
                processed_dates.append(date_str)
            except Exception as ex:
                logger.warning(f"Skip {row.flight_id}. Start {start}. End {end}. Because of the {ex}")

    batch_features = pd.concat(batch_features)
    batch_features["flight_id"] = processed_flights
    batch_features["date"] = processed_dates
    batch_features = batch_features.drop_duplicates(subset=["flight_id"])

    # Extend previous results with new batch
    if extraction_file_path.exists():
        updated_results = [pd.read_csv(extraction_file_path)]
    else:
        updated_results = []

    updated_results.append(batch_features)
    updated_results = pd.concat(updated_results)
    updated_results = updated_results.drop_duplicates(subset=["flight_id"])

    updated_results.to_csv(extraction_file_path, index=False)
    logger.debug(f"Result was stored into the file {extraction_file_path.name}")


def trajectory_features_preparation(reference_file: Union[Path, str]):
    """ Iterate through .parquet files and generate features for machine learning models

    :param reference_file:
    """
    if isinstance(reference_file, str):
        reference_file = Path(reference_file)

    # Define the place where to save the results
    file_base_name = reference_file.name.split('.csv')[0]
    file_with_results = f"extracted_trajectory_features_for_{file_base_name}.csv"
    extraction_file_path = Path(get_data_path(), file_with_results)

    parquet_files = list(filter(lambda x: x.name.endswith('.parquet'), list(get_data_path().iterdir())))
    parquet_files.sort()
    reference_df = pd.read_csv(reference_file, parse_dates=['date', 'actual_offblock_time', 'arrival_time'])

    for file in parquet_files:
        logger.info(f"Reference file '{reference_file.name}'. Parquet file {file.name} is processing ...")
        if _is_file_was_assimilated_already(parquet_file=file,
                                            extraction_file_path=extraction_file_path,
                                            reference_df=reference_df):
            logger.info(f'Finished assimilating process. File was already successfully assimilated')
            continue

        starting_time = datetime.now()
        _assimilate_file(file, extraction_file_path, reference_df)
        spend_time = datetime.now() - starting_time
        logger.info(f'Finished assimilating process. Spend seconds: {spend_time.total_seconds()}')
