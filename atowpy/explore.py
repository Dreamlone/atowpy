from pathlib import Path
from typing import Union, Optional, List
from matplotlib import pyplot as plt

import pandas as pd
from loguru import logger

from atowpy.paths import results_folder
from atowpy.read import read_challenge_set, read_submission_set


class DataExplorer:
    """ Helper class for creating visualizations """

    def __init__(self, working_directory: Union[Path, str]):
        if isinstance(working_directory, str):
            working_directory = Path(working_directory)

        self.working_directory = working_directory.resolve()
        self.results_folder = results_folder("exploration_plots")

    def show_flight_list(self):
        """ Explore

        Docs: https://ansperformance.eu/study/data-challenge/data.html#flight-list
            - flight identification: unique ID (flight_id), (obfuscated)
            callsign (callsign)
            - origin/destination: Aerodrome of DEParture (ADEP)
            (adep [ICAO code]), Aerodrome of DEStination (ADES)
            (ades [ICAO code]) and ancillary info, i.e. airport
            name (name_adep, name_ades) and country code (country_code_adep,
             country_code_ades [ISO2C])
            - timing: date of flight (date [ISO 8601 UTC date]), Actual
            Off-Block Time (AOBT) (actual_offblock_time [ISO 8601 UTC date
            and time]), ARriVal Time (ARVT) (arrival_time [ISO 8601 UTC date
            and time)
            - aircraft: aircraft type code (aircraft_type [ICAO aircraft type]),
            Wake Turbulence Category (WTC) (wtc)
            - airline: (obfuscated) Aircraft Operator (AO) code (airline),
            - operational values: flight duration (flight_duration [min]) ,
            taxi-out time (taxiout_time [min]),
            route length (flown_distance [nmi]),
            (estimated) TakeOff Weight (TOW) (tow [kg])

        """
        df = read_challenge_set(self.working_directory)
        # TODO generate pictures here

    def show_submission_set(self):
        """ Explore submission_set.csv file """
        submission_df = read_submission_set(self.working_directory)
        challenge_df = read_challenge_set(self.working_directory)

        # First step - checking flight_ids overlapping
        submission_flight_ids = set(submission_df["flight_id"].unique())
        challenge_flight_ids = set(challenge_df["flight_id"].unique())
        common_ids = submission_flight_ids.intersection(challenge_flight_ids)
        if len(common_ids) < 1:
            logger.info(f"Flight indices overlapping. No overlapping detected.")
        else:
            logger.info(f"Flight indices overlapping. Overlapping detected."
                        f"Common indices {common_ids}")

        # Use seaborn to show difference between train and test
