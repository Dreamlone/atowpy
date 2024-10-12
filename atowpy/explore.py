from pathlib import Path
from typing import Union, Optional, List
from matplotlib import pyplot as plt

import pandas as pd
from loguru import logger
import seaborn as sns

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
        """
        Explore dataset through visualizations

        List of columns:
            - flight_id: unique ID of the flight, (obfuscated)
            - date: date of the flight
            - callsign: aircraft route id (for example flight from London
                Heathrow to Cork will have the same callsign id)
            - adep: Aerodrome of DEParture (ADEP)
            - name_adep: ...
            - country_code_adep: ...
            - ades: Aerodrome of DEStination (ADES)
            - name_ades: ...
            - country_code_ades: ...
            - actual_offblock_time: Actual Off-Block Time (AOBT)
            - arrival_time: arrival time
            - aircraft_type: aircraft type code
            - wtc: Wake Turbulence Category
            - airline: Aircraft Operator
            - flight_duration: flight duration, min
            - taxiout_time: taxiout_time, min
            - flown_distance: route length, nmi
            - tow: TakeOff Weight: tow [kg]
        """
        df = read_challenge_set(self.working_directory)

        # Check dependencies with target

        self._kde_plot(df, "date", "airline", "flight_list", "rainbow")
        self._kde_plot(df, "date", "aircraft_type", "flight_list", "coolwarm")
        self._kde_plot(df, "date", "wtc", "flight_list", "Spectral")

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

        self._kde_plot(submission_df, "date", "airline", "submission_set", "rainbow")
        self._kde_plot(submission_df, "date", "aircraft_type", "submission_set", "coolwarm")
        self._kde_plot(submission_df, "date", "wtc", "submission_set", "Spectral")

    def _kde_plot(self, df: pd.DataFrame, x_col: str, y_col: str, label: str, palette: str = "rainbow"):
        """
        Generate kde plot
        Reference: https://seaborn.pydata.org/examples/multiple_conditional_kde.html
        """
        logger.debug(f"Generating KDE plot '{label}'. X column: '{x_col}'. Y column: '{y_col}'")

        fig_size = (12.0, 7.0)
        fig, ax = plt.subplots(figsize=fig_size)

        height = 7
        with sns.axes_style("whitegrid"):
            sns.displot(
                data=df,
                x=x_col, hue=y_col,
                kind="kde", height=height,
                multiple="fill",
                palette=palette, ax=ax)
            plt.savefig(Path(self.results_folder, f'{label}_kde_{x_col}_{y_col}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()
