from datetime import datetime
from pathlib import Path
from typing import Union, Optional, List
from matplotlib import pyplot as plt

import pandas as pd
from loguru import logger
import seaborn as sns
from traffic.core import Traffic

from atowpy.paths import results_folder
from atowpy.read import read_challenge_set, read_submission_set

DEFAULT_DATETIME_CHECK = datetime.strptime('2022-02-01', '%Y-%m-%d')


class DataExplorer:
    """ Helper class for creating visualizations """

    def __init__(self, working_directory: Union[Path, str],
                 check_before: Union[datetime, None] = DEFAULT_DATETIME_CHECK):
        if isinstance(working_directory, str):
            working_directory = Path(working_directory)

        self.working_directory = working_directory.resolve()
        self.results_folder = results_folder("exploration_plots")

        self.results_folder_details = Path(self.results_folder, "details")
        self.results_folder_details.mkdir(exist_ok=True, parents=True)
        self.check_before = check_before

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
            - tow: TakeOff Weight: tow [kg], target
        """
        df = read_challenge_set(self.working_directory)
        if self.check_before is not None:
            df = df[df["date"] <= self.check_before]
            logger.warning(f"According to set check_before variable, dataset is clipped. Current len: {len(df)}")

        callsigns = list(df["callsign"].unique())
        callsigns.sort()
        logger.info(f"Flight list exploration. Number of 'unique callsign': {len(callsigns)}")

        aircraft_types = list(df["aircraft_type"].unique())
        aircraft_types.sort()
        logger.info(f"Flight list exploration. Number of unique 'aircraft_type': {len(aircraft_types)}")

        # Details
        self._trajectories_investigation_per_callsigns(df, callsigns)
        self._aircraft_type_tow_plot(df, aircraft_types)

        self._kde_plot(df, "date", "airline", "flight_list", "rainbow")
        self._kde_plot(df, "date", "aircraft_type", "flight_list", "coolwarm")
        self._kde_plot(df, "date", "wtc", "flight_list", "Spectral")

    def show_submission_set(self):
        """ Explore submission_set.csv file """
        submission_df = read_submission_set(self.working_directory)
        challenge_df = read_challenge_set(self.working_directory)

        # First step - checking flight_ids overlapping
        for column_to_check in ["flight_id", "callsign"]:
            submission_flight_ids = set(submission_df[column_to_check].unique())
            challenge_flight_ids = set(challenge_df[column_to_check].unique())
            common_ids = submission_flight_ids.intersection(challenge_flight_ids)
            if len(common_ids) < 1:
                logger.info(f"Overlapping checking. No overlapping detected for {column_to_check}.")
            else:
                logger.info(f"Overlapping checking. Overlapping detected for {column_to_check}."
                            f"Common indices {len(common_ids)}")

        self._kde_plot(submission_df, "date", "airline", "submission_set", "rainbow")
        self._kde_plot(submission_df, "date", "aircraft_type", "submission_set", "coolwarm")
        self._kde_plot(submission_df, "date", "wtc", "submission_set", "Spectral")

    def _trajectories_investigation_per_callsigns(self, df: pd.DataFrame, callsigns):
        """ Generate plots per callsign with tow variable and """
        threshold = 100
        for callsign in callsigns:
            logger.debug(f"Trajectories investigation per callsign {callsign}")
            df_callsign = df[df["callsign"] == callsign]

            if max(df_callsign["tow"]) - min(df_callsign["tow"]) < threshold:
                # The same weight all the time
                logger.debug(f"Difference is less than {threshold} kg for {callsign}. Skip")
                continue

            size_per_item = 2.0
            fig_size = (14.0, size_per_item * len(df_callsign))
            fig, axs = plt.subplots(len(df_callsign), 2, figsize=fig_size)
            plt.suptitle(f'Callsign {callsign}', fontsize=20)

            i = 0
            for _, row in df_callsign.iterrows():
                parquet_file = row.date.strftime("%Y-%m-%d.parquet")
                t = (Traffic.from_file(Path(self.working_directory, parquet_file))
                     .query(f'flight_id == {row.flight_id}')
                     .filter()
                     .resample('1s')
                     .eval())
                flight = t.query(f'flight_id == {row.flight_id}')

                start = row.actual_offblock_time.strftime("%Y-%m-%d %H:%M")
                end = row.arrival_time.strftime("%Y-%m-%d %H:%M")

                dataframe_for_analysis = flight.between(start, end).data

                axs[i, 0].plot(dataframe_for_analysis['timestamp'], dataframe_for_analysis['groundspeed'],
                               c='blue', label='groundspeed')
                axs[i, 0].set_title(f'Flight {row.date}. TOW {row.tow}')
                axs[i, 1].plot(dataframe_for_analysis['timestamp'], dataframe_for_analysis['altitude'],
                               c='red', label='altitude')
                axs[i, 1].set_title(f'Flight {row.date}. TOW {row.tow}')
                i += 1
            plt.show()

    def _aircraft_type_tow_plot(self, df: pd.DataFrame, aircraft_types):
        min_tow = min(df["tow"])
        max_tow = max(df["tow"])
        min_flight_duration = min(df["flight_duration"])
        max_flight_duration = max(df["flight_duration"])
        for aircraft_type in aircraft_types:
            logger.debug(f"Generating aircraft type vs tow plot for {aircraft_type}")
            df_aircraft_type = df[df["aircraft_type"] == aircraft_type]

            with sns.axes_style("darkgrid"):
                fig_size = (11.0, 4.0)
                fig, ax = plt.subplots(figsize=fig_size)
                title = f"Aircraft type: {aircraft_type}"
                ax = sns.stripplot(
                    data=df_aircraft_type, x="callsign", y="tow", hue="flight_duration",
                    hue_norm=(min_flight_duration, max_flight_duration),
                    palette="Reds", ax=ax)
                ax.set_ylim([min_tow, max_tow])
                ax.set(xticklabels=[])
                ax.set_title(title)
                plt.savefig(Path(self.results_folder_details, f'investigation_{aircraft_type}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()

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
