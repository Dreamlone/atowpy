import pickle
from pathlib import Path
from loguru import logger

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import contextily as cx
from scipy.ndimage import gaussian_filter

from atowpy.model.simple import SimpleModel
from atowpy.paths import results_folder, get_submissions_path
from atowpy.project import prepare_points_layer
from atowpy.read import read_challenge_set, read_submission_set


FEATURES_TO_EXCLUDE = ["flight_id"]
# Remain only 'altitude', 'groundspeed', 'u_component_of_wind', 'v_component_of_wind', 'latitude', 'longitude',
# 'vertical_rate' variables in the dataset
FEATURES_BASE_NAMES_TO_REMOVE = ['track',
                                 'temperature', 'specific_humidity',
                                 'track_unwrapped']


class TrajectoryModel(SimpleModel):
    """ Advanced machine learning model which utilizes
    """

    def __init__(self, model: str = "load", apply_validation: bool = True,
                 vis: bool = True):
        super().__init__(model, apply_validation)
        self.vis = vis

        # In case if during the prediction features was not extracted for each flight
        self.was_prediction_data_full: bool = True
        self.path_to_backup_submission = Path(get_submissions_path(),
                                              "team_loyal_hippo_v6_d6020e5c-d553-4262-acfa-cb16ab34cc86.csv")

    def load_data_for_model_fit(self, folder_with_files: Path):
        extracted = Path(folder_with_files, "extracted_trajectory_features_for_challenge_set.csv")
        extracted = pd.read_csv(extracted)
        extracted = extracted.drop(columns=["date"])

        # Extend features names
        extracted_names = list(filter(lambda x: x not in FEATURES_TO_EXCLUDE, list(extracted.columns)))
        for feature in FEATURES_BASE_NAMES_TO_REMOVE:
            extracted_names = list(filter(lambda x: feature not in x, extracted_names))
        extracted_names.sort()
        logger.debug(f"Feature to use: {extracted_names}")
        self.num_features.extend(extracted_names)

        challenge_set = read_challenge_set(folder_with_files)
        merged = challenge_set.merge(extracted, on="flight_id")
        merged_cols = list(merged.columns)
        altitude_names = list(filter(lambda x: "altitude" in x, merged_cols))
        groundspeed_names = list(filter(lambda x: "groundspeed" in x, merged_cols))
        track_names = list(filter(lambda x: "track_lag" in x, merged_cols))
        lats = list(filter(lambda x: "latitude" in x, merged_cols))
        lons = list(filter(lambda x: "longitude" in x, merged_cols))

        if self.vis:
            sigma_value = 2
            # Prepare visualizations with extracted features
            # TODO redesign, this logic is too complicated
            batch_size = 5
            folder = results_folder("exploration_plots")
            i = 0
            ax_id = 0
            for row_id, row in merged.iterrows():
                if row_id % batch_size == 0:
                    ax_id = 0
                    if i != 0:
                        logger.debug(f"Finished creating extracted features plot number {i}")
                        # Need to save previous picture
                        plt.savefig(Path(folder, f'extracted_features_{i}.png'),
                                    dpi=300, bbox_inches='tight')
                        plt.close()

                    # Create new picture
                    i += 1
                    if i >= 8:
                        break
                    fig_size = (15.0, 16)
                    fig, axs = plt.subplots(batch_size, 4, figsize=fig_size)

                # Do the actual visualization
                logger.debug(f"Batch {i}. Generating row {ax_id}")
                axs[ax_id, 0].plot(np.array(row[altitude_names], dtype=float), c='blue')
                axs[ax_id, 0].plot(gaussian_filter(np.ravel(np.array(row[altitude_names], dtype=float)),
                                                   sigma=sigma_value),
                                   '--', c='blue', alpha=0.5)
                axs[ax_id, 0].set_title(f"Altitude for flight {row.name_adep} - {row.name_ades}.",
                                        fontsize=8)
                axs[ax_id, 0].set_ylim(-50, 25000)
                axs[ax_id, 0].grid(True)
                axs[ax_id, 1].plot(np.array(row[groundspeed_names]), c='orange')
                axs[ax_id, 1].plot(gaussian_filter(np.ravel(np.array(row[groundspeed_names], dtype=float)),
                                                   sigma=sigma_value),
                                   '--', c='orange', alpha=0.5)
                axs[ax_id, 1].set_title(f"Groundspeed for flight {row.name_adep} - {row.name_ades}.",
                                        fontsize=8)
                axs[ax_id, 1].set_ylim(0, 400)
                axs[ax_id, 1].grid(True)
                axs[ax_id, 2].plot(np.array(row[track_names]), c='red')
                axs[ax_id, 2].set_title(f"Track for flight {row.name_adep} - {row.name_ades}.",
                                        fontsize=8)
                axs[ax_id, 2].grid(True)

                df_with_coordinates = pd.DataFrame({"lat": np.array(row[lats]),
                                                    "lon": np.array(row[lons]),
                                                    "time_index": np.arange(0, len(lats))})
                buffer = 0.1
                wider_borders = pd.DataFrame({"lat": [min(np.array(row[lats])) - buffer,
                                                      max(np.array(row[lats])) + buffer],
                                              "lon": [min(np.array(row[lons])) - buffer,
                                                      max(np.array(row[lons])) + buffer]})

                df_with_coordinates = prepare_points_layer(df_with_coordinates, lon="lon", lat="lat")
                wider_borders = prepare_points_layer(wider_borders, lon="lon", lat="lat")
                ax = wider_borders.plot(ax=axs[ax_id, 3], alpha=0.01, color='black')
                ax = df_with_coordinates.plot(ax=ax, color='black',
                                              zorder=1, markersize=3, edgecolor='black')
                ax = df_with_coordinates.plot(ax=ax, column='time_index', alpha=0.6, legend=True,
                                              cmap='Reds', legend_kwds={'label': "Datetime index"},
                                              zorder=1, markersize=2)
                cx.add_basemap(ax, crs=df_with_coordinates.crs, source=cx.providers.CartoDB.Voyager)
                ax.set_title(f"TOW {row.tow}", fontsize=10)
                ax_id += 1
        return merged

    def load_data_for_submission(self, folder_with_files: Path):
        extracted = Path(folder_with_files, "extracted_trajectory_features_for_final_submission_set.csv")
        extracted = pd.read_csv(extracted)
        extracted = extracted.drop(columns=["date"])

        # Extend features names
        extracted_names = list(filter(lambda x: x not in FEATURES_TO_EXCLUDE, list(extracted.columns)))
        for feature in FEATURES_BASE_NAMES_TO_REMOVE:
            extracted_names = list(filter(lambda x: feature not in x, extracted_names))
        extracted_names.sort()
        logger.debug(f"Feature to use: {extracted_names}")
        self.num_features.extend(extracted_names)

        df = read_submission_set(folder_with_files)
        submission_set_len = len(df)
        extracted_features_len = len(extracted)
        if submission_set_len != extracted_features_len:
            submission_flights = set(df["flight_id"])
            extracted_flights = set(extracted["flight_id"])
            uncovered_flights = submission_flights - extracted_flights
            logger.warning(f"Dataset with extracted features does not contain all the flights. "
                           f"Submission set length: {submission_set_len}. "
                           f"Extracted features set length: {extracted_features_len}. "
                           f"Missed flights: {submission_set_len - extracted_features_len}. "
                           f"Flight indices: {uncovered_flights}")
            self.was_prediction_data_full = False

        merged = df.merge(extracted, on="flight_id")
        return merged
