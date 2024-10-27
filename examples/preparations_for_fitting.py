from pathlib import Path

from atowpy.features import trajectory_features_preparation
from atowpy.paths import get_data_path


def prepare_trajectory_features():
    """
    Launch feature extraction from trajectory files.

    Important! This script is time-consuming (may take days), so it
     can be launched iteratively (stop and re-launch again).
     Thus, this script is stateful
    """
    # Prepare features for train first
    for dataset in ["challenge_set.csv", "final_submission_set.csv"]:
        file_path = Path(get_data_path(), dataset)
        trajectory_features_preparation(file_path)


if __name__ == '__main__':
    prepare_trajectory_features()
