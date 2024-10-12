from pathlib import Path

from atowpy.explore import DataExplorer
from atowpy.paths import get_data_path

import warnings
warnings.filterwarnings('ignore')


def explore_dataset():
    """ Script for initial dataset scanning """
    explorer = DataExplorer(get_data_path())

    explorer.show_flight_list()
    explorer.show_submission_set()


if __name__ == '__main__':
    explore_dataset()
