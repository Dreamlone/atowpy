from pathlib import Path

from atowpy.explore import DataExplorer
from atowpy.paths import get_data_path


def explore_dataset():
    """ Script for initial dataset scanning """
    data_path = get_data_path()
    explorer = DataExplorer(get_data_path())

    explorer.show_submission_set()
    explorer.show_flight_list()


if __name__ == '__main__':
    explore_dataset()
