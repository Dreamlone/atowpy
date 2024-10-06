from atowpy.load import DataLoader
from atowpy.paths import get_data_path


def download_data_into_local_folder():
    """
    Example how to use data loader to get files in local folder
    class conveniently allows to define where to store the data (folder)
    and will not try to re-load already copied files
    """
    loader = DataLoader(get_data_path())
    loader.download_into_folder()


if __name__ == '__main__':
    download_data_into_local_folder()
