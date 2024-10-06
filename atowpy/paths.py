from pathlib import Path


def get_project_path() -> Path:
    return Path(__file__).parent.parent


def get_data_path():
    return Path(get_project_path(), 'data')


def get_models_path():
    return Path(get_project_path(), 'models')


def get_submissions_path():
    return Path(get_project_path(), 'submissions')


def results_folder(folder_name: str) -> Path:
    """ Return path to the folder with results (create folder if needed) """
    folder = Path(get_project_path(), folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    return folder
