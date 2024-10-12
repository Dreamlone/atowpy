from pathlib import Path


def get_project_path() -> Path:
    return Path(__file__).parent.parent


def get_data_path():
    return Path(get_project_path(), 'data')


def get_models_path():
    models_folder = Path(get_project_path(), 'models')
    models_folder.mkdir(parents=True, exist_ok=True)
    return models_folder


def get_submissions_path():
    submission_folder = Path(get_project_path(), 'submissions')
    submission_folder.mkdir(parents=True, exist_ok=True)
    return submission_folder


def results_folder(folder_name: str) -> Path:
    """ Return path to the folder with results (create folder if needed) """
    folder = Path(get_project_path(), folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    return folder
