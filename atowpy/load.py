from pathlib import Path
from typing import Union, Optional, List
from loguru import logger

from pyopensky.s3 import S3Client


class DataLoader:
    """ Class for downloading data from S3 bucket """

    def __init__(self, working_directory: Union[Path, str]):
        if isinstance(working_directory, str):
            working_directory = Path(working_directory)

        self.working_directory = working_directory.resolve()
        if self.working_directory.exists() is False:
            self.working_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def show_content(bucket_name: str = "competition-data"):
        s3 = S3Client()
        for obj in s3.s3client.list_objects(bucket_name, recursive=True):
            logger.info(f"Bucket: {obj.bucket_name}. Object {obj.object_name}")

    def download_into_folder(self, bucket_name: str = "competition-data",
                             files_to_process: Optional[List] = None):
        """ Load files into folder """
        s3 = S3Client()

        for obj in s3.s3client.list_objects(bucket_name, recursive=True):
            logger.debug(f"Loading... bucket: {obj.bucket_name}, object name: {obj.object_name}")

            if Path(self.working_directory, obj.object_name).exists() is True:
                logger.debug("Skip file because it is already loaded")
            else:
                if files_to_process is None or obj.object_name in files_to_process:
                    s3.download_object(obj, self.working_directory)
                    logger.debug("File was successfully loaded")
                else:
                    logger.debug("File is ignored")
