from typing import Union, List

import pandas as pd
import dask.distributed
import dask.dataframe as dd
import dask.array as da
import dask.bag as db
from threading import Lock


class DaskClientMeta(type):
    """ Thread-safe realization of Dask Client singleton """
    # Objects for synchronization
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls):
        with cls._lock:
            if cls not in cls._instances:
                client = configure_local_client()
                instance = super().__call__(client)
                cls._instances[cls] = instance
        return cls._instances[cls]


class DaskClientWrapper(metaclass=DaskClientMeta):
    """
    Class for processing dask client.
    Ensures that only one Dask client is generated at
    a time to run dask during algorithm execution
    """

    def __init__(self, client: Union[dask.distributed.Client, None]):
        self.client = client

    def close(self):
        """ Dask client shutdown """
        self.client.close()
        self.client.shutdown()
        # Remove all instances
        DaskClientMeta._instances = {}


def configure_local_client():
    cluster = dask.distributed.LocalCluster()
    client = dask.distributed.Client(cluster)
    return client


def close_dask_client():
    client_holder = DaskClientWrapper()
    client_holder.close()


def convert_into_dask_array(array):
    if isinstance(array, da.Array):
        return array

    array_shape = array.shape
    if len(array_shape) <= 1:
        # Convert into column
        array = array.reshape(-1, 1)
    converted = da.from_array(array, chunks=3)
    return converted


def convert_into_dask_dataframe(table):
    """ Convert pandas dataframe (if necessary) into dask dataframe """
    if type(table) is dd.DataFrame:
        return table

    elif isinstance(table, List):
        partition_size = 700000

        # Convert list with recordings (dictionaries) into Bag and then into dataframe
        table = db.from_sequence(table, partition_size=partition_size)
        table = table.to_dataframe()
        return table
    elif type(table) is pd.DataFrame:
        partition_size = 700000

        table = dd.from_pandas(table, chunksize=partition_size)
        return table
