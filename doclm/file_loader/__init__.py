"""
 this module implements file read capabilities form on prem to azure blob
"""
import os
from io import BytesIO

from .azureblob import AzureBlobFileDownloader, log


class LoadLocal:
    """
    Class to read from local path
    """
    def __call__(self, file, *args, **kwargs):
        with open(file, "rb") as f_h:
            stream = BytesIO(f_h.read())
            stream.seek(0, 0)
        return stream

    async def __acall__(self, file, *args, **kwargs):
        with open(file, "rb") as f_h:
            stream = BytesIO(f_h.read())
            stream.seek(0, 0)
        return stream

    def close(self):
        pass


def azure_reader():
    """
    :return: azure blobfile downloader
    """
    blob_name = os.environ['AZ_CONT_NAME']
    connection_string = os.environ['AZ_CONN_STRING']
    return AzureBlobFileDownloader(connection_string, blob_name)


loaders = {
    'azure_blob': azure_reader,
    'local': LoadLocal
}
if os.getenv('LOGDIR', None):
    os.makedirs(os.getenv('LOGDIR'), exist_ok=True)


def load_reader(name="azure_blob"):
    log.info("loading file loader")
    assert name in loaders
    return loaders[name]()
