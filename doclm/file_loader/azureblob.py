import os
import logging
from io import BytesIO

from tenacity import retry, stop_after_attempt, wait_random_exponential
from azure.storage.blob import ContainerClient
from azure.storage.blob.aio import ContainerClient as aContainerClient
from ..encryption import AESCipher, get_file_psk

log = logging.getLogger("doclogger")
log.disabled = False


class AzureBlobFileDownloader:
    """
    Class to download from azure blob, object is called return a BytesIO object
    """
    def __init__(self, conn_string, contain_name):
        """

        :param conn_string: connection string to azure blob
        :param contain_name: container name for blob
        """
        self.conn_string = conn_string
        self.container_name = contain_name
        self.container_client = None
        self.acontainer_client = None

    async def get_async_container_client(self):
        """creates async blob container client """
        if self.acontainer_client is None:
            log.info("creating acontainer_client")
            self.acontainer_client = aContainerClient.from_connection_string(
                conn_str=self.conn_string, container_name=self.container_name
            )
        return self.acontainer_client

    def get_sync_container_client(self):
        """creates blob container client """
        if self.container_client is None:
            log.info("creating container_client")
            self.container_client = ContainerClient.from_connection_string(
                conn_str=self.conn_string, container_name=self.container_name
            )
        return self.container_client

    def __call__(self, *args, **kwargs):
        self.get_sync_container_client()
        return self.get_file_handle(*args, **kwargs)

    async def __acall__(self, *args, **kwargs):
        await self.get_async_container_client()
        return await self.aget_file_handle(*args, **kwargs)

    def list_files(self):
        return list(self.container_client.list_blob_names())

    @retry(wait=wait_random_exponential(min=1, max=70), stop=stop_after_attempt(2))
    def get_file_handle(self, name, secret):
        """
        Downloads blob with name {name}, and decrypts with secret {secret}
        :param name: azure Blob name
        :param secret: Decrypt secretes if file is encrypted
        :return: BytesIO stream
        """
        log.info(" downloading %s from Azure blob", name)
        time_out = int(os.getenv('AZURE_TIMEOUT', '60'))
        stream_downloader = self.container_client.download_blob(name, timeout=time_out)
        log.info("Creating BytesIO Object")
        stream = BytesIO()
        log.info("writing from Azure blob to BytesIO Object")
        stream_downloader.readinto(stream)
        return self._decrypt(stream, name, secret)

    @retry(wait=wait_random_exponential(min=1, max=70), stop=stop_after_attempt(2))
    async def aget_file_handle(self, name, secret):
        """
        Asynchronously Downloads blob with name {name}, and decrypts with secret {secret}
        :param name: azure Blob name
        :param secret: Decrypt secretes if file is encrypted
        :return: BytesIO stream
        """
        log.info("downloading %s from Azure blob", name)
        time_out = int(os.getenv('AZURE_TIMEOUT', '60'))
        stream_downloader = await self.acontainer_client.download_blob(name, timeout=time_out)
        log.info("Creating BytesIO Object")
        stream = BytesIO()
        log.info("writing from Azure blob to BytesIO Object")
        await stream_downloader.readinto(stream)
        return self._decrypt(stream, name, secret)

    @staticmethod
    def _decrypt(stream: BytesIO, name, secret):
        stream.seek(0, 0)

        if secret:
            logging.info("decrypting BytesIO Object")
            decrypted_file_key = AESCipher(get_file_psk()).decrypt(secret)
            decrypted_file = AESCipher(decrypted_file_key).decrypt_file(stream, name)
            logging.info("Closing BytesIO copy Object")
            stream.close()
            logging.info("returning decrypted file Object")
            return BytesIO(decrypted_file.read())

        logging.info("returning BytesIO Object")
        return stream

    def close(self):
        """
        closes the blob connection
        :return: None
        """
        if self.container_client is not None:
            print('closing container_client')
            self.container_client.close()
            self.container_client = None
        elif self.acontainer_client is not None:
            print('closing acontainer_client')
            self.acontainer_client.close()
            self.acontainer_client = None
