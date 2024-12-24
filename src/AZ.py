"""
Author: Noah Alex
Contact: noahcalex@gmail.com
Year: 2024
Company: Grandeur Peak Global Advisors
"""

import os
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import (
    BlobServiceClient,
    BlobClient,
    ContainerClient,
    BlobPrefix,
)
from dataclasses import dataclass
import pandas as pd
from io import BytesIO


@dataclass
class Azure:
    def __init__(self, environment: str = "DEV"):
        """
        Initializes the Azure class.
        """
        self.environment = environment.upper()

        self.keyVaultName = os.environ[f"{self.environment}_KEY_VAULT_NAME"]

        KVUri = f"https://{self.keyVaultName}.vault.azure.net"
        credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=KVUri, credential=credential)

        self.connection_string = os.environ[
            f"{self.environment}_AZURE_STORAGE_CONNECTION_STRING"
        ]
        self.container_name = "llm-blob"

        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        self.container_client = self.blob_service_client.get_container_client(
            self.container_name
        )
        if not self.container_client.exists():
            self.container_client = self.blob_service_client.create_container(
                self.container_name
            )

    def upload_blob(
        self, file_name: str, blob_name: str, overwrite: bool = False
    ) -> None:
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name, blob=blob_name
        )
        with open(file_name, "rb") as data:
            blob_client.upload_blob(data, overwrite=overwrite)

    def list_blobs(self, print_blob_names: bool = False) -> list:
        blobs = self.container_client.list_blobs()
        if print_blob_names:
            for blob in blobs:
                print(blob.name)
        return blobs

    def safe_cast(self, val, to_type, default=None):
        try:
            return to_type(val)
        except (ValueError, TypeError):
            return default

    def get_latest_blob_folder_number(self, folder) -> int:
        blobs = self.list_blobs()
        found_blobs = [
            blob.name.split(folder)[1]
            for blob in blobs
            if len(blob.name.split(folder)) > 1
        ]

        subfolders = [self.safe_cast(blob.split("/")[0], int) for blob in found_blobs]

        if subfolders == [] or subfolders == [None]:
            return 0
        else:
            latest = sorted(subfolders)[-1]
            return latest

    def get_latest_blob_file(self, folder: str, file_name: str) -> None:
        latest_folder = self.get_latest_blob_folder_number(folder=folder)
        dir = f"{folder}{latest_folder}/{file_name}"
        local_path = f"{file_name}"
        self.get_blob(blob_name=dir, local_path=local_path)
        return local_path

    def close(self) -> None:
        self.client.close()
        self.container_client.close()
        self.blob_service_client.close()

    def _get_blob_client(self, blob_name: str):
        return self.blob_service_client.get_blob_client(
            container=self.container_name, blob=blob_name
        )

    def _get_blob(self, blob_name: str):
        blob_client = self._get_blob_client(blob_name=blob_name)
        return blob_client.download_blob()

    def get_blob(self, blob_name: str, local_path: str = "/downloads") -> None:
        """
        Gets a blob from the blob container.
        """
        with open(local_path, "wb") as download_file:
            data = self._get_blob(blob_name=blob_name)
            data.readinto(download_file)

    def get_buffered_streams(self, blob_names=None, blob_list=None):
        buffered_streams = []
        if blob_names:
            for blob_name in blob_names:
                stream = self._get_blob(blob_name).readall()
                buffer = BytesIO(stream)
                buffer.seek(0)
                buffered_streams.append(buffer)
        elif blob_list:
            for blob in blob_list:
                blob_name = blob.name
                stream = self._get_blob(blob_name).readall()
                buffer = BytesIO(stream)
                buffer.seek(0)
                buffered_streams.append(buffer)

        return buffered_streams

    def delete_blob(self, blob_name: str) -> None:
        """
        Deletes a blob from the blob container.
        """
        blob_client = self._get_blob_client(blob_name=blob_name)
        blob_client.delete_blob()

    def blob_to_df(self, blob_name: str) -> None:
        blob_data = self._get_blob(blob_name=blob_name)
        data = blob_data.readall().decode("utf-8")
        df = pd.read_csv(pd.compat.StringIO(data))
        return df

    def set_secret(self) -> None:
        """
        Creates a secret in the key vault.
        """

        secretName = input("Input a name for your secret > ")
        secretValue = input("Input a value for your secret > ")

        print(
            f"Creating a secret in {self.keyVaultName} called '{secretName}' with the value '{secretValue}' ..."
        )
        self.client.set_secret(secretName, secretValue)
        print(" done.")

    def __str__(self) -> None:
        return self.keyVaultName

    def get_secret(self, secret_name: str):
        """
        Gets a secret from the key vault.
        """
        print(f"Retrieving your secret from {self.keyVaultName}.")
        retrieved_secret = self.client.get_secret(secret_name)
        print(f"Your secret is '{retrieved_secret.value}'.")
        return retrieved_secret.value

    def delete_secret(self, secret_name: str):
        """
        Deletes a secret from the key vault.

            inputs:
                secret_name (str): The name of the secret to delete.
            returns:
                None
        """

        print(f"Deleting secret from {self.keyVaultName}")
        poller = self.client.begin_delete_secret(secret_name)
        deleted_secret = poller.result()
        print(" done.")
