import hashlib
import time

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from .. import config


class Storage:
    def __init__(self):
        """Создаёт S3-клиент для доступа к хранилищу. Принимает: без параметров."""
        self.client = boto3.client(
            "s3",
            endpoint_url=config.S3_ENDPOINT,
            aws_access_key_id=config.S3_ACCESS_KEY,
            aws_secret_access_key=config.S3_SECRET_KEY,
            config=Config(signature_version="s3v4", connect_timeout=5, read_timeout=30, retries={"max_attempts": 2}),
            region_name="us-east-1",
        )
        self.bucket = config.S3_BUCKET

    def upload_image(self, key: str, data: bytes):
        """Загружает файл в хранилище и проверяет ETag на совпадение с MD5 тела.
        Принимает: key — ключ объекта, data — байты файла."""
        resp = self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
        etag = (resp.get("ETag") or "").strip('"')
        if len(etag) == 32 and etag != hashlib.md5(data, usedforsecurity=False).hexdigest():
            raise OSError(f"Хранилище записало {key} с другой контрольной суммой")

    def put(self, key: str, data: bytes):
        """Загружает служебный файл в хранилище без проверки ETag. Принимает: key — ключ объекта, data — байты файла."""
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get(self, key: str) -> bytes | None:
        """Читает файл из хранилища. Принимает: key — ключ объекта."""
        return self.get_image(key)

    def get_image(self, key: str, attempts: int = 3) -> bytes | None:
        """Читает файл из хранилища с повторными попытками при сбоях соединения.
        Принимает: key — ключ объекта, attempts — количество попыток чтения."""
        for i in range(attempts):
            try:
                resp = self.client.get_object(Bucket=self.bucket, Key=key)
                return resp["Body"].read()
            except self.client.exceptions.NoSuchKey:
                return None
            except (BotoCoreError, ClientError, OSError) as e:
                status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 500) if isinstance(e, ClientError) else 500
                if i == attempts - 1 or status < 500:
                    raise
                time.sleep(0.5 * (i + 1))

    def delete_image(self, key: str):
        """Удаляет файл из хранилища, игнорируя ошибки. Принимает: key — ключ объекта."""
        try:
            self.client.delete_object(Bucket=self.bucket, Key=key)
        except Exception:
            pass


storage = Storage()
