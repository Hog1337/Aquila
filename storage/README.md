# Хранилище галереи (SeaweedFS, S3-API)

Здесь лежат сами кадры галереи. Эмбеддинги и метаданные хранятся в векторной СУБД (Qdrant, ТЗ п.6), а в бакете только изображения; ключ объекта = `image_id` из `images/` датасета.

Выбран [SeaweedFS](https://github.com/seaweedfs/seaweedfs): лицензия Apache 2.0, работает без интернета, S3-API совместим с `boto3`, масштабируется до сотен миллионов файлов (задел под ANN-галерею порядка 10^6 объектов, ТЗ п.10).

## Запуск

```bash
docker compose up -d --build --wait   # из каталога storage/ (или из aquila/ вместе с остальными сервисами)
```

`--build` нужен при первом запуске и после правки `seaweedfs-entrypoint.sh`: скрипт вшит в образ (`Dockerfile`), а не монтируется с хоста.

Готовность: контейнер `seaweedfs` в состоянии `healthy`, то есть S3 слушает порт и бакет `gallery` создан. Без `.env` стек стартует с dev-ключами из `docker-compose.yml`: этого достаточно для локальной проверки, для демо-хоста ключи задаются отдельно (см. `.env.example`).

| Что | Значение по умолчанию |
|---|---|
| S3-endpoint снаружи | `http://localhost:8333` (`S3_BIND`, `S3_PORT`) |
| S3-endpoint из сети compose | `http://seaweedfs:8333`, сеть `falcon-reid` |
| Бакет | `gallery` |
| Образ | `chrislusf/seaweedfs:4.47` (`SEAWEEDFS_VERSION`), процесс работает от пользователя `seaweed` (uid 1000), корневая ФС read-only |
| Данные | volume `seaweedfs-data` (переживает `down`, удаляется только `down -v`) |

Две идентичности: `reid-backend` (чтение и запись только в бакет `gallery`, ключ `S3_APP_ACCESS_KEY` отдаём backend-у) и `admin` (все права, ключ `S3_ADMIN_ACCESS_KEY`, нигде в сервисах не используется). Порты master, filer и volume наружу не публикуются.

## Загрузка галереи

```bash
pip install -r scripts/requirements.txt
python scripts/upload_gallery.py ./dataset/images
```

Скрипт идемпотентен (повторный запуск пропускает уже загруженное), параметры берёт из переменных окружения `S3_ENDPOINT`, `S3_BUCKET`, `S3_APP_ACCESS_KEY`, `S3_APP_SECRET_KEY`. Ключ объекта = имя файла, **без** префикса `images/`.

**Согласование с backend.** Backend читает кадры галереи по ключу `images/<image_id>.jpg` (и пишет так же при импорте и «Добавить ТС»), кадры запросов по `queries/<uuid>.jpg`, файлы экспорта по `exports/<job_id>/<файл>`. Кадры, загруженные этим скриптом, лежат в корне бакета и backend их не находит (миниатюры и карты внимания пустые), пока схемы не приведены к одной: `../docs/backend.md`, раздел 8, №4. Имеет ли `image_id` в датасете расширение, по репозиторию неизвестно (в черновых скриптах разработчика `image_id` без расширения, файлы `.jpg`).

## Подключение из backend

```python
import boto3
s3 = boto3.client("s3", endpoint_url="http://seaweedfs:8333",
                  aws_access_key_id=..., aws_secret_access_key=..., region_name="us-east-1")
s3.get_object(Bucket="gallery", Key=f"images/{image_id}.jpg")["Body"].read()   # так читает backend
```

Адресация path-style: `http://seaweedfs:8333/gallery/<ключ>`. Backend подключён к сети `falcon-reid` (`S3_ENDPOINT=http://seaweedfs:8333`, ключ доступа из `storage/.env`).

## Обслуживание

```bash
docker compose logs seaweedfs                                   # логи
docker compose exec seaweedfs sh -c 'echo s3.bucket.list | weed shell -master=127.0.0.1:9333 -filer=127.0.0.1:8888'
docker compose down -v                                          # ВНИМАНИЕ: удалит все загруженные изображения
```

Обновление версии: поменять `SEAWEEDFS_VERSION` в `.env` и выполнить `docker compose up -d --build --wait`. Остановка контейнера занимает около 20 с (сброс томов), не прерывайте её `docker kill`. Формат данных в пределах 4.x совместим, но перед обновлением на демо-хосте сделайте копию volume.

## Запуск отдельно от остальных

Имя проекта compose у всех частей и корня одно (`falcon-reid`): общая сеть `falcon-reid` и тома `falcon-reid_<ключ>` одинаковы при запуске из корня и из этого каталога. Если соседние сервисы уже работают, при запуске отсюда Compose предупредит про orphan-контейнеры: это нормально, **не используйте `--remove-orphans`** (он остановит соседей); предупреждение убирает `COMPOSE_IGNORE_ORPHANS=true`.

## Ограничения

- Ключи только из латиницы и цифр: они подставляются в `s3.json` без экранирования.
- Один контейнер `weed server` (master + volume + filer + S3): этого хватает для демо и галереи в 10^5 – 10^6 файлов на одном узле. Для кластера сервисы разносятся по отдельным контейнерам, схема в документации SeaweedFS.
- Версии для списка зависимостей в Readme решения (ТЗ п.7, п.12): SeaweedFS 4.47, boto3 >=1.34.
