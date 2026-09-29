# Сторонние компоненты (ТЗ п. 7, п. 12)

Сверено с кодом на коммите `60fc04e` (26.09.2026). Почти все компоненты свободные (лицензии Apache-2.0, MIT, BSD, PostgreSQL License и т.п.). **Исключение — веса DINOv3 от Meta**: они распространяются по собственной лицензии «DINOv3 License», а не по стандартной открытой (см. ниже). ТЗ п. 6 отдаёт предпочтение свободному ПО, п. 7 разрешает публичные предобученные веса.

Пометка «неизвестно» означает, что по репозиторию установить нельзя.

## Веса моделей и датасеты

| Источник | Версия | Лицензия | Назначение |
|---|---|---|---|
| DINOv3 ViT-L/16 (`inference/weights/dinov3-vitl16/`, формат `transformers`, `DINOv3ViTModel`), Meta | версия/коммит исходного репозитория весов **неизвестны** (в `config.json`: `transformers_version` 4.56.0.dev0); в метаданных карточки модели (`README.md` весов) указана базовая модель `facebook/dinov3-vit7b16-pretrain-lvd1689m`; как связаны с ней именно эти веса ViT-L/16, по репозиторию неизвестно | **DINOv3 License** (Meta, редакция от 19.08.2025), текст в `inference/weights/dinov3-vitl16/LICENSE.md`; `license: other` в карточке модели | предобученный backbone эмбеддинга |
| Чекпоинт ReID `inference/weights/JDNFV_MASKED_FT_best_fp16.pt` (LoRA rank 32, α=64, блоки 14–23, голова HeadBNNeck 2048d) | тег `JDNFV_MASKED_FT` | собственная разработка команды | дообученная часть модели |
| Датасет, на котором обучен чекпоинт | **Falcon ReID** (организаторы) + **DN-ReID** (CC BY-NC) + **VeRI-Wild** (CC BY-NC) — см. [`training/DATASETS.md`](training/DATASETS.md) | Служебная (организаторов) + CC BY-NC | обучение |

Основные условия DINOv3 License, прочитанные в файле (это не юридическое заключение, полный текст обязателен к прочтению): права предоставлены на использование, копирование, распространение и изменение; при передаче материалов третьим лицам нужно приложить копию лицензии; публикуя результаты исследований с использованием DINO Materials, нужно указать это; запрещены реверс-инжиниринг, использование с нарушением экспортного контроля и в военных целях; лицензия расторгается при нарушении; право Калифорнии. Файл лицензии лежит рядом с весами и сопровождает их в репозитории. Является ли такая лицензия допустимой для целей конкурса (ТЗ п. 6: «предпочтение свободному ПО», п. 7: «открытые» внешние ресурсы), по репозиторию решить нельзя; **вопрос остаётся на усмотрение команды и жюри**.

Суммарный размер весов: 1 212 559 808 Б (`model.safetensors`) + 438 302 047 Б (чекпоинт) = 1 650 861 855 Б (около 1,65 ГБ), меньше ограничения ТЗ п. 7 (2 ГБ). Веса лежат в git через Git LFS (`.gitattributes` корня и каталога весов) и монтируются в контейнер inference, в образ не вшиваются.

Обучающий код — в [`training/`](training/), описание обучения — в [`docs/training.md`](docs/training.md). Для полного воспроизведения требуются внешние датасеты (DN-ReID, VeRI-Wild) и GPU ≥24 ГБ.

## Образы контейнеров

Базовые образы скачиваются при сборке (`docker compose build`), из них собираются образы решения: `postgres`, `qdrant`, `seaweedfs`, `frontend`, `backend`, `inference` и init-образы.

| Образ | Версия | Назначение | Лицензия |
|---|---|---|---|
| `qdrant/qdrant` | v1.19.1 | векторная СУБД | Apache-2.0 |
| `chrislusf/seaweedfs` | 4.47 | S3-хранилище кадров | Apache-2.0 |
| `postgres` | 17 (17.11, вариант alpine) | реляционная СУБД | PostgreSQL License |
| `python` | 3.12 (3.12.14, alpine) | init-сервис Qdrant, вспомогательные операции резервного копирования | PSF License |
| `python` | 3.12 (вариант slim; точный патч-релиз не зафиксирован) | образ backend | PSF License |
| `pytorch/pytorch` | 2.2.0-cuda12.1-cudnn8-runtime | образ inference (PyTorch, CUDA 12.1, cuDNN 8) | PyTorch: BSD-3-Clause; CUDA и cuDNN: собственные лицензии NVIDIA, входят в образ |
| `node` | 20 (20.20.2, alpine; npm 10.8.2) | сборка фронтенда (только на этапе сборки) | MIT |
| `nginxinc/nginx-unprivileged` | 1.27 (1.27.5, alpine) | раздача фронтенда | BSD-2-Clause |

Базовый слой alpine (3.23.4) состоит из пакетов под разными свободными лицензиями (musl: MIT, BusyBox: GPL-2.0, и др.). Образ inference основан на дистрибутиве с менеджером `apt` (пакеты `libgl1-mesa-glx`, `libglib2.0-0`, `curl` ставятся `apt` при сборке; их версии и лицензии не зафиксированы).

## Python-пакеты backend и inference

Файлы `backend/requirements.txt` и `inference/requirements.txt` задают **только нижние границы версий**, lock-файлов нет; точные версии в собранных образах **неизвестны** и зависят от даты сборки (узнать: `pip list` внутри собранного образа). Лицензии указаны по данным проектов.

| Пакет | Требование | Где | Назначение | Лицензия |
|---|---|---|---|---|
| fastapi | >=0.110 | backend, inference | веб-фреймворк, OpenAPI | MIT |
| uvicorn | >=0.29 | backend, inference | ASGI-сервер | BSD-3-Clause |
| python-multipart | >=0.0.9 | backend, inference | разбор multipart | Apache-2.0 |
| asyncpg | >=0.29 | backend | драйвер PostgreSQL | Apache-2.0 |
| qdrant-client | >=1.9 | backend | клиент Qdrant (проверено на 1.19.x по комментарию в коде) | Apache-2.0 |
| boto3 | >=1.34 | backend | клиент S3 | Apache-2.0 |
| httpx | >=0.27 | backend | клиент inference и healthcheck | BSD-3-Clause |
| numpy | >=1.26 | backend, inference | массивы, метрики, экспорт `.npy` | BSD-3-Clause |
| Pillow | >=10.0 | backend, inference | вырезка и кодирование изображений | HPND |
| torch | >=2.2 | inference | нейросеть | BSD-3-Clause |
| torchvision | >=0.18 | inference | преобразования изображений | BSD-3-Clause |
| transformers | >=5.0 | inference | загрузка DINOv3, процессор изображений | Apache-2.0 |
| safetensors | >=0.5 | inference | формат весов | Apache-2.0 |

Совместимость `transformers>=5.0` с torch 2.2 из базового образа и с кодом загрузки DINOv3 (`self.model.model.layer[...]`) по репозиторию не подтверждена; в `config.json` весов указана `transformers_version` 4.56.0.dev0.

## Пакеты фронтенда (npm)

Точные версии всех 112 пакетов зафиксированы в `frontend/package-lock.json`.

| Пакет | Версия | Роль | Лицензия |
|---|---|---|---|
| react | 18.3.1 | UI | MIT |
| react-dom | 18.3.1 | UI | MIT |
| @fontsource/manrope | 5.3.0 | шрифт | OFL-1.1 |
| @fontsource/unbounded | 5.3.0 | шрифт | OFL-1.1 |
| vite | 5.4.21 | сборка (dev) | MIT |
| @vitejs/plugin-react | 4.7.0 | сборка (dev) | MIT |

Лицензии транзитивных зависимостей: MIT (102), ISC (5), OFL-1.1 (2), Apache-2.0 (1), BSD-3-Clause (1), CC-BY-4.0 (1).

## Клиенты для скриптов загрузки

| Библиотека | Требование | Где | Назначение | Лицензия |
|---|---|---|---|---|
| psycopg[binary] | >=3.2,<4 | `postgres/scripts/` | доступ к PostgreSQL | LGPL-3.0 |
| boto3 | >=1.34,<2 | `storage/scripts/` | доступ к S3 (SeaweedFS) | Apache-2.0 |
| qdrant-client | >=1.15,<2 (проверено на 1.19.1) | `vector/scripts/` | доступ к Qdrant | Apache-2.0 |
| numpy | >=1.26,<3 | `vector/scripts/` | чтение `embeddings.npy` | BSD-3-Clause |

Черновые скрипты `scripts/evaluate_api.py` и `scripts/load_dataset.py` (не описаны в инструкции запуска) дополнительно используют `pandas`, `psycopg2`, `httpx`; списка их зависимостей в репозитории нет. Скрипт `evaluate_api.py` вызывает внешний `evaluate.py` (эталонный скрипт организаторов, по пути `/home/limon/...` на машине разработчика); в репозитории его нет.

Backend использует `asyncpg`, а не `psycopg`. Прежняя версия таблицы «Клиенты для backend (рекомендуемые версии)» описывала предполагаемый выбор и заменена таблицей выше.

## Инструменты сборки и запуска

| Инструмент | Версия | Лицензия |
|---|---|---|
| Docker Engine | 24+ | Apache-2.0 |
| Docker Compose | v2.20+ | Apache-2.0 |
| NVIDIA Container Toolkit (для `runtime: nvidia` у inference) | версия не указана | Apache-2.0 |
| Git LFS (получение весов модели) | версия не указана | MIT |
| GNU Bash, coreutils, tar, gzip | из ОС | GPL-3.0 |
| Jenkins (Pipeline, Docker Pipeline), GitLab Container Registry (CI/CD на aquila) | версии не зафиксированы в репозитории | не рассматривались |
