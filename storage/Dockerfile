# Тонкая надстройка над официальным образом SeaweedFS: добавляет скрипт запуска (генерация s3.json, создание бакета).
# Скрипт вшит в образ, а не монтируется с хоста: так стек не зависит от путей и прав на хосте (в том числе при деплое из CI/CD).
ARG SEAWEEDFS_IMAGE=chrislusf/seaweedfs:4.47
FROM ${SEAWEEDFS_IMAGE}

COPY --chmod=0555 seaweedfs-entrypoint.sh /seaweedfs-entrypoint.sh

ENTRYPOINT ["/bin/sh", "/seaweedfs-entrypoint.sh"]
