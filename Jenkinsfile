// CI/CD решения «Фалькон ReID» (frontend, backend, inference, Qdrant, SeaweedFS, PostgreSQL).
//
// Всё выполняется на агенте docker-host (Linux, Docker Engine, Compose v2, плагин Docker Pipeline).
// Схема: проверки → сборка образов → smoke-тест на агенте → (только main/master) публикация в GitLab Registry
// → деплой на aquila. На aquila образы тянутся из реестра read-only токеном, сама сборка там не выполняется;
// стек поднимает тот же ./run.sh, что и у разработчика (режим --registry).
//
// Что нужно один раз настроить (в home-infra / на сервере):
//   - credentials 'aquila-deploy' (SSH private key), 'gitlab-registry' (write_registry, для Jenkins)
//     и 'gitlab-registry-pull' (read_registry, им пользуется aquila);
//   - в GitLab включён Container Registry проекта;
//   - пользователь на aquila входит по этому ключу, состоит в группе docker, каталог /opt/falcon-reid доступен ему на запись;
//   - хост aquila резолвится с docker-host (иначе замените DEPLOY_HOST на IP).
// Секреты сервисов (.env) создаются при первом деплое на самом aquila и живут только там: ни в git, ни в Jenkins их нет.
//
// Inference (PyTorch + DINOv3, runtime nvidia) и его веса:
//   - на агенте docker-host GPU и весов нет: образ inference собирается и публикуется, но в smoke-тесте не запускается
//     (SKIP_SERVICES=inference);
//   - веса (inference/weights: dinov3-vitl16-bf16/ и JDNFV_MASKED_FT_best_fp16.pt) лежат в git через LFS, на aquila их не копируем:
//     при деплое каталог inference/weights переносится из предыдущего выката, при первом выкате его кладут вручную
//     в /opt/falcon-reid/stack/inference/weights. Без весов и GPU снимите галку DEPLOY_INFERENCE;
//   - чтобы агент не скачивал ~1.6 ГБ весов на каждом checkout, задайте на узле docker-host GIT_LFS_SKIP_SMUDGE=1.

// Публикация образов в GitLab Registry. names: имена из BUILT_IMAGES через пробел (frontend backend ...).
// Образы пушатся параллельно; -q.
def pushImages(String names) {
    withEnv(["PUSH_NAMES=${names}"]) {
        withCredentials([usernamePassword(credentialsId: 'gitlab-registry',
                                          usernameVariable: 'REG_USER', passwordVariable: 'REG_PASS')]) {
            sh '''
                set -eu
                REG_HOST="${REGISTRY_BASE%%/*}"
                export DOCKER_CONFIG="$(mktemp -d)"
                trap 'rm -rf "$DOCKER_CONFIG"' EXIT
                printf '%s' "$REG_PASS" | docker login "$REG_HOST" -u "$REG_USER" --password-stdin
                pids=""
                for pair in $BUILT_IMAGES; do
                    name="${pair%%=*}"
                    case " $PUSH_NAMES " in *" $name "*) ;; *) continue ;; esac
                    docker tag "${pair#*=}" "${REGISTRY_BASE}/${name}:${TAG}"
                    docker push -q "${REGISTRY_BASE}/${name}:${TAG}" &
                    pids="$pids $!"
                done
                rc=0
                for pid in $pids; do wait "$pid" || rc=1; done
                exit "$rc"
            '''
        }
    }
}

pipeline {
    agent { label 'docker-host' }

    options {
        timestamps()
        ansiColor('xterm')
        timeout(time: 40, unit: 'MINUTES')
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '20', artifactNumToKeepStr: '5'))
    }

    parameters {
        string(name: 'REGISTRY_BASE', defaultValue: 'registry.git.hog1337.com/wallcrepers2/aquila',
               description: 'Префикс образов в GitLab Container Registry без тега: <префикс>/frontend, backend, inference, seaweedfs, postgres, postgres-init, qdrant, qdrant-init')
        string(name: 'FRONTEND_PORT', defaultValue: '8080', description: 'Порт UI на хосте aquila')
        booleanParam(name: 'DEPLOY_INFERENCE', defaultValue: true,
                     description: 'Поднимать inference на aquila. Нужны GPU с NVIDIA Container Toolkit и веса модели в /opt/falcon-reid/stack/inference/weights')
        string(name: 'API_UPSTREAM', defaultValue: 'http://backend:8000',
               description: 'Адрес backend инференса, как его видит контейнер frontend на aquila')
    }

    environment {
        DEPLOY_HOST    = '192.168.20.3'
        DEPLOY_DIR     = '/opt/falcon-reid/stack'
        // Параметры дублируем в environment: при первой сборке после (пере)создания job они доступны
        // в Groovy как params.*, но в shell как переменные окружения не попадают.
        REGISTRY_BASE  = "${params.REGISTRY_BASE ?: ''}"
        FRONTEND_PORT  = "${params.FRONTEND_PORT ?: '8080'}"
        API_UPSTREAM   = "${params.API_UPSTREAM ?: 'http://backend:8000'}"
        // params.DEPLOY_INFERENCE бывает null при первой сборке после создания job: тогда считаем, что включено
        DEPLOY_INFERENCE = "${params.DEPLOY_INFERENCE == false ? 'false' : 'true'}"
        // Образы, которые собирает CI: имя в реестре -> локальный тег из compose
        BUILT_IMAGES   = 'frontend=falcon-reid-frontend:latest backend=falcon-reid-backend:latest inference=falcon-reid-inference:latest seaweedfs=falcon-reid-seaweedfs:4.47 postgres=falcon-reid-postgres:17-alpine postgres-init=falcon-reid-postgres-init:17-alpine qdrant=falcon-reid-qdrant:v1.19.1 qdrant-init=falcon-reid-qdrant-init:v1.19.1'
        BUILDX_NO_DEFAULT_ATTESTATIONS = '1'
    }

    stages {
        stage('Подготовка') {
            steps {
                script {
                    def branch = (env.BRANCH_NAME ?: env.GIT_BRANCH ?: 'local').replaceAll('^origin/', '')
                    env.IS_MAIN = (branch == 'main' || branch == 'master').toString()
                    def safeBranch = branch.replaceAll('[^A-Za-z0-9_.-]', '-').toLowerCase()
                    def sha = (env.GIT_COMMIT ?: 'nogit').take(7)
                    env.TAG = "${safeBranch}-${env.BUILD_NUMBER}-${sha}"
                    env.SMOKE_PROJECT = "ci-falcon-${env.BUILD_NUMBER}"
                    echo "Ветка: ${branch}, тег образов: ${env.TAG}, деплой: ${env.IS_MAIN}"
                }
            }
        }

        stage('Проверки') {
            parallel {
                stage('Compose и образы') {
                    steps {
                        sh '''
                            docker compose config --quiet
                        '''
                    }
                }
                stage('Shellcheck') {
                    agent {
                        docker {
                            image 'koalaman/shellcheck-alpine:stable'
                            reuseNode true
                        }
                    }
                    steps {
                        sh 'shellcheck -x -P SCRIPTDIR run.sh scripts/*.sh postgres/scripts/*.sh storage/seaweedfs-entrypoint.sh'
                    }
                }
            }
        }

        // Inference собирается отдельно
        stage('Сборка образов') {
            steps {
                sh '''
                    docker compose build --pull frontend backend seaweedfs postgres postgres-init qdrant qdrant-init
                    for pair in $BUILT_IMAGES; do docker image ls "${pair#*=}" || true; done
                '''
            }
        }

       
        stage('Smoke-тест, inference и публикация') {
            failFast true
            parallel {
                stage('Smoke-тест на агенте') {
                    environment {
                        COMPOSE_PROJECT_NAME = "${env.SMOKE_PROJECT}"
                        // Порты на хост не публикуем на фиксированные номера (0 = случайный): на агенте они могут быть заняты
                        FRONTEND_PORT = '0'
                        POSTGRES_PORT = '0'
                        QDRANT_HTTP_PORT = '0'
                        QDRANT_GRPC_PORT = '0'
                        S3_PORT = '0'
                        BACKEND_PORT = '0'
                        INFERENCE_PORT = '0'
                        SKIP_SERVICES = 'inference'
                    }
                    steps {
                        sh '''
                            ./run.sh
                            scripts/smoke.sh
                        '''
                    }
                    post {
                        always {
                            sh '''
                                docker compose logs --no-color --tail 80 || true
                                docker compose down -v --remove-orphans || true
                                # Секреты smoke-теста одноразовые
                                rm -f .env postgres/.env vector/.env storage/.env backend/.env
                            '''
                        }
                    }
                }

                stage('Публикация: основные образы') {
                    when {
                        allOf {
                            expression { env.IS_MAIN == 'true' }
                            expression { env.REGISTRY_BASE?.trim() }
                        }
                    }
                    steps {
                        script { pushImages('frontend backend seaweedfs postgres postgres-init qdrant qdrant-init') }
                    }
                }

                stage('Inference') {
                    stages {
                        stage('Сборка inference') {
                            steps {
                                sh '''
                                    docker compose build --pull inference
                                    docker image ls falcon-reid-inference:latest
                                '''
                            }
                        }
                        stage('Публикация inference') {
                            when {
                                allOf {
                                    expression { env.IS_MAIN == 'true' }
                                    expression { env.REGISTRY_BASE?.trim() }
                                }
                            }
                            steps {
                                script { pushImages('inference') }
                            }
                        }
                    }
                }
            }
        }

        stage('Развёртывание на aquila') {
            when {
                allOf {
                    expression { env.IS_MAIN == 'true' }
                    expression { env.REGISTRY_BASE?.trim() }
                }
            }
            steps {
                withCredentials([
                    sshUserPrivateKey(credentialsId: 'aquila-deploy', keyFileVariable: 'DEPLOY_KEY', usernameVariable: 'DEPLOY_USER'),
                    usernamePassword(credentialsId: 'gitlab-registry-pull', usernameVariable: 'PULL_USER', passwordVariable: 'PULL_PASS'),
                    sshUserPrivateKey(credentialsId: 'gitlab-aquila', keyFileVariable: 'GIT_KEY')
                ]) {
                    sh '''
                        set -eu
                        mkdir -p "$WORKSPACE/.cache"
                        rssh() {
                            ssh -i "$DEPLOY_KEY" -o IdentitiesOnly=yes -o BatchMode=yes \
                                -o StrictHostKeyChecking=accept-new \
                                -o UserKnownHostsFile="$WORKSPACE/.cache/known_hosts" \
                                "$DEPLOY_USER@$DEPLOY_HOST" "$@"
                        }
                        REG_HOST="${REGISTRY_BASE%%/*}"

                        
                        rssh "rm -rf '$DEPLOY_DIR.new' && mkdir -p '$DEPLOY_DIR.new'"
                        git archive --format=tar HEAD -- . ':(exclude)inference/weights' | rssh "tar -C '$DEPLOY_DIR.new' -xf -"

                       
                        rssh "DEPLOY_DIR='$DEPLOY_DIR' bash -s" <<'REMOTE'
                            set -euo pipefail
                            for f in .env postgres/.env vector/.env storage/.env; do
                                if [ -f "$DEPLOY_DIR/$f" ]; then cp -p "$DEPLOY_DIR/$f" "$DEPLOY_DIR.new/$f"; fi
                            done
                            if [ -d "$DEPLOY_DIR/backups" ]; then mv "$DEPLOY_DIR/backups" "$DEPLOY_DIR.new/backups"; fi
                            if [ -d "$DEPLOY_DIR/inference/weights" ]; then
                                mkdir -p "$DEPLOY_DIR.new/inference"
                                mv "$DEPLOY_DIR/inference/weights" "$DEPLOY_DIR.new/inference/weights"
                            fi
                            rm -rf "$DEPLOY_DIR.old"
                            if [ -d "$DEPLOY_DIR" ]; then mv "$DEPLOY_DIR" "$DEPLOY_DIR.old"; fi
                            mv "$DEPLOY_DIR.new" "$DEPLOY_DIR"
                            rm -rf "$DEPLOY_DIR.old"
REMOTE

                        SKIP_SERVICES=""
                        if [ "$DEPLOY_INFERENCE" = true ]; then
                            # "путь:ожидаемый_размер" для крупных файлов; размер берём из указателя, он не зависит от smudge на агенте
                            need=""
                            for f in inference/weights/JDNFV_MASKED_FT_best_fp16.pt inference/weights/dinov3-vitl16-bf16/model.safetensors inference/weights/yolo11s-seg.pt; do
                                need="$need $f:$(git cat-file -p "HEAD:$f" | awk '/^size /{print $2}')"
                            done
                            weights_on_aquila() {
                                rssh "cd '$DEPLOY_DIR' && bash -s -- $need" <<'REMOTE'
                                    for spec in "$@"; do
                                        [ "$(stat -c %s "${spec%%:*}" 2>/dev/null || echo 0)" = "${spec##*:}" ] || exit 1
                                    done
REMOTE
                            }
                            if weights_on_aquila; then
                                echo "Веса модели уже на aquila: загрузка пропущена"
                            else
                                echo "Весов модели на aquila нет или размер отличается: забираю из Git LFS и загружаю"
                                command -v git-lfs >/dev/null || { echo "На агенте нет git-lfs: установите его или положите веса на aquila в $DEPLOY_DIR/inference/weights" >&2; exit 1; }
                                # Ключ чтения репозитория (тот же, что у checkout) нужен и для git lfs pull: checkout отдаёт его только себе
                                GIT_SSH_COMMAND="ssh -i $GIT_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$WORKSPACE/.cache/known_hosts_git" \
                                    git lfs pull --include="inference/weights/**"
                                tar -C inference -cf - weights | rssh "mkdir -p '$DEPLOY_DIR/inference' && tar -C '$DEPLOY_DIR/inference' -xf -"
                                weights_on_aquila || { echo "После загрузки размеры весов на aquila не совпали с ожидаемыми" >&2; exit 1; }
                            fi
                        else
                            SKIP_SERVICES="inference"
                            echo "DEPLOY_INFERENCE выключен: inference на aquila не поднимаем"
                        fi

                        # 3. Токен на чтение живёт на aquila только на время выката
                        trap 'rssh "docker logout $REG_HOST" >/dev/null 2>&1 || true' EXIT
                        printf '%s' "$PULL_PASS" | rssh "docker login '$REG_HOST' -u '$PULL_USER' --password-stdin"

                        # 4. Тот же скрипт, что у разработчика: режим --registry берёт образы из реестра по тегу сборки
                        rssh "cd '$DEPLOY_DIR' && REGISTRY_BASE='$REGISTRY_BASE' TAG='$TAG' FRONTEND_PORT='$FRONTEND_PORT' API_UPSTREAM='$API_UPSTREAM' SKIP_SERVICES='$SKIP_SERVICES' ./run.sh --registry"
                        rssh "cd '$DEPLOY_DIR' && FRONTEND_PORT='$FRONTEND_PORT' scripts/smoke.sh"

                        # 5. Старые образы приложения на aquila удаляем, оставляя 3 последних тега каждого образа
                        for pair in $BUILT_IMAGES; do
                            name="${pair%%=*}"
                            rssh "docker images '$REGISTRY_BASE/$name' --format '{{.Repository}}:{{.Tag}}' | tail -n +4 | xargs -r docker rmi || true"
                        done
                    '''
                }
            }
        }
    }

    post {
        always {
            // Освобождаем место на агенте: висячие слои и старые образы решения (кроме 5 последних тегов)
            sh '''
                docker image prune -f || true
                for pair in $BUILT_IMAGES; do
                    name="${pair%%=*}"
                    docker images "${REGISTRY_BASE}/${name}" --format '{{.Repository}}:{{.Tag}}' | tail -n +6 | xargs -r docker rmi || true
                done
            '''
        }
        success  { echo "Готово: ${env.REGISTRY_BASE}/*:${env.TAG}" }
        unstable { echo 'Сборка завершена с предупреждениями.' }
        failure  { echo 'Сборка упала: смотрите лог стадии выше.' }
    }
}
