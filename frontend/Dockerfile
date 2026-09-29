# ---- Стадия 1: сборка статики (Vite + React) --------------------------------
ARG NODE_IMAGE=node:20-alpine
ARG NGINX_IMAGE=nginxinc/nginx-unprivileged:1.27-alpine

FROM ${NODE_IMAGE} AS build
WORKDIR /app

# Зависимости npm скачиваются при сборке образа (ей нужен интернет); при запуске сеть не нужна.
# Манифесты копируем отдельно: слой с зависимостями не пересобирается, пока package*.json не менялись.
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund

COPY . .
RUN npm run build

# ---- Стадия 2: раздача через nginx (без root) -------------------------------
FROM ${NGINX_IMAGE} AS runtime

# Адрес backend инференса (микросервис) и DNS-резолвер: по умолчанию имя сервиса из docker-compose
ENV API_UPSTREAM=http://backend:8000 \
    NGINX_RESOLVER=127.0.0.11

# *.template обрабатывается entrypoint-ом образа (envsubst) в /etc/nginx/conf.d/default.conf
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
  CMD wget -q --spider http://127.0.0.1:8080/healthz || exit 1
