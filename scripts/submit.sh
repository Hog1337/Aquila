#!/bin/sh
# Пакетный прогон решения на закрытом тесте (ТЗ п.8) одной командой:
#
#   scripts/submit.sh --input /path/to/test --output /path/to/out [options]
#
# --input должен содержать images/ (плоский каталог кадров) и CSV галереи/запросов
# (по умолчанию test_gallery.csv / test_query.csv).
#
# Если стек уже запущен — использует его (без пересборки). Если нет — поднимает через ./run.sh.
# По умолчанию стек и данные сохраняются после завершения.
#
# Флаги:
#   --gt-csv PATH         — ground truth для валидации после экспорта
#   --clear-before        — очистить volumes перед импортом (старт с чистого листа)
#   --clear-after         — удалить импортированные данные из галереи после завершения
#                          (стек остаётся запущенным)
#   --rerank              — использовать Query Expansion
#   --threshold N         — порог отказа (по умолчанию 0.87)
#   --gallery-csv NAME    — имя CSV галереи внутри --input (по умолч. test_gallery.csv)
#   --query-csv NAME      — имя CSV запросов внутри --input (по умолч. test_query.csv)
set -eu

INPUT=""
OUTPUT=""
EXTRA_ARGS=""
GT_CSV=""
CLEAR_BEFORE=0
CLEAR_AFTER=0

while [ $# -gt 0 ]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --gt-csv) GT_CSV="$2"; shift 2 ;;
    --clear-before) CLEAR_BEFORE=1; shift ;;
    --clear-after) CLEAR_AFTER=1; shift ;;
    --threshold) EXTRA_ARGS="$EXTRA_ARGS --threshold $2"; shift 2 ;;
    --gallery-csv) EXTRA_ARGS="$EXTRA_ARGS --gallery-csv $2"; shift 2 ;;
    --query-csv) EXTRA_ARGS="$EXTRA_ARGS --query-csv $2"; shift 2 ;;
    --rerank) EXTRA_ARGS="$EXTRA_ARGS --rerank"; shift ;;
    *) echo "Неизвестный аргумент: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$INPUT" ] || [ -z "$OUTPUT" ]; then
  echo "Использование: $0 --input <каталог с images/+CSV> --output <каталог для результата>" >&2
  echo ""
  echo "Опции:"
  echo "  --gt-csv PATH       ground truth CSV (image_id,vehicle_id,camera_id,split)"
  echo "  --clear-before      очистить volumes перед импортом"
  echo "  --clear-after       удалить импортированные данные из галереи после завершения"
  echo "                       (стек остаётся запущенным)"
  echo "  --rerank            использовать Query Expansion"
  echo "  --threshold N       порог отказа (по умолчанию 0.87)"
  echo "  --gallery-csv NAME  CSV галереи (по умолч. test_gallery.csv)"
  echo "  --query-csv NAME    CSV запросов (по умолч. test_query.csv)"
  exit 1
fi
if [ ! -d "$INPUT" ]; then
  echo "Не найден каталог --input: $INPUT" >&2
  exit 1
fi
mkdir -p "$OUTPUT"
# Гостевой доступ: контейнер может бегать от root или другого пользователя
chmod 777 "$OUTPUT"

# Абсолютные пути
INPUT_ABS=$(cd "$INPUT" && pwd)
OUTPUT_ABS=$(cd "$OUTPUT" && pwd)
GT_ABS=""
if [ -n "$GT_CSV" ]; then
  GT_ABS=$(cd "$(dirname "$GT_CSV")" && pwd)/$(basename "$GT_CSV")
  if [ ! -f "$GT_ABS" ]; then
    echo "GT CSV не найден: $GT_ABS" >&2
    exit 1
  fi
fi

# MSYS на Windows
if command -v cygpath >/dev/null 2>&1; then
  INPUT_ABS=$(cygpath -w "$INPUT_ABS")
  OUTPUT_ABS=$(cygpath -w "$OUTPUT_ABS")
  [ -n "$GT_ABS" ] && GT_ABS=$(cygpath -w "$GT_ABS")
  export MSYS_NO_PATHCONV=1
fi

cd "$(dirname "$0")/.."   # корень решения

# Очистка перед запуском (--clear-before)
if [ "$CLEAR_BEFORE" = 1 ]; then
  echo "[submit] Очищаю volumes (docker compose down -v)..."
  docker compose down -v 2>/dev/null || true
fi

# Очистка после завершения (--clear-after): удалить импортированные данные, стек не трогать
cleanup() {
  if [ "$CLEAR_AFTER" = 1 ]; then
    echo "[submit] Очищаю импортированные данные из галереи (python -m app.cli clear)..."
    docker compose run --rm backend python -m app.cli clear 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Поднимаем стек, только если ещё не запущен
if docker compose ps -q 2>/dev/null | grep -q .; then
  echo "[submit] Стек уже запущен. Использую текущий."
else
  echo "[submit] Стек не запущен. Поднимаю (./run.sh)..."
  ./run.sh
fi

echo "[submit] Запускаю python -m app.cli submit --input /input --output /output $EXTRA_ARGS"
# shellcheck disable=SC2086
docker compose run --rm \
  -v "$INPUT_ABS":/input:ro \
  -v "$OUTPUT_ABS":/output \
  backend python -m app.cli submit --input /input --output /output $EXTRA_ARGS

# Оценка качества, если указан GT
if [ -n "$GT_ABS" ]; then
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  ОЦЕНКА КАЧЕСТВА (--gt-csv)"
  echo "═══════════════════════════════════════════════════════════════"
  # Парсим имена CSV из EXTRA_ARGS (если заданы)
  GALLERY_CSV="test_gallery.csv"
  QUERY_CSV="test_query.csv"
  for arg in $EXTRA_ARGS; do
    case "$arg" in
      --gallery-csv=*) GALLERY_CSV="${arg#*=}" ;;
      --query-csv=*) QUERY_CSV="${arg#*=}" ;;
    esac
  done

  docker compose run --rm \
    -v "$OUTPUT_ABS":/output \
    -v "$GT_ABS":/gt.csv:ro \
    -v "$INPUT_ABS":/input:ro \
    backend python -m app.cli evaluate \
      --gt /gt.csv \
      --submission /output/submission.csv \
      --candidates /output/candidates.csv \
      --embeddings /output/embeddings.npy \
      --query "/input/$QUERY_CSV" \
      --gallery "/input/$GALLERY_CSV" \
      --json /output/evaluation_report.json
  echo "═══════════════════════════════════════════════════════════════"
fi

echo "[submit] Готово. Результат в $OUTPUT_ABS"
