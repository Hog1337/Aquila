export const PAGES = [
  ['search', 'search', 'Поиск'],
  ['gallery', 'db', 'Галерея'],
  ['metrics', 'chart', 'Метрики и порог'],
  ['history', 'hist', 'История'],
  ['export', 'exp', 'Экспорт'],
]

export const TOP_N = 10
export const TH_MIN = 0.3
export const TH_MAX = 0.95
export const MIN_BOX_PX = 40
export const MAX_UPLOAD_MB = 50
export const IMAGE_TYPES = ['image/jpeg', 'image/png']
export const GALLERY_PAGE = 30
export const HISTORY_PAGE = 20
export const TNR_OPTIONS = [0.85, 0.9, 0.95]
export const TNR_DEFAULT = 0.9

export const ARTIFACTS = [
  ['submission.csv', 'Топ-10 идентификаторов галереи для каждого запроса'],
  ['embeddings.npy', 'Матрица эмбеддингов float32: сначала запросы, затем галерея, в порядке строк исходных CSV'],
  ['candidates.csv', 'Принятые кандидаты с оценками; для отказа строка не добавляется'],
]
