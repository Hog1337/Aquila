const BASE = '/api/v1'

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message)
    this.status = status
    this.body = body || null
  }
}

// Собирает query-строку из объекта параметров, пропуская пустые значения. Принимает: query — объект параметров запроса.
const qs = (query) => {
  const p = new URLSearchParams()
  Object.entries(query || {}).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== '') p.set(k, v) })
  const s = p.toString()
  return s ? `?${s}` : ''
}

// Извлекает текст ошибки из тела ответа FastAPI. Принимает: body — разобранное JSON-тело ответа.
function detailText(body) {
  if (!body) return ''
  if (typeof body.detail === 'string') return body.detail
  if (Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join('; ')
  return ''
}

// Формирует текст ошибки по ответу сервера. Принимает: res — объект Response.
async function errorMessage(res) {
  try {
    const text = detailText(await res.json())
    if (text) return text
  } catch {}
  return `Ошибка сервера (${res.status})`
}

// Отправляет multipart-форму с отслеживанием прогресса передачи. Принимает: path — путь запроса, form — FormData,
// опции { onProgress, onSent, signal, timeout } — колбэки прогресса/отправки, сигнал отмены и таймаут в мс.
function upload(path, form, { onProgress, onSent, signal, timeout } = {}) {
  return new Promise((resolve, reject) => {
    const abort = () => new DOMException('Загрузка отменена', 'AbortError')
    if (signal && signal.aborted) { reject(abort()); return }
    const xhr = new XMLHttpRequest()
    xhr.open('POST', BASE + path)
    xhr.responseType = 'text'
    if (timeout) xhr.timeout = timeout
    xhr.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded, e.total) }
    xhr.upload.onload = () => { if (onSent) onSent() }
    xhr.onload = () => {
      let body = null
      try { body = xhr.responseText ? JSON.parse(xhr.responseText) : null } catch {}
      if (xhr.status >= 200 && xhr.status < 300) resolve(body)
      else if (xhr.status === 413) reject(new ApiError('Файл слишком большой для сервера', 413))
      else if ([502, 503, 504].includes(xhr.status)) reject(new ApiError(`Сервер не принял запрос (${xhr.status}): backend недоступен или перегружен`, xhr.status))
      else reject(new ApiError(detailText(body) || `Ошибка сервера (${xhr.status})`, xhr.status, body))
    }
    xhr.onerror = () => reject(new ApiError('Нет связи с сервером', 0))
    xhr.ontimeout = () => reject(new ApiError('Сервер не ответил вовремя', 0))
    xhr.onabort = () => reject(abort())
    if (signal) signal.addEventListener('abort', () => xhr.abort(), { once: true })
    xhr.send(form)
  })
}

// Выполняет JSON-запрос к API. Принимает: path — путь запроса, опции { method, query, json, form, signal }.
async function request(path, { method = 'GET', query, json, form, signal } = {}) {
  const init = { method, signal, headers: {} }
  if (json !== undefined) {
    init.body = JSON.stringify(json)
    init.headers['Content-Type'] = 'application/json'
  }
  if (form) init.body = form
  let res
  try {
    res = await fetch(BASE + path + qs(query), init)
  } catch (e) {
    if (e.name === 'AbortError') throw e
    throw new ApiError('Нет связи с сервером', 0)
  }
  if (!res.ok) {
    // Клонируем ответ до того, как прочитать body
    const errRes = res.clone()
    const msg = await errorMessage(res)
    let body = null
    try {
      const parsed = await errRes.json()
      if (parsed && parsed.detail && typeof parsed.detail === 'object') body = parsed.detail
      else body = parsed
    } catch {}
    throw new ApiError(msg, res.status, body)
  }
  if (res.status === 204) return null
  return res.json()
}

export const api = {
  // Возвращает сводку статуса сервиса. Принимает: signal — сигнал отмены запроса.
  status: (signal) => request('/status', { signal }),

  // Возвращает текущий единый порог отказа. Принимает: signal — сигнал отмены запроса.
  getThreshold: (signal) => request('/threshold', { signal }),
  // Сохраняет новое значение порога отказа. Принимает: value — значение порога, reason — причина изменения.
  putThreshold: (value, reason) => request('/threshold', { method: 'PUT', json: { value, reason } }),

  // Запускает поиск по кадру с рамкой. Принимает: file — файл кадра, box — рамка {x,y,w,h} в пикселях кадра,
  // threshold — порог отказа, topN — размер топа кандидатов, rerank — включить переранжирование.
  search(file, box, threshold, topN, rerank) {
    const form = new FormData()
    form.append('image', file)
    form.append('x', box.x)
    form.append('y', box.y)
    form.append('w', box.w)
    form.append('h', box.h)
    form.append('top_n', topN)
    if (threshold !== null && threshold !== undefined) form.append('threshold', threshold)
    form.append('rerank', rerank ? 'true' : 'false')
    return request('/queries', { method: 'POST', form })
  },
  // Возвращает запрос поиска по id. Принимает: id — идентификатор запроса, signal — сигнал отмены.
  getQuery: (id, signal) => request(`/queries/${id}`, { signal }),
  // Возвращает список запросов поиска. Принимает: query — параметры фильтрации/пагинации, signal — сигнал отмены.
  listQueries: (query, signal) => request('/queries', { query, signal }),
  // Возвращает Grad-CAM для кандидата запроса. Принимает: queryId — id запроса, rank — место кандидата, signal — сигнал отмены.
  gradcam: (queryId, rank, signal) => request(`/queries/${queryId}/candidates/${rank}/gradcam`, { signal }),

  // Возвращает список объектов галереи. Принимает: query — параметры фильтрации/пагинации, signal — сигнал отмены.
  listGallery: (query, signal) => request('/gallery/objects', { query, signal }),
  // Добавляет объект в галерею с прогрессом загрузки. Принимает: form — FormData объекта, opts — опции upload().
  addObject: (form, opts) => upload('/gallery/objects', form, opts),
  // Очищает всю галерею.
  clearGallery: () => request('/gallery', { method: 'DELETE' }),

  deleteObject: (objectId) => request(`/gallery/objects/${objectId}`, { method: 'DELETE' }),

  // Проверяет, какие image_id из CSV отсутствуют в хранилище.
  checkImages: (queryCsv, galleryCsv) => {
    const form = new FormData()
    form.append('query_csv', queryCsv)
    if (galleryCsv) form.append('gallery_csv', galleryCsv)
    return request('/gallery/check-images', { method: 'POST', form })
  },
  // Возвращает агрегированный статус обработки для списка image_id.
  getProcessingStatus: (imageIds) =>
    request('/gallery/processing-status', { method: 'POST', json: { image_ids: imageIds } }),

  // Загружает одно изображение с отслеживанием прогресса.
  uploadSingleImage: (imageId, file, bbox, onProgress, vehicleId) => {
    const form = new FormData()
    form.append('image_id', imageId)
    form.append('file', file)
    if (bbox) {
      form.append('x', String(bbox.x))
      form.append('y', String(bbox.y))
      form.append('w', String(bbox.w))
      form.append('h', String(bbox.h))
    }
    if (vehicleId) {
      form.append('vehicle_id', vehicleId)
    }
    return upload('/gallery/upload-image', form, { onProgress })
  },

  // Очищает всю историю метрик.
  clearMetrics: () => request('/metrics/runs', { method: 'DELETE' }),

  // Возвращает последний прогон метрик. Принимает: signal — сигнал отмены.
  latestMetrics: (signal) => request('/metrics/runs/latest', { signal }),
  // Запускает новый прогон метрик с GT CSV. Принимает: queryCsv, galleryCsv, gtCsv — файлы CSV, useRerank — флаг.
  startMetricsRun: (queryCsv, galleryCsv, gtCsv, useRerank) => {
    const form = new FormData()
    form.append('query_csv', queryCsv)
    form.append('gallery_csv', galleryCsv)
    form.append('gt_csv', gtCsv)
    form.append('use_rerank', useRerank ? 'true' : 'false')
    return request('/metrics/runs', { method: 'POST', form })
  },
  // Возвращает прогон метрик по id. Принимает: id — id прогона, signal — сигнал отмены.
  getMetricsRun: (id, signal) => request(`/metrics/runs/${id}`, { signal }),

  // Очищает данные, загруженные для прогона метрик.
  cleanupMetricsRun: (jobId, scope) =>
    request(`/metrics/runs/${jobId}/cleanup?scope=${scope}`, { method: 'POST' }),

  // Запускает формирование экспорта. Принимает: queryCsv — CSV с запросами, threshold — порог, useRerank — реранкинг, galleryCsv — опционально CSV галереи.
  startExport: (queryCsv, threshold, useRerank, galleryCsv) => {
    const form = new FormData()
    form.append('query_csv', queryCsv)
    if (galleryCsv) form.append('gallery_csv', galleryCsv)
    form.append('threshold', threshold)
    form.append('use_rerank', useRerank ? 'true' : 'false')
    return request('/exports', { method: 'POST', form })
  },
  // Возвращает задание экспорта по id. Принимает: id — id экспорта, signal — сигнал отмены.
  getExport: (id, signal) => request(`/exports/${id}`, { signal }),
  // Возвращает список последних экспортов. Принимает: limit — ограничение количества, signal — сигнал отмены.
  listExports: (limit, signal) => request('/exports', { query: { limit }, signal }),
}

export const urls = {
  // Строит URL кропа объекта галереи. Принимает: id — id объекта, size — желаемый размер изображения.
  objectCrop: (id, size) => `${BASE}/objects/${id}/crop${qs({ size })}`,
  // Строит URL полного кадра запроса. Принимает: id — id запроса.
  queryImage: (id) => `${BASE}/queries/${id}/image`,
  // Строит URL кропа запроса. Принимает: id — id запроса, size — желаемый размер изображения.
  queryCrop: (id, size) => `${BASE}/queries/${id}/crop${qs({ size })}`,
  // Строит URL CSV с кандидатами запроса. Принимает: id — id запроса, threshold — порог отказа.
  candidatesCsv: (id, threshold) => `${BASE}/queries/${id}/candidates.csv${qs({ threshold })}`,
  // Строит URL CSV истории запросов. Принимает: refused — включать ли только отказы.
  historyCsv: (refused) => `${BASE}/queries/history.csv${qs({ refused: refused ? 'true' : undefined })}`,
  // Строит URL файла экспорта. Принимает: id — id экспорта, name — имя файла.
  exportFile: (id, name) => `${BASE}/exports/${id}/files/${encodeURIComponent(name)}`,
}

// Скачивает файл по URL через временную ссылку. Принимает: url — адрес файла.
export function download(url) {
  const a = document.createElement('a')
  a.href = url
  a.download = ''
  document.body.appendChild(a)
  a.click()
  a.remove()
}
