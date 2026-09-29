import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api.js'
import { isActive } from './hooks.js'

const UploadsContext = createContext(null)
// Возвращает текущее значение контекста загрузок. Аргументов не принимает.
export const useUploads = () => useContext(UploadsContext)

// Проверяет, находится ли задание в активной фазе (передача, ожидание папки или обработка). Принимает: t — объект задания.
export const isBusy = (t) => t.phase === 'uploading' || t.phase === 'awaiting' || t.phase === 'processing'

const POLL_MS = 1000
const PROGRESS_MS = 200
const FLUSH_MS = 250
const CONCURRENCY = 4
const RETRIES = 3
// Вычисляет таймаут передачи одного кадра по его размеру. Принимает: size — размер файла в байтах.
const IMAGE_TIMEOUT_MS = (size) => 20000 + size / 20
const MAX_CONSECUTIVE_FAILS = 6
const AUTO_HIDE_MS = 6000
const VEKT_POLL_MS = 1000
const JOB_PHASE = { queued: 'processing', running: 'processing', done: 'done', failed: 'failed', cancelled: 'cancelled' }
const IMG_RE = /\.(jpe?g|png)$/i

let seq = 0
// Возвращает промис, разрешающийся через заданное время. Принимает: ms — задержка в миллисекундах.
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// Сопоставляет файлы папки с image_id из CSV по имени файла или имени без расширения, без учёта регистра. Принимает: imageIds — список нужных id, files — файлы выбранной папки.
function matchFolder(imageIds, files) {
  const maps = { name: new Map(), stem: new Map(), lname: new Map(), lstem: new Map() }
  let folderName = ''
  for (const f of files) {
    if (!folderName && f.webkitRelativePath) folderName = f.webkitRelativePath.split('/')[0]
    if (!IMG_RE.test(f.name)) continue
    const stem = f.name.replace(/\.[^.]+$/, '')
    const put = (m, k) => { if (!m.has(k)) m.set(k, f) }
    put(maps.name, f.name); put(maps.stem, stem); put(maps.lname, f.name.toLowerCase()); put(maps.lstem, stem.toLowerCase())
  }
  const queue = []
  const missing = []
  for (const id of imageIds) {
    const low = id.toLowerCase()
    const f = maps.name.get(id) || maps.stem.get(id) || maps.lname.get(low) || maps.lstem.get(low)
    if (f) queue.push({ image_id: id, file: f })
    else missing.push(id)
  }
  return { queue, missing, bytes: queue.reduce((s, q) => s + q.file.size, 0), name: folderName || 'папка' }
}

// Провайдер контекста загрузок в галерею: хранит и выполняет задания импорта и добавления объектов. Принимает: children — дочерние элементы.
export function UploadProvider({ children }) {
  const [tasks, setTasks] = useState([])
  const [doneTick, setDoneTick] = useState(0)
  const [expandTick, setExpandTick] = useState(0)
  const tasksRef = useRef(tasks)
  tasksRef.current = tasks
  const ctl = useRef(new Map())
  const inflight = useRef(new Set())

  // Обновляет поля задания по id. Принимает: id — id задания, p — объект патча либо функция (задание) => патч.
  const patch = useCallback((id, p) => {
    setTasks((all) => all.map((t) => (t.id === id ? { ...t, ...(typeof p === 'function' ? p(t) : p) } : t)))
  }, [])
  // Удаляет задание из состояния и его служебные данные. Принимает: id — id задания.
  const remove = useCallback((id) => {
    ctl.current.delete(id)
    setTasks((all) => all.filter((t) => t.id !== id))
  }, [])
  // Переводит задание в состояние ошибки. Принимает: id — id задания, failedAt — где произошла ошибка, error — текст ошибки, errorStatus — HTTP-статус.
  const fail = useCallback((id, failedAt, error, errorStatus = 0) => patch(id, { phase: 'failed', failedAt, error, errorStatus, speed: 0 }), [patch])

  // Переносит состояние задания импорта с сервера в фазу задания в трее. Принимает: id — id задания, job — задание импорта с сервера.
  const applyJob = useCallback((id, job) => {
    const phase = JOB_PHASE[job.status] || 'processing'
    const prev = tasksRef.current.find((t) => t.id === id)
    if (phase === 'done' && prev && prev.phase !== 'done') setDoneTick((n) => n + 1)
    patch(id, (t) => ({ job, phase, error: phase === 'failed' ? job.error || 'Импорт завершился ошибкой' : '', failedAt: phase === 'failed' ? 'processing' : t.failedAt }))
  }, [patch])

  // Строит опции прогресса для передачи одного запроса (CSV или кадр). Принимает: id — id задания, c — служебное состояние задания, onSent — колбэк по завершении отправки тела запроса.
  const progressOpts = useCallback((id, c, onSent) => {
    c.abort = new AbortController()
    c.stat = { t: performance.now(), loaded: 0, total: 0, speed: 0, at: 0 }
    return {
      signal: c.abort.signal,
      onProgress: (loaded, total) => {
        const now = performance.now()
        const s = c.stat
        if (now - s.at < PROGRESS_MS && loaded < total) return
        const dt = (now - s.t) / 1000
        if (dt > 0) {
          const inst = (loaded - s.loaded) / dt
          s.speed = s.speed ? s.speed * 0.7 + inst * 0.3 : inst
        }
        s.t = now; s.loaded = loaded; s.total = total; s.at = now
        patch(id, { loaded, total, speed: s.speed })
      },
      onSent: onSent && (() => {
        if (c.stat.total && c.stat.loaded < c.stat.total) return
        onSent()
      }),
    }
  }, [patch])

  // Переводит задание в ошибку передачи, различая обрыв на отправке и сбой уже после отправки тела. Принимает: id — id задания, c — служебное состояние задания, e — ошибка, atFull — стадия при полностью отправленном теле, atPartial — стадия при обрыве передачи.
  const failedUpload = useCallback((id, c, e, atFull, atPartial = atFull) => {
    if (e.name === 'AbortError') { patch(id, { phase: 'cancelled', speed: 0 }); return }
    const { loaded, total } = c.stat
    const sentAll = total > 0 && loaded >= total
    patch(id, { phase: 'failed', error: e.message, errorStatus: e.status || 0, failedAt: sentAll ? atFull : atPartial, speed: 0, ...(sentAll ? {} : { loaded, total }) })
  }, [patch])

  // Выполняет загрузку одного объекта галереи (кадр с рамкой). Принимает: id — id задания.
  const runObject = useCallback(async (id) => {
    const c = ctl.current.get(id)
    if (!c) return
    patch(id, { phase: 'uploading', loaded: 0, total: 0, speed: 0, error: '', failedAt: null })
    const opts = progressOpts(id, c, () => patch(id, (t) => ({ phase: 'processing', loaded: t.total || t.loaded, speed: 0 })))
    try {
      await api.addObject(c.form, opts)
      patch(id, { phase: 'done' })
      setDoneTick((n) => n + 1)
    } catch (e) {
      failedUpload(id, c, e, 'processing', 'uploading')
    }
  }, [patch, progressOpts, failedUpload])

  // Отправляет CSV и получает маппинг image_id → bbox. Принимает: id — id задания.
  const runCheck = useCallback(async (id) => {
    const c = ctl.current.get(id)
    if (!c) return
    patch(id, { phase: 'uploading', stage: 'csv', loaded: 0, total: 0, speed: 0, error: '', failedAt: null })
    try {
      const result = await api.checkImages(c.csvFile, null)
      c.imageIds = Object.keys(result.all_bbox || {})
      c.bboxMap = result.all_bbox || {}
      c.vehicleIds = result.all_vehicle_ids || {}
      
      if (result.all_present && result.with_vectors === result.total) {
        // Все фото уже есть и завекторизованы — импорт завершён
        patch(id, { phase: 'done', doneTick: Date.now() })
        setDoneTick((n) => n + 1)
      } else {
        // Нужны недостающие фото — ждём папку
        patch(id, { phase: 'awaiting', stage: 'images', speed: 0, 
          session: { id: null, total: result.total, images: c.imageIds.length, bboxMap: c.bboxMap }, 
          folder: null, folderError: '' })
        setExpandTick((n) => n + 1)
      }
    } catch (e) {
      failedUpload(id, c, e, 'csv')
    }
  }, [patch, setDoneTick, failedUpload])

  // Передаёт кадры импорта на сервер по одному в несколько потоков и отслеживает векторизацию.
  // Векторизация начинается сразу после сохранения кадра на S3 (на бэке) и идёт параллельно с загрузкой.
  // Принимает: id — id задания.
  const runImages = useCallback(async (id) => {
    const c = ctl.current.get(id)
    if (!c || !c.queue) return
    c.abort = new AbortController()
    const { signal } = c.abort
    const allImageIds = c.queue.map((q) => q.image_id)
    const bytes = c.queue.reduce((s, q) => s + q.file.size, 0)
    let doneBytes = c.queue.filter((q) => c.done.has(q.image_id)).reduce((s, q) => s + q.file.size, 0)
    const pending = c.queue.filter((q) => !c.done.has(q.image_id))
    const live = new Map()
    const flying = new Set()
    const failures = []
    let consecutive = 0
    let stop = false
    let uploadDone = false
    let vectDone = false
    patch(id, { phase: 'uploading', stage: 'images', error: '', failedAt: null, speed: 0, vect: null, imgs: { done: c.done.size, count: c.queue.length, loaded: doneBytes, bytes, speed: 0, failed: 0, flying: [] } })

    const sample = { t: performance.now(), loaded: doneBytes, speed: 0 }
    const flush = () => {
      let loaded = doneBytes
      live.forEach((v) => { loaded += v })
      const now = performance.now()
      const dt = (now - sample.t) / 1000
      if (dt > 0) {
        const inst = (loaded - sample.loaded) / dt
        sample.speed = sample.speed ? sample.speed * 0.7 + inst * 0.3 : inst
      }
      sample.t = now; sample.loaded = loaded
      loaded = Math.min(loaded, bytes)
      patch(id, { imgs: { done: c.done.size, count: c.queue.length, loaded, bytes, speed: sample.speed, failed: failures.length, flying: [...flying].slice(0, 3) } })
    }
    const timer = setInterval(flush, FLUSH_MS)

    // Поллинг статуса векторизации параллельно с загрузкой
    const pollVect = (async () => {
      while (!signal.aborted) {
        await sleep(VEKT_POLL_MS)
        if (signal.aborted || vectDone) return
        try {
          const status = await api.getProcessingStatus(allImageIds)
          patch(id, { vect: status })
          if (uploadDone && status.ready + status.failed >= status.total) {
            vectDone = true
            return
          }
        } catch (e) {
          // ошибка поллинга — игнорируем
        }
      }
    })()

    let next = 0
    const worker = async () => {
      while (!signal.aborted && !stop) {
        const item = pending[next++]
        if (!item) return
        let ok = false
        let lastErr = null
        for (let attempt = 0; attempt < RETRIES && !signal.aborted; attempt++) {
          try {
            flying.add(item.image_id)
            const bbox = (c.bboxMap || {})[item.image_id] || null
            const vid = (c.vehicleIds || {})[item.image_id] || null
            await api.uploadSingleImage(item.image_id, item.file, bbox, (l) => live.set(item.image_id, l), vid)
            ok = true
            break
          } catch (e) {
            if (e.name === 'AbortError') return
            lastErr = e
            live.delete(item.image_id)
            if (e.status >= 400 && e.status < 500 && e.status !== 408 && e.status !== 429) break
            if (attempt < RETRIES - 1) await sleep(700 * (attempt + 1))
          }
        }
        live.delete(item.image_id)
        flying.delete(item.image_id)
        if (ok) {
          c.done.add(item.image_id)
          doneBytes += item.file.size
          consecutive = 0
        } else if (!signal.aborted) {
          failures.push({ image_id: item.image_id, message: lastErr ? lastErr.message : 'ошибка' })
          if (!(lastErr && lastErr.status >= 400 && lastErr.status < 500)) consecutive += 1
          if (consecutive >= MAX_CONSECUTIVE_FAILS) stop = true
        }
      }
    }
    await Promise.all(Array.from({ length: CONCURRENCY }, worker))
    clearInterval(timer)
    if (signal.aborted) return
    flush()
    uploadDone = true

    if (failures.length) {
      const head = stop ? 'Загрузка остановлена: сервер не отвечает.' : `Не загружено кадров: ${failures.length}.`
      fail(id, 'images', `${head} ${failures[0].message}`)
      return
    }

    // Все кадры загружены — ждём завершения векторизации
    if (!vectDone) {
      patch(id, { phase: 'processing', stage: 'vect' })
      while (!signal.aborted && !vectDone) {
        await sleep(VEKT_POLL_MS)
        if (signal.aborted) return
        try {
          const status = await api.getProcessingStatus(allImageIds)
          patch(id, { vect: status })
          if (status.ready + status.failed >= status.total) {
            vectDone = true
          }
        } catch (e) {}
      }
    }

    if (signal.aborted) return

    // Финальный статус для подстраховки
    const finalStatus = await api.getProcessingStatus(allImageIds)
    const errMsg = finalStatus.failed > 0 ? `${finalStatus.failed} из ${finalStatus.total} с ошибкой` : undefined
    patch(id, { phase: 'done', error: errMsg || '', vect: finalStatus, doneTick: Date.now() })
    setDoneTick((n) => n + 1)
  }, [patch, setDoneTick])

  // Создаёт новое задание и запускает его выполнение. Принимает: task — начальные поля задания, form — FormData запроса, run — функция выполнения задания по id.
  const enqueue = useCallback((task, form, run) => {
    const id = ++seq
    ctl.current.set(id, { abort: null, form, stat: null, csvFile: task.csvFile || null })
    const full = { loaded: 0, total: 0, speed: 0, job: null, error: '', failedAt: null, ...task, id, phase: 'uploading' }
    setTasks((all) => [...all, full])
    tasksRef.current = [...tasksRef.current, full]
    setExpandTick((n) => n + 1)
    run(id)
    return id
  }, [])

  // Запускает импорт по CSV с аннотациями. Принимает: csv — файл CSV, title — заголовок задания в трее.
  // Запускает импорт по CSV с аннотациями. Принимает: csv — файл CSV, title — заголовок задания в трее.
  const startImport = useCallback((csv, title = 'Импорт CSV + папка') => {
    const form = new FormData()
    form.append('query_csv', csv)
    enqueue({ kind: 'import', title, stage: 'csv', files: [{ name: csv.name, size: csv.size }], session: null, folder: null, imgs: null, csvFile: csv }, form, runCheck)
  }, [enqueue, runCheck])

  // Запускает добавление одного объекта в галерею. Принимает: form — FormData объекта, file — файл кадра, vehicleId — идентификатор ТС для заголовка задания.
  const addObject = useCallback((form, file, vehicleId) => {
    enqueue({ kind: 'object', title: vehicleId ? `Добавление ТС ${vehicleId}` : 'Добавление ТС', files: [{ name: file.name || 'кадр', size: file.size }] }, form, runObject)
  }, [enqueue, runObject])

  // Сопоставляет выбранную папку с кадрами со списком CSV и запускает передачу, если совпадения полные. Принимает: id — id задания, files — файлы выбранной папки.
  const chooseFolder = useCallback((id, files) => {
    const c = ctl.current.get(id)
    const t = tasksRef.current.find((x) => x.id === id)
    if (!c || !t || t.phase !== 'awaiting' || !c.imageIds) return
    const m = matchFolder(c.imageIds, files)
    if (!m.queue.length) {
      const example = c.imageIds[0]
      patch(id, { folder: null, folderError: `В папке «${m.name}» нет кадров из CSV. Ищем файлы вида ${example}.jpg или ${example}.png (проверено файлов: ${files.length})` })
      return
    }
    c.queue = m.queue
    c.done = new Set()
    patch(id, { folderError: '', folder: { name: m.name, matched: m.queue.length, missing: m.missing.length, missingSample: m.missing.slice(0, 3), bytes: m.bytes } })
    if (!m.missing.length) runImages(id)
  }, [patch, runImages])

  // Подтверждает передачу папки при неполном совпадении с CSV. Принимает: id — id задания.
  const confirmFolder = useCallback((id) => runImages(id), [runImages])

  // Отменяет задание в зависимости от его текущей фазы. Принимает: id — id задания.
  const cancel = useCallback(async (id) => {
    const t = tasksRef.current.find((x) => x.id === id)
    if (!t) return
    const c = ctl.current.get(id)
    if (t.phase === 'awaiting' || (t.kind === 'import' && ['images', 'vect'].includes(t.stage) && (t.phase === 'uploading' || t.phase === 'processing'))) {
      if (c && c.abort) c.abort.abort()
      patch(id, { phase: 'cancelled', speed: 0 })
      return
    }
    if (t.phase === 'uploading') { if (c && c.abort) c.abort.abort(); return }
    if (t.kind === 'import' && t.job) {
      try { applyJob(id, await api.cancelImport(t.job.id)) } catch (e) { patch(id, { error: e.message }) }
    }
  }, [applyJob, patch])

  // Повторяет неудавшийся шаг задания. Принимает: id — id задания.
  const retry = useCallback((id) => {
    const t = tasksRef.current.find((x) => x.id === id)
    if (!t || !ctl.current.has(id)) return
    if (t.kind === 'object') runObject(id)
    else if (t.failedAt === 'csv') runCheck(id)
    else if (t.failedAt === 'images') runImages(id)
  }, [runObject, runCheck, runImages])

  // Скрывает задание из трея. Принимает: id — id задания.
  const dismiss = useCallback((id) => remove(id), [remove])
  // Скрывает все завершённые задания. Аргументов не принимает.
  const dismissFinished = useCallback(() => {
    setTasks((all) => all.filter(isBusy))
  }, [])

  const hideKey = tasks.filter((t) => t.phase === 'cancelled' || (t.kind === 'object' && t.phase === 'done')).map((t) => t.id).join(',')
  useEffect(() => {
    if (!hideKey) return undefined
    const timer = setTimeout(() => hideKey.split(',').forEach((id) => remove(Number(id))), AUTO_HIDE_MS)
    return () => clearTimeout(timer)
  }, [hideKey, remove])

  const pollKey = tasks.filter((t) => t.kind === 'import' && t.phase === 'processing' && t.job && isActive(t.job)).map((t) => t.id).join(',')
  useEffect(() => {
    if (!pollKey) return undefined
    const timer = setInterval(() => {
      tasksRef.current.forEach((t) => {
        if (t.kind !== 'import' || t.phase !== 'processing' || !t.job || !isActive(t.job) || inflight.current.has(t.id)) return
        inflight.current.add(t.id)
        api.getImport(t.job.id).then(
          (job) => applyJob(t.id, job),
          (e) => patch(t.id, { phase: 'failed', error: e.message, failedAt: 'processing' }),
        ).finally(() => inflight.current.delete(t.id))
      })
    }, POLL_MS)
    return () => clearInterval(timer)
  }, [pollKey, applyJob, patch])

  useEffect(() => {
    // Восстановление не требуется — импорт через upload-image не создаёт заданий на сервере
    return () => { live = false }
  }, [])

  const uploading = tasks.some((t) => t.phase === 'uploading')
  useEffect(() => {
    if (!uploading) return undefined
    const guard = (e) => { e.preventDefault(); e.returnValue = '' }
    window.addEventListener('beforeunload', guard)
    return () => window.removeEventListener('beforeunload', guard)
  }, [uploading])

  const value = useMemo(() => ({
    tasks, doneTick, expandTick,
    importActive: tasks.some((t) => t.kind === 'import' && isBusy(t)),
    startImport, addObject, chooseFolder, confirmFolder, cancel, retry, dismiss, dismissFinished,
  }), [tasks, doneTick, expandTick, startImport, addObject, chooseFolder, confirmFolder, cancel, retry, dismiss, dismissFinished])

  return <UploadsContext.Provider value={value}>{children}</UploadsContext.Provider>
}
