import { useCallback, useEffect, useRef, useState } from 'react'
import { isActive } from './hooks.js'

const PREFIX = 'ldt.v1.'

// Читает значение из localStorage по ключу. Принимает: key — ключ без префикса, fallback — значение по умолчанию.
export function readStored(key, fallback) {
  try {
    const raw = window.localStorage.getItem(PREFIX + key)
    return raw === null ? fallback : JSON.parse(raw)
  } catch {
    return fallback
  }
}

// Записывает значение в localStorage либо удаляет ключ. Принимает: key — ключ без префикса, value — значение (null/undefined удаляет ключ).
export function writeStored(key, value) {
  try {
    if (value === undefined || value === null) window.localStorage.removeItem(PREFIX + key)
    else window.localStorage.setItem(PREFIX + key, JSON.stringify(value))
  } catch {}
}

// Состояние, синхронизированное с хранилищем: читается при инициализации и пишется при каждом изменении. Принимает: key — ключ хранения, initial — значение по умолчанию, ok — функция валидации прочитанного значения.
export function usePersistentState(key, initial, ok = () => true) {
  const [value, setValue] = useState(() => {
    const v = readStored(key, initial)
    return ok(v) ? v : initial
  })
  useEffect(() => { writeStored(key, value) }, [key, value])
  return [value, setValue]
}

// Восстанавливает фоновое задание по сохранённому id и синхронизирует id с хранилищем. Принимает: key — ключ хранения id, job/setJob — состояние задания, fetchJob — функция получения задания по id.
export function useStoredJobId(key, job, setJob, fetchJob) {
  const fetchRef = useRef(fetchJob)
  fetchRef.current = fetchJob
  useEffect(() => {
    const id = readStored(key, null)
    if (id === null) return undefined
    let live = true
    fetchRef.current(id).then(
      (j) => { if (live && isActive(j)) setJob((cur) => cur || j) },
      (e) => { if (live && e.status === 404) writeStored(key, null) },
    )
    return () => { live = false }
  }, [key, setJob])
  useEffect(() => { if (job && job.id !== undefined) writeStored(key, job.id) }, [key, job])
}

// Возвращает функцию с отложенным вызовом (debounce). Принимает: fn — вызываемая функция, ms — задержка в мс.
export function useDebouncedCallback(fn, ms = 200) {
  const fnRef = useRef(fn)
  fnRef.current = fn
  const timer = useRef(null)
  useEffect(() => () => clearTimeout(timer.current), [])
  return useCallback((...args) => {
    clearTimeout(timer.current)
    timer.current = setTimeout(() => fnRef.current(...args), ms)
  }, [ms])
}

const DB = 'ldt-ui'
const STORE = 'frames'

// Открывает соединение с IndexedDB, создавая хранилище при первом запуске. Аргументов не принимает.
function openDb() {
  return new Promise((resolve, reject) => {
    if (!window.indexedDB) { reject(new Error('IndexedDB недоступен')); return }
    const req = window.indexedDB.open(DB, 1)
    req.onupgradeneeded = () => req.result.createObjectStore(STORE)
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

// Выполняет транзакцию над хранилищем файлов и закрывает соединение по завершении. Принимает: mode — режим транзакции ('readonly'/'readwrite'), run — функция, выполняющая операцию над object store.
async function withStore(mode, run) {
  const db = await openDb()
  try {
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE, mode)
      const req = run(tx.objectStore(STORE))
      tx.oncomplete = () => resolve(req.result)
      tx.onerror = () => reject(tx.error)
      tx.onabort = () => reject(tx.error)
    })
  } finally {
    db.close()
  }
}

// Сохраняет файл в IndexedDB по ключу. Принимает: key — ключ записи, file — объект File.
export const saveFile = (key, file) => withStore('readwrite', (s) => s.put({ blob: file, name: file.name || 'frame', type: file.type }, key)).catch(() => {})
// Удаляет сохранённый файл из IndexedDB. Принимает: key — ключ записи.
export const clearFile = (key) => withStore('readwrite', (s) => s.delete(key)).catch(() => {})
// Загружает сохранённый файл из IndexedDB как объект File. Принимает: key — ключ записи.
export const loadFile = (key) => withStore('readonly', (s) => s.get(key)).then(
  (r) => (r && r.blob ? new File([r.blob], r.name, { type: r.type || r.blob.type }) : null),
  () => null,
)
