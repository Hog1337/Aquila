import { useEffect, useRef, useState } from 'react'

// Загружает данные из асинхронной функции, сохраняя старые данные во время повторной загрузки. Принимает: fn — асинхронная функция(signal), deps — массив зависимостей для перезапуска.
export function useAsync(fn, deps) {
  const [state, setState] = useState({ data: null, error: null, loading: true })
  const [tick, setTick] = useState(0)
  useEffect(() => {
    const ac = new AbortController()
    let live = true
    setState((s) => ({ ...s, loading: true, error: null }))
    fn(ac.signal).then(
      (data) => { if (live) setState({ data, error: null, loading: false }) },
      (error) => { if (live && error.name !== 'AbortError') setState({ data: null, error, loading: false }) },
    )
    return () => { live = false; ac.abort() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick])
  return { ...state, reload: () => setTick((t) => t + 1) }
}

// Вызывает переданные функции перезагрузки при каждом возвращении экрана в активное состояние. Принимает: active — активен ли экран сейчас, reloads — функции перезагрузки.
export function useReloadOnActive(active, ...reloads) {
  const was = useRef(active)
  const fns = useRef(reloads)
  fns.current = reloads
  useEffect(() => {
    if (active && !was.current) fns.current.forEach((r) => r())
    was.current = active
  }, [active])
}

const ACTIVE = ['queued', 'running']
// Проверяет, выполняется ли фоновое задание сейчас. Принимает: job — объект задания.
export const isActive = (job) => !!job && ACTIVE.includes(job.status)

// Опрашивает состояние фонового задания раз в интервал, пока оно активно. Принимает: fetchJob — функция получения задания по id, interval — период опроса в мс.
export function useJob(fetchJob, interval = 1000) {
  const [job, setJob] = useState(null)
  const fetchRef = useRef(fetchJob)
  fetchRef.current = fetchJob
  useEffect(() => {
    if (!isActive(job)) return undefined
    const t = setTimeout(async () => {
      try {
        setJob(await fetchRef.current(job.id))
      } catch (e) {
        setJob((j) => ({ ...j, status: 'failed', error: e.message }))
      }
    }, interval)
    return () => clearTimeout(t)
  }, [job, interval])
  return [job, setJob]
}

// Возвращает значение с задержкой обновления. Принимает: value — исходное значение, ms — задержка в мс.
export function useDebounced(value, ms = 300) {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return v
}
