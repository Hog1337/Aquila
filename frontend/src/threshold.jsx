import { createContext, useContext, useEffect, useState } from 'react'
import { api } from './lib/api.js'
import { readStored, writeStored } from './lib/persist.js'
import { TH_MAX, TH_MIN } from './lib/config.js'

const ThresholdContext = createContext(null)
const DRAFT = 'threshold-draft'
// Округляет число до двух знаков после запятой. Принимает: v — число.
const round2 = (v) => Math.round(v * 100) / 100
// Проверяет, что значение — конечное число в допустимом диапазоне порога. Принимает: v — проверяемое значение.
const validDraft = (v) => typeof v === 'number' && Number.isFinite(v) && v >= TH_MIN && v <= TH_MAX
// Проверяет, отличается ли рабочее значение порога от сохранённого. Принимает: th — рабочее значение, saved — сохранённый порог.
export const differs = (th, saved) => !!saved && th !== null && Math.abs(th - saved.value) > 0.004

// Провайдер контекста единого порога отказа: хранит рабочее и сохранённое значения, черновик и ошибку загрузки. Принимает: children — дочерние элементы.
export function ThresholdProvider({ children }) {
  const [th, setRaw] = useState(() => {
    const d = readStored(DRAFT, null)
    return validDraft(d) ? round2(d) : null
  })
  const [saved, setSaved] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    const ac = new AbortController()
    api.getThreshold(ac.signal).then(
      (t) => { setSaved(t); setRaw((cur) => (cur === null ? round2(t.value) : cur)) },
      (e) => { if (e.name !== 'AbortError') setError(e.message) },
    )
    return () => ac.abort()
  }, [])

  useEffect(() => {
    if (!saved || th === null) return
    writeStored(DRAFT, differs(th, saved) ? th : null)
  }, [th, saved])

  // Устанавливает рабочее значение порога с округлением. Принимает: v — новое значение.
  const setTh = (v) => setRaw(round2(v))
  // Сохраняет порог через API и обновляет сохранённое значение. Принимает: reason — причина изменения.
  const save = async (reason) => {
    const t = await api.putThreshold(th, reason)
    setSaved(t)
    return t
  }

  return <ThresholdContext.Provider value={{ th, setTh, saved, save, error, dirty: differs(th, saved) }}>{children}</ThresholdContext.Provider>
}

// Возвращает текущее значение контекста порога отказа. Аргументов не принимает.
export const useThreshold = () => useContext(ThresholdContext)
