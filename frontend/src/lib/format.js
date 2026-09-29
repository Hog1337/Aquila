// Форматирует число до двух знаков после запятой либо возвращает прочерк. Принимает: v — число или null/undefined.
export const f2 = (v) => (v === null || v === undefined ? '—' : Number(v).toFixed(2))
// Форматирует целое число с разделителями разрядов либо возвращает прочерк. Принимает: v — число или null/undefined.
export const fmtInt = (v) => (v === null || v === undefined ? '—' : Number(v).toLocaleString('ru-RU'))
// Форматирует значение в миллисекундах либо возвращает прочерк. Принимает: v — число мс или null/undefined.
export const fmtMs = (v) => (v === null || v === undefined ? '—' : `${Math.round(v)} мс`)
// Форматирует долю в проценты либо возвращает прочерк. Принимает: v — доля от 0 до 1 или null/undefined.
export const fmtPct = (v) => (v === null || v === undefined ? '—' : `${Math.round(v * 100)}%`)

// Форматирует размер в байтах в читаемые единицы (Б/КБ/МБ/ГБ). Принимает: n — размер в байтах или null/undefined.
export function fmtBytes(n) {
  if (n === null || n === undefined) return '—'
  if (n < 1024) return `${n} Б`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} КБ`
  if (n < 1024 ** 3) return `${(n / 1024 / 1024).toFixed(1)} МБ`
  return `${(n / 1024 ** 3).toFixed(2)} ГБ`
}

// Форматирует скорость передачи в байтах в секунду. Принимает: bps — байт в секунду.
export const fmtSpeed = (bps) => `${fmtBytes(Math.round(bps))}/с`

// Форматирует оставшееся время в секундах в короткую строку («≈ 9 с», «≈ 3 мин», «≈ 1 ч 20 мин»). Принимает: sec — секунды.
export function fmtEta(sec) {
  if (!Number.isFinite(sec) || sec < 0 || sec > 86400) return ''
  if (sec < 60) return `≈ ${Math.max(1, Math.round(sec))} с`
  if (sec < 3600) return `≈ ${Math.round(sec / 60)} мин`
  return `≈ ${Math.floor(sec / 3600)} ч ${Math.round((sec % 3600) / 60)} мин`
}

// Форматирует ISO-дату в локализованную дату и время либо возвращает прочерк. Принимает: iso — строка ISO-даты или пусто.
export const fmtDateTime = (iso) => (iso ? new Date(iso).toLocaleString('ru-RU', { dateStyle: 'short', timeStyle: 'medium' }) : '—')
