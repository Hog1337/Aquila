import { useEffect } from 'react'
import Icon from '../components/Icon.jsx'
import Thumb from '../components/Thumb.jsx'
import Seg from '../components/Seg.jsx'
import Pager from '../components/Pager.jsx'
import { api, download, urls } from '../lib/api.js'
import { useAsync, useReloadOnActive } from '../lib/hooks.js'
import { usePersistentState } from '../lib/persist.js'
import { HISTORY_PAGE } from '../lib/config.js'
import { f2, fmtInt, fmtMs, fmtPct } from '../lib/format.js'

// Проверяет, приходятся ли две даты на один день. Принимает: a, b — объекты Date.
const sameDay = (a, b) => a.toDateString() === b.toDateString()
// Отрисовывает время крупно и дату мелко, заменяя сегодняшнюю дату словом «Сегодня». Принимает: iso — строка ISO-даты.
function When({ iso }) {
  const d = new Date(iso)
  const day = sameDay(d, new Date()) ? 'Сегодня' : d.toLocaleDateString('ru-RU')
  return (
    <div>
      <div className="t-main">{d.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</div>
      <div className="t-sub">{day}</div>
    </div>
  )
}

// Экран истории запросов поиска с переходом к любому запросу на экран «Поиск». Принимает: onOpenQuery — открыть запрос по id, active — активен ли экран сейчас.
export default function HistoryPage({ onOpenQuery, active }) {
  const [filter, setFilter] = usePersistentState('history.filter', 'all', (v) => v === 'all' || v === 'ref')
  const [offset, setOffset] = usePersistentState('history.offset', 0, (v) => Number.isInteger(v) && v >= 0)
  const refusedOnly = filter === 'ref'
  const list = useAsync(
    (signal) => api.listQueries({ refused: refusedOnly ? 'true' : undefined, limit: HISTORY_PAGE, offset }, signal),
    [refusedOnly, offset],
  )
  useReloadOnActive(active, list.reload)
  useEffect(() => {
    if (list.data && offset > 0 && offset >= list.data.total) setOffset(0)
  }, [list.data, offset, setOffset])
  const rows = list.data ? list.data.items : []
  const total = list.data ? list.data.total : 0
  const s = list.data ? list.data.summary : null
  // Открывает запрос по клавише Enter или пробелу. Принимает: e — событие клавиатуры, id — id запроса.
  const onKey = (e, id) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpenQuery(id) } }

  return (
    <div className="panel">
      <div className="mg4" style={{ gridTemplateColumns: 'repeat(3,minmax(0,1fr))' }}>
        <div className="tile"><div className="k">Запросов сегодня</div><div className="v">{fmtInt(s && s.today_count)}</div></div>
        <div className="tile"><div className="k">Доля отказов</div><div className="v">{fmtPct(s && s.refused_share)}</div></div>
        <div className="tile"><div className="k">Медианное время</div><div className="v">{fmtMs(s && s.median_ms)}</div></div>
      </div>
      <div className="tools" style={{ margin: '16px 0 9px' }}>
        <Seg options={[['all', 'Все'], ['ref', 'Только отказы']]} value={filter} onChange={(v) => { setFilter(v); setOffset(0) }} label="Фильтр истории" />
        <span className="sp" />
        <button className="ghost" onClick={() => download(urls.historyCsv(refusedOnly))}><Icon name="dl" /> Экспорт CSV</button>
      </div>
      {list.error && <div className="errline show" role="alert">{list.error.message}</div>}
      <div className="tw" hidden={rows.length === 0}>
        <table className="hist">
          <thead>
            <tr><th>Время</th><th className="c-img">Кадр</th><th>Лучшее сходство и порог</th><th>Результат</th><th className="num c-ms">Ответ</th><th className="c-ms" /></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.id} className={`rowlink ${r.refused ? 'ref' : ''}`} tabIndex={0}
                onClick={() => onOpenQuery(r.id)} onKeyDown={(e) => onKey(e, r.id)} aria-label="Открыть запрос на экране «Поиск»"
              >
                <td><When iso={r.created_at} /></td>
                <td className="c-img"><div className="rtn"><Thumb src={urls.queryCrop(r.id, 96)} /></div></td>
                <td>
                  <div className="sc-cell">
                    <span className="sc-n">{f2(r.best_score)}</span>
                    <div className="track"><i style={{ width: `${Math.round(Math.max(0, r.best_score) * 100)}%` }} /><b style={{ left: `${Math.round(r.threshold * 100)}%` }} /></div>
                    <span className="t-sub">порог {f2(r.threshold)}</span>
                  </div>
                </td>
                <td><span className={`chip ${r.refused ? 'no' : 'ok'}`}>{r.refused ? 'отказ' : `принято ${r.accepted_count}`}</span></td>
                <td className="num c-ms">{fmtMs(r.elapsed_ms)}</td>
                <td className="go c-ms"><Icon name="chevr" size={16} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!list.loading && !list.error && rows.length === 0 && (
        <div className="empty" style={{ marginTop: 12 }}>{refusedOnly ? 'Отказов пока нет' : 'Запросов пока нет'}</div>
      )}
      <Pager offset={offset} limit={HISTORY_PAGE} total={total} onChange={setOffset} />
    </div>
  )
}
