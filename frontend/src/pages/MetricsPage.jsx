import { useEffect, useMemo, useRef, useState } from 'react'
import { usePersistentState, useStoredJobId } from '../lib/persist.js'
import Icon from '../components/Icon.jsx'
import MetricsUploadPanel from '../components/MetricsUploadPanel.jsx'
import { useThreshold } from '../threshold.jsx'
import { api, ApiError } from '../lib/api.js'
import { useAsync, useJob, isActive, useReloadOnActive } from '../lib/hooks.js'
import { TH_MAX, TH_MIN, TNR_DEFAULT, TNR_OPTIONS } from '../lib/config.js'
import { f2, fmtDateTime, fmtInt } from '../lib/format.js'

const X0 = 34, PAD_R = 10, Y0 = 26
const TICKS = Array.from({ length: Math.floor((TH_MAX - TH_MIN) * 10) + 1 }, (_, i) => Math.round((TH_MIN + i / 10) * 10) / 10).filter((t) => t <= TH_MAX)

function useWidth() {
  const ref = useRef(null)
  const [w, setW] = useState(400)
  useEffect(() => {
    const el = ref.current
    if (!el) return undefined
    const ro = new ResizeObserver(() => setW(Math.max(240, Math.round(el.clientWidth))))
    ro.observe(el)
    setW(Math.max(240, Math.round(el.clientWidth)))
    return () => ro.disconnect()
  }, [])
  return [ref, w]
}

const edge = (x, w, half) => (x < X0 + half ? 'start' : x > w - PAD_R - half ? 'end' : 'middle')

function nearest(curve, th) {
  if (!curve.length || th === null) return null
  return curve.reduce((a, b) => (Math.abs(b.threshold - th) < Math.abs(a.threshold - th) ? b : a))
}

function recommended(curve, minTNR) {
  let best = null
  curve.forEach((p) => { if (p.tnr >= minTNR && (best === null || p.f1 > best.f1)) best = p })
  return best
}

function Chart({ curve, th, rec, onChange }) {
  const [ref, W] = useWidth()
  const X1 = W - PAD_R
  const H = Math.round(Math.min(330, Math.max(232, W * 0.42)))
  const Y1 = H - 50
  const sy = (v) => Y1 - v * (Y1 - Y0)
  const sx = (t) => X0 + ((t - TH_MIN) / (TH_MAX - TH_MIN)) * (X1 - X0)
  const paths = useMemo(() => {
    const pts = curve.filter((p) => p.threshold >= TH_MIN && p.threshold <= TH_MAX)
    const line = (key) => pts.map((p) => `${sx(p.threshold).toFixed(1)},${sy(p[key]).toFixed(1)}`).join(' ')
    return { f1: line('f1'), precision: line('precision'), recall: line('recall'), tnr: line('tnr') }
  }, [curve, W, H])
  const cur = th === null ? null : sx(th)
  const m = nearest(curve, th)
  const line = (pts, stroke, w, dash) => (
    <polyline fill="none" stroke={stroke} strokeWidth={w} strokeDasharray={dash} strokeLinecap="round" strokeLinejoin="round" points={pts} />
  )
  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img" aria-label="Precision, Recall, F1 и TNR в зависимости от порога" style={{ display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={X0} x2={X1} y1={sy(v)} y2={sy(v)} stroke="var(--line)" strokeWidth="1" />
            <text className="ax" x={X0 - 8} y={sy(v) + 4} textAnchor="end">{v}</text>
          </g>
        ))}
        {TICKS.map((t) => (
          <text key={t} className="ax" x={sx(t)} y={Y1 + 18} textAnchor="middle">{t.toFixed(1)}</text>
        ))}
        {line(paths.precision, 'var(--accent)', 1.8, '0')}
        {line(paths.recall, 'var(--accent-2)', 1.8, '6 3')}
        {line(paths.tnr, 'var(--hot)', 1.8, '2 3')}
        {line(paths.f1, '#fff', 2.8, '0')}
        {cur !== null && <line x1={cur} x2={cur} y1={Y0} y2={Y1} stroke="#fff" strokeOpacity=".7" strokeWidth="1" strokeDasharray="3 3" />}
        {cur !== null && m && <circle cx={cur} cy={sy(m.f1)} r="5" fill="#fff" />}
        {cur !== null && <text className="ax" x={cur} y="14" textAnchor={edge(cur, W, 34)} style={{ fill: 'var(--ink)' }}>порог {f2(th)}</text>}
        {rec && (
          <g>
            <polygon points={`${sx(rec.threshold) - 4},${Y1 + 30} ${sx(rec.threshold) + 4},${Y1 + 30} ${sx(rec.threshold)},${Y1 + 23}`} fill="var(--accent-2)" />
            <text className="ax" x={sx(rec.threshold)} y={Y1 + 44} textAnchor={edge(sx(rec.threshold), W, 52)} style={{ fill: 'var(--accent-2)' }}>рекомендуем {f2(rec.threshold)}</text>
          </g>
        )}
      </svg>
      <input className="chart-slider" type="range" min={TH_MIN} max={TH_MAX} step="0.01" value={th ?? TH_MIN} disabled={th === null} onChange={(e) => onChange(+e.target.value)} aria-label="Порог отказа" style={{ width: X1 - X0 + 16, marginLeft: X0 - 8 }} />
    </div>
  )
}

// Панель загрузки CSV и запуска оценки — стиль как в UploadTray галереи
export default function MetricsPage({ active }) {
  const { th, setTh, saved, save } = useThreshold()
  const [showRunPanel, setShowRunPanel] = useState(false)
  const [minT, setMinT] = usePersistentState('metrics.min-tnr', TNR_DEFAULT, (v) => TNR_OPTIONS.includes(v))
  const [msg, setMsg] = useState('')
  const [msgErr, setMsgErr] = useState(false)
  const [saving, setSaving] = useState(false)
  const [cleaning, setCleaning] = useState(null) // 'query' | 'gallery' | 'all' | null

  const latest = useAsync(
    (signal) => api.latestMetrics(signal).catch((e) => { if (e.status === 404) return null; throw e }),
    [],
  )
  useReloadOnActive(active, latest.reload)
  const [job, setJob] = useJob(api.getMetricsRun)
  useStoredJobId('metrics.job', job, setJob, api.getMetricsRun)

  const run = (job && job.status === 'done' && job.run) || latest.data
  const curve = (run && run.details && run.details.curve) || []
  const m = nearest(curve, th)
  const rec = recommended(curve, minT)

  const change = (v) => { setTh(v); setMsg(''); setMsgErr(false) }

  const onStart = (newJob) => {
    setShowRunPanel(false)
    setJob(newJob)
  }

  const onSave = async () => {
    const reason = run && m
      ? `Выбран в интерфейсе по кривой запуска #${run.id} (${run.model_version}, ${run.n_queries} запросов): F1 ${f2(m.f1)}, TNR ${f2(m.tnr)}`
      : 'Выбран в интерфейсе'
    setSaving(true)
    try {
      await save(reason)
      setMsg(`Сохранено: порог ${f2(th)} применяется ко всем новым запросам`)
      setMsgErr(false)
    } catch (e) {
      setMsg(`Не удалось сохранить: ${e.message}`)
      setMsgErr(true)
    } finally {
      setSaving(false)
    }
  }

  const KPIS = [
    ['mAP', run && run.map, 'средняя точность ранжирования'],
    ['Rank-1', run && run.rank1, 'верный автомобиль на 1-м месте'],
    ['Rank-5', run && run.rank5, 'верный автомобиль в первой пятёрке'],
    ['PR-AUC', run && run.pr_auc, 'площадь под кривой Precision-Recall, не зависит от порога'],
  ]
  const dirty = !!saved && th !== null && Math.abs(th - saved.value) > 0.004

  return (
    <div className="panel">
      <div className="hd">
        <span className="hint">
          {run ? `Валидация · ${run.model_version} · ${fmtDateTime(run.created_at)} · ${fmtInt(run.n_queries)} запросов` : 'Оценка ещё не запускалась'}
        </span>
        <span className="sp" />
        <button className="ghost" style={{ color: 'var(--hot)' }} disabled={isActive(job)} onClick={async () => {
          if (!confirm('Очистить всю историю метрик?')) return
          try {
            await api.clearMetrics()
            latest.reload()
            setJob(null)
            setMsg('История метрик очищена')
            setMsgErr(false)
          } catch (e) {
            setMsg(e.message); setMsgErr(true)
          }
        }}><Icon name="close" size={14} /> Очистить метрики</button>
        <button className="pri" onClick={() => setShowRunPanel((v) => !v)} disabled={isActive(job)}>
          <Icon name="play" /> {showRunPanel ? 'Закрыть' : 'Оценить метрики'}
        </button>
      </div>

      {showRunPanel && <MetricsUploadPanel onStart={onStart} onClose={() => setShowRunPanel(false)} />}

      {latest.error && <div className="errline show" role="alert">{latest.error.message}</div>}
      {job && (
        <div className="job">
          <div className={job.status === 'failed' ? 'err' : undefined} style={{ minWidth: 150, fontWeight: 600 }} role={job.status === 'failed' ? 'alert' : undefined}>
            {job.status === 'done' ? 'Оценка готова' : job.status === 'failed' ? `Ошибка: ${job.error || 'оценка не выполнена'}` : `Оценка ${job.progress ?? 0}%`}
          </div>
          {isActive(job) && <div className="pbar" role="progressbar" aria-label="Оценка на валидации" aria-valuemin="0" aria-valuemax="100" aria-valuenow={job.progress ?? 0}><i style={{ width: `${job.progress ?? 0}%` }} /></div>}
          {job.status === 'done' && (
            <div className="tools" style={{ marginTop: 10 }}>
              <span className="hint">Что делаем с загруженными данными?</span>
              <span className="sp" />
              {[['query', 'Удалить query'], ['gallery', 'Удалить gallery'], ['all', 'Удалить всё']].map(([scope, label]) => (
                <button key={scope} className="ghost sm" disabled={cleaning !== null}
                  onClick={async () => {
                    setCleaning(scope)
                    try {
                      await api.cleanupMetricsRun(job.id, scope)
                      latest.reload()
                      setCleaning(null)
                    } catch (e) {
                      setCleaning(null)
                      setMsg(e.message); setMsgErr(true)
                    }
                  }}>
                  {cleaning === scope ? <><svg width="14" height="14" viewBox="0 0 20 20" aria-hidden="true" style={{ display: 'inline-block', verticalAlign: 'middle' }}><circle cx="10" cy="10" r="8" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeDasharray="13 38" className="ut-spin" /></svg> Удаление...</> : <><Icon name="close" size={14} /> {label}</>}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      <div className="mg4">
        {KPIS.map(([k, v, d]) => (
          <div className="tile kpi" key={k}><div className="k">{k}</div><div className="v">{f2(v)}</div><div className="d">{d}</div></div>
        ))}
      </div>
      <div className="mgrid">
        <div className="panel">
          <div className="hd"><h2>Качество отказа по порогу</h2></div>
          {curve.length === 0 ? (
            <div className="empty">Нет данных: нажмите «Оценить метрики» и загрузите CSV</div>
          ) : (
            <>
              <Chart curve={curve} th={th} rec={rec} onChange={change} />
              <div className="leg">
                <span><i className="ln" style={{ borderColor: '#fff' }} />F1</span>
                <span><i className="ln" style={{ borderColor: 'var(--accent)' }} />Precision</span>
                <span><i className="ln" style={{ borderColor: 'var(--accent-2)', borderTopStyle: 'dashed' }} />Recall</span>
                <span><i className="ln" style={{ borderColor: 'var(--hot)', borderTopStyle: 'dotted' }} />TNR</span>
              </div>
            </>
          )}
        </div>
        <div className="panel">
          <div className="hd"><h2>Порог отказа</h2></div>
          <div className="big" style={{ textAlign: 'left' }}>{f2(th)}</div>
          <div className="hint" style={{ margin: '2px 0 12px' }}>{curve.length ? 'Двигайте ползунок под графиком' : 'Ползунок появится под графиком после оценки'}</div>
          <div className="kv"><span>Precision</span><b>{f2(m && m.precision)}</b></div>
          <div className="kv"><span>Recall</span><b>{f2(m && m.recall)}</b></div>
          <div className="kv"><span>F1</span><b>{f2(m && m.f1)}</b></div>
          <div className="kv"><span>TNR</span><b>{f2(m && m.tnr)}</b></div>
          <p className="hint" style={{ margin: '13px 0 6px' }}>Максимум F1 при TNR не ниже</p>
          <div className="tools" style={{ margin: 0 }}>
            <select value={minT} onChange={(e) => setMinT(+e.target.value)} aria-label="Минимальный TNR">
              {TNR_OPTIONS.map((v) => <option key={v} value={v}>{v.toFixed(2)}</option>)}
            </select>
            <span className="hint">→ порог <b style={{ color: 'var(--ink)' }}>{rec ? f2(rec.threshold) : '—'}</b></span>
          </div>
          <div className="tools" style={{ marginTop: 15 }}>
            <button className="ghost" disabled={!rec} onClick={() => change(rec.threshold)}>Взять рекомендованный</button>
            <button className="pri" disabled={th === null || saving || !dirty} onClick={onSave}>{th !== null && !dirty ? 'Сохранено' : 'Сохранить'}</button>
          </div>
          {dirty && (
            <div className="thstate" role="status">
              <Icon name="info" size={15} />
              <div><b>Не сохранено.</b> В системе записан {f2(saved.value)}, выбран {f2(th)}. Выбор хранится в этом браузере; в системе порог сменится после «Сохранить».</div>
            </div>
          )}
          <p className={msgErr ? 'hint err' : 'hint'} style={{ marginTop: 10 }} role={msgErr ? 'alert' : undefined} aria-live="polite">{msg}</p>
          {saved && !dirty && <p className="hint" style={{ marginTop: 4 }}>Действующий порог в системе: {f2(saved.value)}</p>}
        </div>
      </div>
    </div>
  )
}
