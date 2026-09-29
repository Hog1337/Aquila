import { memo, useCallback, useEffect, useRef, useState } from 'react'
import Icon from '../components/Icon.jsx'
import Thumb from '../components/Thumb.jsx'
import GradCamDialog from '../components/GradCamDialog.jsx'
import { useThreshold } from '../threshold.jsx'
import { api, download, urls } from '../lib/api.js'
import { IMAGE_TYPES, MAX_UPLOAD_MB, MIN_BOX_PX, TH_MAX, TH_MIN, TOP_N } from '../lib/config.js'
import { f2, fmtInt, fmtMs } from '../lib/format.js'
import { clearFile, loadFile, readStored, saveFile, useDebouncedCallback, writeStored } from '../lib/persist.js'

const FRAME_KEY = 'search'

// Ограничивает число диапазоном 0–1. Принимает: v — число.
const clamp01 = (v) => Math.min(1, Math.max(0, v))

// Загружает изображение по URL и возвращает его натуральные размеры. Принимает: url — адрес изображения (в т.ч. blob:).
function measure(url) {
  return new Promise((resolve, reject) => {
    const im = new Image()
    im.onload = () => resolve({ w: im.naturalWidth, h: im.naturalHeight })
    im.onerror = () => reject(new Error('Не удалось прочитать изображение'))
    im.src = url
  })
}

// Числовое поле координаты рамки, применяющее значение по Enter или при потере фокуса. Принимает: label — подпись поля, value — текущее значение в пикселях, disabled — заблокировано ли поле, onChange — обработчик подтверждённого значения.
function CoordField({ label, value, disabled, onChange }) {
  const [draft, setDraft] = useState(null)
  const commit = () => {
    if (draft !== null && draft !== '') onChange(+draft)
    setDraft(null)
  }
  return (
    <label className="coord">
      <span>{label}</span>
      <input
        type="number" inputMode="numeric" value={draft ?? value ?? ''} placeholder="—" disabled={disabled}
        onChange={(e) => setDraft(e.target.value)} onBlur={commit}
        onKeyDown={(e) => { if (e.key === 'Enter') commit(); if (e.key === 'Escape') setDraft(null) }}
        aria-label={`Координата ${label}, пиксели`}
      />
    </label>
  )
}

// Панель запроса: загрузка кадра, рисование рамки BBox и запуск поиска. Принимает: image — { url, w, h } загруженного кадра, initialBox — рамка в долях кадра для восстановления, busy — идёт ли поиск, error — текст ошибки, onPick — выбор файла, onClear — сброс кадра, onBoxChange — изменение рамки завершено, onBoxSync — рамка изменилась (для сохранения), onSearch — запуск поиска, rerank — включён ли re-ranking, onRerankChange — переключение re-ranking.
function QueryPanel({ image, initialBox, busy, error, onPick, onClear, onBoxChange, onBoxSync, onSearch, rerank, onRerankChange }) {
  const { th, setTh } = useThreshold()
  const stageRef = useRef(null)
  const fileRef = useRef(null)
  const drag = useRef(null)
  const [box, setBox] = useState(initialBox)
  const [err, setErr] = useState('')
  const [over, setOver] = useState(false)

  useEffect(() => { onBoxSync(box) }, [box, onBoxSync])

  const W = image ? image.w : 0
  const H = image ? image.h : 0

  // Переводит координаты события в доли ширины/высоты сцены. Принимает: e — событие указателя.
  const point = (e) => {
    const r = stageRef.current.getBoundingClientRect()
    return { x: clamp01((e.clientX - r.left) / r.width), y: clamp01((e.clientY - r.top) / r.height) }
  }

  // Начинает рисование, перемещение или изменение размера рамки. Принимает: e — событие нажатия указателя.
  const onDown = (e) => {
    if (!image) return
    const p = point(e)
    setErr('')
    stageRef.current.setPointerCapture(e.pointerId)
    const h = e.target.dataset && e.target.dataset.h
    if (box && h) {
      drag.current = { mode: 'resize', ax: h.includes('w') ? box.r : box.l, ay: h.includes('n') ? box.b : box.t }
    } else if (box && e.target.id === 'box') {
      drag.current = { mode: 'move', off: { dx: p.x - box.l, dy: p.y - box.t, w: box.r - box.l, h: box.b - box.t } }
    } else {
      drag.current = { mode: 'draw', ax: p.x, ay: p.y }
      setBox({ l: p.x, t: p.y, r: p.x, b: p.y })
    }
    e.preventDefault()
  }

  // Обновляет рамку во время перетаскивания указателя. Принимает: e — событие перемещения указателя.
  const onMove = (e) => {
    const d = drag.current
    if (!d) return
    const p = point(e)
    if (d.mode === 'move') {
      const l = Math.min(Math.max(p.x - d.off.dx, 0), 1 - d.off.w)
      const t = Math.min(Math.max(p.y - d.off.dy, 0), 1 - d.off.h)
      setBox({ l, t, r: l + d.off.w, b: t + d.off.h })
    } else {
      setBox({ l: Math.min(d.ax, p.x), t: Math.min(d.ay, p.y), r: Math.max(d.ax, p.x), b: Math.max(d.ay, p.y) })
    }
  }

  // Завершает перетаскивание рамки и отбрасывает её, если она меньше минимального размера. Аргументов не принимает.
  const onUp = () => {
    if (!drag.current) return
    drag.current = null
    onBoxChange()
    setBox((b) => (b && ((b.r - b.l) * W < MIN_BOX_PX || (b.b - b.t) * H < MIN_BOX_PX) ? null : b))
  }

  const px = box && image && { l: Math.round(box.l * W), t: Math.round(box.t * H), w: Math.round((box.r - box.l) * W), h: Math.round((box.b - box.t) * H) }
  // Обновляет одну координату рамки в пикселях, ограничивая её размерами кадра. Принимает: key — имя координаты (l/t/w/h), v — новое значение в пикселях.
  const setPx = (key, v) => {
    const cur = { ...px, [key]: v }
    const w = Math.min(Math.max(cur.w, MIN_BOX_PX), W)
    const h = Math.min(Math.max(cur.h, MIN_BOX_PX), H)
    const l = Math.min(Math.max(cur.l, 0), W - w)
    const t = Math.min(Math.max(cur.t, 0), H - h)
    setBox({ l: l / W, t: t / H, r: (l + w) / W, b: (t + h) / H })
    onBoxChange()
  }

  // Проверяет тип и размер выбранного файла и передаёт его наверх. Принимает: file — выбранный файл кадра.
  const pickFile = (file) => {
    if (!file) return
    if (!IMAGE_TYPES.includes(file.type)) { setErr('Допустимы только JPEG и PNG'); return }
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) { setErr(`Файл больше ${MAX_UPLOAD_MB} МБ`); return }
    setErr('')
    onPick(file)
  }

  useEffect(() => {
    const onPaste = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
      if (!stageRef.current || stageRef.current.offsetParent === null || document.querySelector('dialog[open]')) return
      const items = e.clipboardData && e.clipboardData.items
      if (!items) return
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const blob = item.getAsFile()
          if (blob) { pickFile(blob); e.preventDefault(); return }
        }
      }
    }
    document.addEventListener('paste', onPaste)
    return () => document.removeEventListener('paste', onPaste)
  })

  // Проверяет наличие кадра и рамки и запускает поиск. Аргументов не принимает.
  const go = () => {
    if (!image) { setErr('Сначала загрузите кадр'); return }
    if (!box) { setErr('Сначала выделите автомобиль рамкой'); return }
    setErr('')
    onSearch({ x: px.l, y: px.t, w: px.w, h: px.h }, rerank)
  }

  return (
    <div className="panel">
      <div className="hd"><h2>1. Запрос</h2><span className="sp" /><span className="meta">{image ? `${image.w} × ${image.h}` : ''}</span></div>
      <div
        id="stage" ref={stageRef} className={`${image ? '' : 'drop-on'} ${over ? 'over' : ''}`} role="application" aria-label="Кадр. Обведите автомобиль рамкой мышью или пальцем"
        style={image ? { aspectRatio: `${image.w}/${image.h}`, width: `min(100%, calc(70vh * ${image.w / image.h}))`, margin: '0 auto' } : { cursor: 'default' }}
        onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}
        onDragOver={(e) => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)} onDrop={(e) => { e.preventDefault(); setOver(false); pickFile(e.dataTransfer.files[0]) }}
      >
        {image && <img className="frame" src={image.url} alt="Кадр запроса" draggable={false} />}
        {box && (
          <div
            id="box"
            style={{ display: 'block', left: `${box.l * 100}%`, top: `${box.t * 100}%`, width: `${(box.r - box.l) * 100}%`, height: `${(box.b - box.t) * 100}%` }}
          >
            {['nw', 'ne', 'sw', 'se'].map((h) => <div key={h} className="hnd" data-h={h} />)}
          </div>
        )}
        {!image && (
          <div className="drop">
            <span className="drop-ic"><Icon name="upload" size={26} /></span>
            <div className="drop-t">Перетащите кадр сюда</div>
            <button className="pri" onClick={() => fileRef.current.click()}><Icon name="upload" /> Выбрать файл</button>
            <div className="hint">или вставьте из буфера: Ctrl+V · JPEG или PNG до {MAX_UPLOAD_MB} МБ</div>
          </div>
        )}
        {image && !box && <div id="emptyhint">Обведите автомобиль рамкой мышью</div>}
      </div>
      <div className="tools">
        <div className="coords">
          {[['x', 'l'], ['y', 't'], ['w', 'w'], ['h', 'h']].map(([label, key]) => (
            <CoordField key={label} label={label} disabled={!box} value={px && px[key]} onChange={(v) => setPx(key, v)} />
          ))}
        </div>
        <span className="sp" />
        <button className="ghost" disabled={!box} onClick={() => { setBox(null); onBoxChange() }}><Icon name="refresh" /> Сбросить</button>
      </div>
      <div className="tools">
        <span className="hint">Порог</span>
        <b className="thval">{f2(th)}</b>
        <input type="range" min={TH_MIN} max={TH_MAX} step="0.01" value={th ?? TH_MIN} disabled={th === null} onChange={(e) => setTh(+e.target.value)} style={{ flex: 1, minWidth: 120 }} aria-label="Порог уверенности" />
      </div>
      <div className="tools actions">
        <label className="toggle">
          <input type="checkbox" checked={rerank} onChange={(e) => onRerankChange(e.target.checked)} />
          <span className="hint">Re-ranking (QE)</span>
        </label>
        <span className="sp" />
        <input ref={fileRef} type="file" accept={IMAGE_TYPES.join(',')} hidden onChange={(e) => { pickFile(e.target.files[0]); e.target.value = '' }} />
        {image && <button className="ghost" onClick={onClear}><Icon name="close" /> Убрать кадр</button>}
        {image && <button className="ghost" onClick={() => fileRef.current.click()}><Icon name="upload" /> Другой кадр</button>}
        <button className="pri" onClick={go} disabled={busy}><Icon name="search" /> {busy ? 'Поиск…' : 'Найти'}</button>
      </div>
      <div className={err || error ? 'errline show' : 'errline'} role="alert">{err || error}</div>
    </div>
  )
}

// Панель сравнения запроса с текущим кандидатом, с вердиктом и переходом между кандидатами. Принимает: result — результат поиска, sel — индекс выбранного кандидата, th — порог отказа, onSelect — выбор кандидата по индексу, onGradCam — открыть Grad-CAM, stale — устарела ли рамка с момента поиска, busy — идёт ли поиск, hasImage — загружен ли кадр.
const ComparePanel = memo(function ComparePanel({ result, sel, th, onSelect, onGradCam, stale, busy, hasImage }) {
  const cands = result ? result.candidates : []
  const cand = cands[sel]
  const accepted = th === null ? 0 : cands.filter((c) => c.score >= th).length
  const best = cands.length ? cands[0].score : null
  const ok = !!cand && th !== null && cand.score >= th
  const refused = !!cand && accepted === 0
  const step = hasImage ? 1 : 0
  return (
    <div className="panel" id="compare">
      <div className="hd">
        <h2>2. Сравнение</h2>
        <span className="sp" />
        {cand && (
          <>
            <span className="cnav">
              <button className="ghost ico" aria-label="Предыдущий кандидат" disabled={sel <= 0} onClick={() => onSelect(sel - 1)}><Icon name="chevl" /></button>
              <span className="meta">#{cand.rank} из {cands.length}</span>
              <button className="ghost ico" aria-label="Следующий кандидат" disabled={sel >= cands.length - 1} onClick={() => onSelect(sel + 1)}><Icon name="chevr" /></button>
            </span>
            <button className="pri" onClick={onGradCam}><Icon name="eye" /> Grad-CAM</button>
          </>
        )}
      </div>
      {!result && (
        <>
          <ol className="steps">
            {['Загрузите кадр', 'Обведите автомобиль рамкой', 'Нажмите «Найти»'].map((t, i) => (
              <li key={t} className={i < step ? 'done' : i === step ? 'now' : ''}><span>{i < step ? <Icon name="check" size={13} /> : i + 1}</span>{t}</li>
            ))}
          </ol>
          <div className="cg ghost" aria-hidden="true">
            <div><div className="ph" /><div className="hint" style={{ marginTop: 8 }}>Запрос</div></div>
            <div><div className="ph" /><div className="hint" style={{ marginTop: 8 }}>Лучший кандидат</div></div>
            <div className="cg-s">{busy && <div className="big">…</div>}<div className="hint">{busy ? 'Идёт поиск…' : 'Здесь появится оценка сходства'}</div></div>
          </div>
        </>
      )}
      {result && result.reranked && <div className="hint" style={{ marginBottom: 9 }}><Icon name="info" size={14} /> Результаты с QE re-ranking</div>}
      {stale && <div className="hint" style={{ marginBottom: 9 }}><Icon name="info" size={14} /> Рамка изменилась — нажмите «Найти»</div>}
      {result && cands.length === 0 && (
        <div className="banner">
          <Icon name="alert" size={18} />
          <div>Галерея пуста: сравнивать не с чем. Загрузите объекты на экране «Галерея».</div>
        </div>
      )}
      {refused && (
        <div className="banner">
          <Icon name="alert" size={18} />
          <div>
            <b>Отказ: уверенного совпадения нет.</b>
            <div className="hint" style={{ color: 'inherit', opacity: 0.85 }}>Ближайший кандидат сходен на {f2(best)}, до порога {f2(th)} не хватает {f2(th - best)}.</div>
          </div>
        </div>
      )}
      {cand && (
        <div className={`cg ${refused ? 'refused' : ''}`}>
          <div><div className="ph"><Thumb src={urls.queryCrop(result.query_id)} alt="Запрос" /></div><div className="hint" style={{ marginTop: 8 }}>Запрос</div></div>
          <div>
            <div className="ph cand"><Thumb src={cand.object_id ? urls.objectCrop(cand.object_id) : null} alt="Кандидат" /></div>
            <div className="hint" style={{ marginTop: 8 }}>{refused ? 'Ближайший, но не совпадение · ' : ''}{[cand.image_id, cand.vehicle_id || 'id неизвестен'].filter(Boolean).join(' · ')}</div>
          </div>
          <div className="cg-s">
            <div className="big">{f2(cand.score)}</div>
            <span className={`chip ${ok ? 'ok' : 'no'}`}>{ok ? 'принят' : refused ? 'отказ' : 'ниже порога'}</span>
            <div className="hint">косинусное сходство, порог {f2(th)}</div>
          </div>
        </div>
      )}
    </div>
  )
})

// Список кандидатов ниже сравнения; ничего не отрисовывает, пока поиска ещё не было. Принимает: result — результат поиска, th — порог отказа, sel — индекс выбранного кандидата, onSelect — выбор кандидата по индексу.
const ResultsPanel = memo(function ResultsPanel({ result, th, sel, onSelect }) {
  if (!result || result.candidates.length === 0) return null
  const cands = result.candidates
  const accepted = th === null ? 0 : cands.filter((c) => c.score >= th).length
  // Выбирает кандидата по клавише Enter или пробелу. Принимает: e — событие клавиатуры, i — индекс кандидата.
  const onRowKey = (e, i) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(i) }
  }
  return (
    <div className="panel cands">
      <div className="hd">
        <h2>3. Кандидаты</h2>
        <span className="meta">{accepted === 0 ? 'отказ' : `топ-${cands.length}`}</span>
        <span className="sp" />
        <button className="ghost" onClick={() => download(urls.candidatesCsv(result.query_id, th))}><Icon name="dl" /> CSV</button>
      </div>
      <div className="sum3">
        <div className="tile"><div className="k">Принято</div><div className="v">{accepted} / {cands.length}</div></div>
        <div className="tile"><div className="k">Лучшее</div><div className="v">{f2(cands[0].score)}</div></div>
        <div className="tile"><div className="k">Порог</div><div className="v">{f2(th)}</div></div>
      </div>
      <div className="rows">
        {cands.map((c, i) => {
          const pass = th !== null && c.score >= th
          return (
            <div
              key={c.rank}
              className={`row ${pass ? '' : 'rej'} ${i === sel ? 'sel' : ''}`}
              role="button" tabIndex={0} aria-pressed={i === sel}
              onClick={() => onSelect(i)} onKeyDown={(e) => onRowKey(e, i)}
            >
              <span className="rk">{c.rank}</span>
              <div className="thumb"><Thumb src={c.object_id ? urls.objectCrop(c.object_id, 128) : null} /></div>
              <div style={{ minWidth: 0 }}>
                <div className="idl">{c.image_id} <span className={`chip ${pass ? 'ok' : ''}`}>{pass ? 'принят' : 'ниже порога'}</span></div>
                <div className="sub">{c.vehicle_id || 'id неизвестен'}</div>
                <div className="track"><i style={{ width: `${Math.round(Math.max(0, c.score) * 100)}%` }} />{th !== null && <b style={{ left: `${Math.round(th * 100)}%` }} />}</div>
              </div>
              <span className="sc">{f2(c.score)}</span>
            </div>
          )
        })}
      </div>
      <p className="hint" style={{ margin: '10px 0 0' }}>
        Поиск {fmtMs(result.timing && result.timing.search_ms)} · инференс {fmtMs(result.timing && result.timing.inference_ms)} · галерея {fmtInt(result.gallery_size)} объектов
      </p>
    </div>
  )
})

// Экран поиска: запрос, сравнение с ближайшим кандидатом и список кандидатов. Принимает: restore — { id, n } запроса из истории для восстановления, либо отсутствует.
export default function SearchPage({ restore }) {
  const { th } = useThreshold()
  const snap = useRef(undefined)
  if (snap.current === undefined) snap.current = readStored('search', null)
  const [file, setFile] = useState(null)
  const [image, setImage] = useState(null)
  const [initialBox, setInitialBox] = useState(null)
  const [result, setResult] = useState(null)
  const [sel, setSel] = useState(0)
  const [rerank, setRerank] = useState(() => (snap.current && snap.current.rerank !== undefined) ? !!snap.current.rerank : true)
  const [stale, setStale] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [gc, setGc] = useState({ open: false, n: 0 })
  const blobUrl = useRef(null)
  const armed = useRef(false)
  const userActed = useRef(false)
  const boxRef = useRef(null)
  const latest = useRef({})

  // Устанавливает выбранный файл кадра, его размеры и рамку. Принимает: f — файл кадра, box — рамка в пикселях или долях кадра, normalized — рамка уже в долях кадра.
  const applyFile = useCallback(async (f, box, normalized = false) => {
    const url = URL.createObjectURL(f)
    let dims
    try {
      dims = await measure(url)
    } catch (e) {
      URL.revokeObjectURL(url)
      throw e
    }
    if (blobUrl.current) URL.revokeObjectURL(blobUrl.current)
    blobUrl.current = url
    setFile(f)
    setImage({ url, ...dims, key: url })
    setInitialBox(!box ? null : normalized ? box : { l: box.x / dims.w, t: box.y / dims.h, r: (box.x + box.w) / dims.w, b: (box.y + box.h) / dims.h })
    setResult(null)
    setSel(0)
    setStale(false)
    setError('')
  }, [])

  useEffect(() => () => { if (blobUrl.current) URL.revokeObjectURL(blobUrl.current) }, [])

  // Обрабатывает выбор нового файла кадра пользователем. Принимает: f — выбранный файл.
  const onPick = useCallback((f) => {
    userActed.current = true
    armed.current = true
    applyFile(f, null).catch((e) => setError(e.message))
  }, [applyFile])

  latest.current = { result, sel, rerank, stale, gcOpen: gc.open }
  // Сохраняет снимок текущего сеанса поиска в хранилище. Аргументов не принимает.
  const persistNow = useCallback(() => {
    if (!armed.current) return
    const l = latest.current
    writeStored(FRAME_KEY, { queryId: l.result ? l.result.query_id : null, box: boxRef.current, sel: l.sel, rerank: l.rerank, stale: l.stale, gc: !!l.result && l.gcOpen })
  }, [])
  const persistSoon = useDebouncedCallback(persistNow, 150)
  // Запоминает текущую рамку и планирует отложенное сохранение снимка. Принимает: b — текущая рамка.
  const syncBox = useCallback((b) => { boxRef.current = b; persistSoon() }, [persistSoon])
  useEffect(() => { persistNow() }, [result, sel, rerank, stale, gc.open, image, persistNow])
  useEffect(() => {
    window.addEventListener('pagehide', persistNow)
    return () => window.removeEventListener('pagehide', persistNow)
  }, [persistNow])
  useEffect(() => { if (file) saveFile(FRAME_KEY, file) }, [file])

  useEffect(() => {
    const s = snap.current
    if (!s || restore) { armed.current = true; return undefined }
    let live = true
    setBusy(true)
    ;(async () => {
      try {
        let r = null
        if (s.queryId != null) {
          try { r = await api.getQuery(s.queryId) } catch (e) { if (e.status !== 404) throw e }
          if (!r) {
            writeStored(FRAME_KEY, null)
            clearFile(FRAME_KEY)
            armed.current = true
            return
          }
        }
        let f = await loadFile(FRAME_KEY)
        if (!f && r) {
          const res = await fetch(urls.queryImage(r.query_id))
          if (res.ok) { const blob = await res.blob(); f = new File([blob], `query-${r.query_id}`, { type: blob.type }) }
        }
        if (!live || userActed.current) return
        if (f) {
          await applyFile(f, 'box' in s ? s.box : r && r.bbox, 'box' in s)
          if (!live || userActed.current) return
          if (r) {
            setResult(r)
            setSel(Math.min(Math.max(s.sel | 0, 0), Math.max(r.candidates.length - 1, 0)))
            if (s.gc && r.candidates.length) setGc({ open: true, n: 1 })
          }
          setStale(!!s.stale)
        }
        armed.current = true
        persistNow()
      } catch (e) {
        if (live) setError(`Не удалось восстановить прошлый запрос: ${e.message}`)
      } finally {
        if (live) setBusy(false)
      }
    })()
    return () => { live = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!restore) return undefined
    userActed.current = true
    armed.current = true
    let live = true
    setBusy(true)
    setError('')
    ;(async () => {
      try {
        const r = await api.getQuery(restore.id)
        const res = await fetch(urls.queryImage(restore.id))
        if (!res.ok) throw new Error('Кадр запроса недоступен')
        const blob = await res.blob()
        if (!live) return
        await applyFile(new File([blob], `query-${restore.id}`, { type: blob.type }), r.bbox)
        if (!live) return
        setResult(r)
      } catch (e) {
        if (live) setError(e.message)
      } finally {
        if (live) setBusy(false)
      }
    })()
    return () => { live = false }
  }, [restore, applyFile])

  // Сбрасывает экран поиска в исходное состояние и стирает сохранённый снимок. Аргументов не принимает.
  const clearAll = useCallback(() => {
    userActed.current = true
    armed.current = true
    if (blobUrl.current) { URL.revokeObjectURL(blobUrl.current); blobUrl.current = null }
    boxRef.current = null
    setFile(null)
    setImage(null)
    setInitialBox(null)
    setResult(null)
    setSel(0)
    setStale(false)
    setError('')
    setGc({ open: false, n: 0 })
    clearFile(FRAME_KEY)
  }, [])

  // Отмечает результат поиска устаревшим после изменения рамки. Аргументов не принимает.
  const markStale = useCallback(() => setStale(true), [])
  // Выполняет поиск по текущему кадру и рамке. Принимает: box — рамка BBox в пикселях кадра, withRerank — включить ли re-ranking для этого запроса.
  const search = useCallback(async (box, withRerank) => {
    userActed.current = true
    armed.current = true
    setBusy(true)
    setError('')
    try {
      setResult(await api.search(file, box, th, TOP_N, withRerank ?? rerank))
      setSel(0)
      setStale(false)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }, [file, th, rerank])
  // Открывает окно Grad-CAM, увеличивая счётчик открытий. Аргументов не принимает.
  const openGc = useCallback(() => setGc((g) => ({ open: true, n: g.n + 1 })), [])
  // Закрывает окно Grad-CAM. Аргументов не принимает.
  const closeGc = useCallback(() => setGc((g) => ({ ...g, open: false })), [])
  // Выбирает кандидата из списка и прокручивает панель сравнения в видимую область при необходимости. Принимает: i — индекс кандидата.
  const selectFromList = useCallback((i) => {
    setSel(i)
    const el = document.getElementById('compare')
    if (el && el.getBoundingClientRect().top < 80) {
      const calm = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      el.scrollIntoView({ block: 'start', behavior: calm ? 'auto' : 'smooth' })
    }
  }, [])

  const cand = result && result.candidates[sel]

  return (
    <>
      <div className="grid2">
        <QueryPanel key={image ? image.key : 'none'} image={image} initialBox={initialBox} busy={busy} error={error} onPick={onPick} onClear={clearAll} onBoxChange={markStale} onBoxSync={syncBox} onSearch={search} rerank={rerank} onRerankChange={setRerank} />
        <ComparePanel result={result} sel={sel} th={th} onSelect={setSel} onGradCam={openGc} stale={stale} busy={busy} hasImage={!!image} />
      </div>
      <ResultsPanel result={result} th={th} sel={sel} onSelect={selectFromList} />
      {cand && <GradCamDialog open={gc.open} openKey={gc.n} onClose={closeGc} queryId={result.query_id} cand={cand} />}
    </>
  )
}
