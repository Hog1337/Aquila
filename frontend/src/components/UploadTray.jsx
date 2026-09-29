import { useEffect, useRef, useState } from 'react'
import Icon from './Icon.jsx'
import { isBusy, useUploads } from '../lib/uploads.jsx'
import { fmtBytes, fmtEta, fmtInt, fmtSpeed } from '../lib/format.js'

// Ограничивает число диапазоном. Принимает: v — число, lo — нижняя граница, hi — верхняя граница.
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))
// Вычисляет процент a от b, ограниченный 0–100. Принимает: a — числитель, b — знаменатель.
const pctOf = (a, b) => (b > 0 ? clamp(Math.floor((a / b) * 100), 0, 100) : 0)
// Вычисляет процент выполнения фонового задания. Принимает: j — объект задания с полями progress/processed/total.
const jobPct = (j) => clamp(Math.round(j.progress ?? (j.processed / j.total) * 100), 0, 100)

// Строит описание шага обработки: векторизация для импорта, расчёт признака для одного объекта.
// Для импорта использует t.vect (batch-статус) или t.job (легаси).
// Принимает: t — объект задания.
function serverStep(t) {
  const job = t.job
  const vect = t.vect
  const running = job && job.status === 'running' && job.total > 0
  const isImport = t.kind === 'import'

  // Определяем состояние: активен, если фаза processing, или uploading+vect уже есть
  let state = 'pending'
  if (t.phase === 'processing') state = 'active'
  else if (t.phase === 'done') state = 'done'
  else if (t.phase === 'failed' && (t.failedAt === 'processing' || t.failedAt === 'start')) state = 'failed'
  else if (t.phase === 'uploading' && vect && isImport) state = 'active' // показываем vect ещё во время загрузки

  // Ветка с batch-статусом (векторизация через upload-image)
  if (vect && isImport) {
    let sub = 'После загрузки кадров'
    if (state === 'active') {
      const done = vect.ready + vect.failed
      sub = `Завекторизовано ${fmtInt(done)} из ${fmtInt((vect.tracked || vect.total))}`
      if (vect.failed > 0) sub += ` · ${vect.failed} с ошибкой`
    } else if (state === 'done') {
      sub = `Обработано ${fmtInt(vect.ready)} из ${fmtInt((vect.tracked || vect.total))}`
      if (vect.failed > 0) sub += ` · ${vect.failed} с ошибкой`
    }
    const pct = state === 'done' ? 100 : (state === 'active' ? pctOf(vect.ready + vect.failed, (vect.tracked || vect.total)) : null)
    return { key: 'proc', label: 'Векторизация', sub, state, pct }
  }

  // Легаси: job-статус из сессионного импорта
  let sub = isImport ? 'После загрузки кадров' : 'После передачи кадра'
  if (state === 'active') {
    if (!isImport) sub = 'Вычисляем вектор автомобиля'
    else if (!job) sub = 'Запускаем обработку'
    else sub = running ? `${fmtInt(job.processed)} из ${fmtInt(job.total)}` : 'В очереди'
  } else if (state === 'done') {
    sub = !isImport ? 'Добавлено в галерею' : (job ? `Обработано ${fmtInt(job.processed)} из ${fmtInt(job.total)}${job.error ? ` · ${job.error}` : ''}` : 'Готово')
  } else if (state === 'failed') sub = t.failedAt === 'start' ? 'Не запущено' : 'Не выполнено'
  const pct = state === 'done' ? 100 : (state === 'active' ? (running ? jobPct(job) : null) : 0)
  return { key: 'proc', label: isImport ? 'Обработка на сервере' : 'Расчёт признака и запись', sub, state, pct }
}

// Строит описание шага передачи файла одним запросом (CSV или одиночный кадр). Принимает: t — объект задания.
function singleFileStep(t) {
  const f = t.files[0]
  const partial = t.total > 0 ? clamp(t.loaded / t.total, 0, 1) * f.size : 0
  const sending = t.phase === 'uploading' && (t.kind === 'object' || t.stage === 'csv')
  const failedHere = t.phase === 'failed' && (t.failedAt === 'csv' || t.failedAt === 'uploading')
  let state = 'done'
  if (sending) state = 'active'
  else if (failedHere) state = 'failed'
  else if (t.phase === 'cancelled' && !t.session && t.kind === 'import') state = 'pending'
  else if (t.phase === 'cancelled' && t.kind === 'object' && partial < f.size) state = 'pending'
  let sub = fmtBytes(f.size)
  if (state === 'active') {
    sub = `${fmtBytes(Math.round(partial))} из ${fmtBytes(f.size)}`
    if (t.speed > 1024) sub += ` · ${fmtSpeed(t.speed)}`
    const eta = t.speed > 1024 ? fmtEta((f.size - partial) / t.speed) : ''
    if (eta) sub += ` · ${eta}`
  } else if (state === 'failed' || (t.phase === 'cancelled' && partial > 0 && partial < f.size)) {
    sub = state === 'failed' && t.total > 0 && t.loaded >= t.total ? 'Отклонён сервером' : `Передано ${fmtBytes(Math.round(partial))} из ${fmtBytes(f.size)}`
  } else if (t.kind === 'import' && t.session) {
    sub = `${fmtInt(t.session.total)} строк · ${fmtInt(t.session.images)} кадров`
  }
  return { key: 'file', label: f.name, sub, state, pct: state === 'done' ? 100 : pctOf(partial, f.size) }
}

// Строит описание шага «Кадры из папки»: ожидание выбора, сопоставление с CSV, передача по одному. Принимает: t — объект задания.
function imagesStep(t) {
  const { session, folder, imgs } = t
  let state = 'pending'
  if (t.phase === 'awaiting') state = 'await'
  else if (t.phase === 'uploading' && t.stage === 'images') state = 'active'
  else if (t.phase === 'failed' && t.failedAt === 'images') state = 'failed'
  else if (t.phase === 'processing' || t.phase === 'done' || (t.phase === 'failed' && (t.failedAt === 'start' || t.failedAt === 'processing'))) state = 'done'
  const label = folder ? `Папка «${folder.name}»` : 'Кадры из папки'
  let sub = 'После загрузки CSV'
  let warn = false
  if (state === 'await') {
    if (!folder) sub = `Выберите папку: нужно ${fmtInt(session.images)} кадров`
    else {
      const more = folder.missing > folder.missingSample.length ? ` и ещё ${fmtInt(folder.missing - folder.missingSample.length)}` : ''
      sub = `Найдено ${fmtInt(folder.matched)} из ${fmtInt(session.images)} кадров. Нет в папке: ${folder.missingSample.join(', ')}${more}`
      warn = true
    }
  } else if (imgs && (state === 'active' || state === 'failed' || t.phase === 'cancelled')) {
    sub = `${fmtInt(imgs.done)} из ${fmtInt(imgs.count)} кадров · ${fmtBytes(Math.round(imgs.loaded))} из ${fmtBytes(imgs.bytes)}`
    if (t.vect) {
      const vdone = t.vect.ready + t.vect.failed
      if (vdone > 0) sub += ` · завекторизовано ${fmtInt(vdone)} из ${fmtInt((t.vect.tracked || t.vect.total))}`
    }
    if (state === 'active' && imgs.speed > 1024) {
      sub += ` · ${fmtSpeed(imgs.speed)}`
      const eta = fmtEta((imgs.bytes - imgs.loaded) / imgs.speed)
      if (eta) sub += ` · ${eta}`
    }
    if (state === 'active' && imgs.speed <= 1024 && imgs.flying && imgs.flying.length) sub += `. Ждём ответа сервера: ${imgs.flying.join(', ')}`
  } else if (state === 'done' && imgs) {
    sub = `${fmtInt(imgs.count)} кадров · ${fmtBytes(imgs.bytes)}`
    if (t.vect) {
      const vdone = t.vect.ready + t.vect.failed
      sub += ` · завекторизовано ${fmtInt(vdone)} из ${fmtInt((t.vect.tracked || t.vect.total))}`
    }
  }
  const pct = state === 'done' ? 100 : (imgs && state !== 'await' && state !== 'pending' ? pctOf(imgs.loaded, imgs.bytes) : 0)
  return { key: 'imgs', label, sub, state, pct, warn }
}

// Собирает список шагов задания в зависимости от его вида. Принимает: t — объект задания.
function stepsOf(t) {
  if (t.kind === 'object') return [singleFileStep(t), serverStep(t)]
  if (t.restored) return [serverStep(t)]
  return [singleFileStep(t), imagesStep(t), serverStep(t)]
}

// Вычисляет общий процент для шапки задания по текущей фазе. Принимает: t — объект задания.
function taskPct(t) {
  if (t.phase === 'uploading') {
    if (t.kind === 'import' && t.stage === 'images') return t.imgs ? pctOf(t.imgs.loaded, t.imgs.bytes) : 0
    return t.total > 0 ? pctOf(t.loaded, t.total) : 0
  }
  if (t.phase === 'processing') {
    if (t.vect) return pctOf(t.vect.ready + t.vect.failed, (t.vect.tracked || t.vect.total))
    const j = t.job
    return j && j.status === 'running' && j.total > 0 ? jobPct(j) : null
  }
  return t.phase === 'done' ? 100 : 0
}

// Формирует короткую строку текущей фазы задания. Принимает: t — объект задания.
function phaseLine(t) {
  if (t.phase === 'awaiting') {
    return t.folder ? `Найдено ${fmtInt(t.folder.matched)} из ${fmtInt(t.session.images)} кадров` : 'Выберите папку с кадрами'
  }
  if (t.phase === 'uploading') {
    if (t.kind === 'import' && t.stage === 'images') return t.imgs ? `Кадры · ${fmtInt(t.imgs.done)} из ${fmtInt(t.imgs.count)}` : 'Кадры'
    if (t.kind === 'import') return 'Передача CSV'
    return t.speed > 1024 ? `Передача · ${fmtSpeed(t.speed)}` : 'Передача'
  }
  if (t.kind === 'object') return 'Расчёт признака'
  // Фаза processing — показываем прогресс векторизации
  if (t.vect) {
    const done = t.vect.ready + t.vect.failed
    return `Векторизация · ${fmtInt(done)} из ${fmtInt((t.vect.tracked || t.vect.total))}`
  }
  const j = t.job
  if (!j) return 'Запускаем обработку'
  if (j.status === 'running' && j.total > 0) return `Обработка · ${fmtInt(j.processed)} из ${fmtInt(j.total)}`
  return 'В очереди на обработку'
}

// Строит сводку по всем заданиям для свёрнутой шапки трея. Принимает: tasks — список заданий.
function summary(tasks) {
  const act = tasks.filter(isBusy)
  if (act.length) {
    const waiting = act.filter((t) => t.phase === 'awaiting')
    if (waiting.length === act.length) {
      return { state: 'await', line1: act.length === 1 ? act[0].title : `Ждут папку: ${act.length}`, line2: phaseLine(act[0]), pct: 0 }
    }
    const working = act.filter((t) => t.phase !== 'awaiting')
    const known = working.map(taskPct).filter((p) => p !== null)
    return {
      state: 'busy',
      line1: act.length === 1 ? act[0].title : `Загрузок в работе: ${act.length}`,
      line2: act.length === 1 ? phaseLine(act[0]) : 'Передача файлов и обработка на сервере',
      pct: known.length ? Math.round(known.reduce((a, b) => a + b, 0) / known.length) : null,
    }
  }
  const failed = tasks.filter((t) => t.phase === 'failed')
  if (failed.length) return { state: 'failed', line1: 'Ошибка загрузки', line2: failed[0].error || 'Загрузка не выполнена', pct: 100 }
  if (tasks.every((t) => t.phase === 'cancelled')) return { state: 'idle', line1: 'Отменено', line2: 'Файлы не добавлены', pct: 0 }
  // Показываем итог векторизации, если есть vect-данные
  const v = tasks.find((t) => t.kind === 'import' && t.vect && (t.vect.tracked || t.vect.total) > 0)
  const j = tasks.find((t) => t.kind === 'import' && t.job)
  return {
    state: 'done', line1: 'Готово',
    line2: v ? `Обработано ${fmtInt(v.vect.ready)} из ${fmtInt((v.vect.tracked || v.vect.total))}${v.vect.failed > 0 ? ` · ${v.vect.failed} с ошибкой` : ''}` : (j ? `Обработано ${fmtInt(j.job.processed)} из ${fmtInt(j.job.total)}` : 'Добавлено в галерею'),
    pct: 100,
  }
}

// Отрисовывает круглый индикатор состояния шага. Принимает: state — состояние шага ('done'/'failed'/'await'/'active'/'pending').
function StateIcon({ state }) {
  if (state === 'done') {
    return (
      <svg className="ut-si" width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
        <circle cx="10" cy="10" r="9" fill="var(--accent)" />
        <path className="ut-tick" d="M6 10.4 8.8 13.1 14 7.1" pathLength="1" fill="none" stroke="var(--accent-ink)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    )
  }
  if (state === 'failed') {
    return (
      <svg className="ut-si" width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
        <circle cx="10" cy="10" r="9" fill="var(--hot)" />
        <path d="M7 7l6 6M13 7l-6 6" fill="none" stroke="var(--accent-ink)" strokeWidth="2" strokeLinecap="round" />
      </svg>
    )
  }
  return (
    <svg className="ut-si" width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
      <circle cx="10" cy="10" r="8" fill="none" stroke={state === 'await' ? 'var(--accent-2)' : 'var(--line-strong)'} strokeWidth="1.6" />
      {state === 'await' && <circle className="ut-beat" cx="10" cy="10" r="3" fill="var(--accent-2)" />}
      {state === 'active' && <circle className="ut-spin" cx="10" cy="10" r="8" fill="none" stroke="var(--accent-2)" strokeWidth="1.8" strokeLinecap="round" strokeDasharray="13 38" />}
    </svg>
  )
}

// Отрисовывает строку одного шага задания с полосой прогресса. Принимает: s — описание шага из stepsOf().
function Step({ s }) {
  const ind = s.state === 'active' && s.pct === null
  return (
    <li className={`ut-step ${s.state}`}>
      <StateIcon state={s.state} />
      <div className="ut-sb">
        <div className="ut-sl">
          <span className="ut-sn" title={s.label}>{s.label}</span>
          <span className="ut-spc">{s.state === 'active' && s.pct !== null ? `${s.pct}%` : ''}</span>
        </div>
        <div className={s.warn ? 'ut-ss warn' : 'ut-ss'}>{s.sub}</div>
        <div
          className={ind ? 'ut-bar ind' : 'ut-bar'} role="progressbar" aria-label={s.label}
          aria-valuemin="0" aria-valuemax="100" aria-valuenow={ind ? undefined : s.pct}
        >
          <i style={{ '--p': s.pct ?? 0 }} />
        </div>
      </div>
    </li>
  )
}

// Отрисовывает карточку одного задания со всеми его шагами и действиями. Принимает: t — задание, single — единственное ли это задание в трее, page — текущий экран, onNavigate — переход между экранами.
function Task({ t, single, page, onNavigate }) {
  const { cancel, retry, dismiss, chooseFolder, confirmFolder } = useUploads()
  const dirRef = useRef(null)
  const steps = stepsOf(t)
  const src = t.job && t.job.source_name
  const busy = isBusy(t)
  const canCancel = t.phase === 'uploading' || t.phase === 'awaiting' || (t.kind === 'import' && t.phase === 'processing')
  const rejected = t.errorStatus >= 400 && t.errorStatus < 500
  const canRetry = t.phase === 'failed' && !t.restored && !rejected && (t.kind === 'object' || ['csv', 'images', 'start', 'processing'].includes(t.failedAt))
  // Открывает системный диалог выбора папки с кадрами. Аргументов не принимает.
  const pickFolder = () => dirRef.current && dirRef.current.click()
  return (
    <section className="ut-task" aria-label={t.title}>
      {(!single || src) && (
        <div className="ut-th">
          {!single && <b>{t.title}</b>}
          {src && <span className="ut-src" title={src}>{src}</span>}
        </div>
      )}
      <ol className="ut-steps">{steps.map((s) => <Step key={s.key} s={s} />)}</ol>
      {t.phase === 'awaiting' && t.folderError && <div className="errline show" role="alert">{t.folderError}</div>}
      {t.phase === 'failed' && t.error && <div className="errline show" role="alert">{t.error}</div>}
      <input
        ref={dirRef} type="file" hidden multiple webkitdirectory="" aria-hidden="true" tabIndex={-1}
        onChange={(e) => { chooseFolder(t.id, [...e.target.files]); e.target.value = '' }}
      />
      <div className="ut-act">
        {canCancel && <button className="ghost sm" onClick={() => cancel(t.id)}>Отменить</button>}
        {t.phase === 'awaiting' && t.folder && t.folder.missing > 0 && <button className="ghost sm" onClick={pickFolder}><Icon name="folder" /> Другая папка</button>}
        {t.phase === 'awaiting' && !t.folder && <button className="pri sm" onClick={pickFolder}><Icon name="folder" /> Выбрать папку</button>}
        {t.phase === 'awaiting' && t.folder && t.folder.missing > 0 && <button className="pri sm" onClick={() => confirmFolder(t.id)}><Icon name="upload" /> Загрузить {fmtInt(t.folder.matched)}</button>}
        {canRetry && <button className="pri sm" onClick={() => retry(t.id)}><Icon name="refresh" /> {t.kind === 'import' && (t.failedAt === 'processing' || t.failedAt === 'start') ? 'Повторить обработку' : 'Повторить'}</button>}
        {t.phase === 'done' && t.kind === 'import' && page !== 'gallery' && <button className="pri sm" onClick={() => onNavigate('gallery')}><Icon name="db" /> Показать в галерее</button>}
        {!busy && <button className="ghost sm" onClick={() => dismiss(t.id)}>Скрыть</button>}
      </div>
    </section>
  )
}

// Отрисовывает трей загрузок в галерею, плавающий поверх всех экранов. Принимает: page — текущий экран, onNavigate — переход между экранами.
export default function UploadTray({ page, onNavigate }) {
  const { tasks, expandTick, dismissFinished } = useUploads()
  const [min, setMin] = useState(false)
  const [shown, setShown] = useState(tasks.length > 0)
  const [leaving, setLeaving] = useState(false)
  const last = useRef(tasks)
  if (tasks.length) last.current = tasks
  const view = tasks.length ? tasks : last.current

  const seen = useRef(expandTick)
  useEffect(() => {
    if (expandTick !== seen.current) { seen.current = expandTick; setMin(false) }
  }, [expandTick])

  useEffect(() => {
    if (tasks.length) { setShown(true); setLeaving(false); return undefined }
    if (!shown) return undefined
    setLeaving(true)
    const timer = setTimeout(() => { setShown(false); setLeaving(false) }, 240)
    return () => clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tasks.length])

  if (!shown) return null

  const sm = summary(view)
  const busy = sm.state === 'busy'
  const ind = busy && sm.pct === null
  const idle = !view.some(isBusy)
  const first = view.find(isBusy)
  const announce = `${sm.line1}. ${first ? phaseLine(first).split(' · ')[0] : sm.line2}`

  return (
    <aside className={`ut ${min ? 'min' : ''} ${leaving ? 'leaving' : ''}`} data-state={sm.state} aria-label="Загрузки в галерею">
      <div className="ut-hd">
        <div className={ind ? 'ut-fill ind' : 'ut-fill'} aria-hidden="true"><i style={{ '--p': sm.pct ?? 0 }} /></div>
        <button className="ut-main" onClick={() => setMin((m) => !m)} aria-expanded={!min} aria-controls="ut-body">
          <span className="ut-ic" aria-hidden="true">
            {sm.state === 'done' ? <Icon name="check" size={18} /> : sm.state === 'failed' ? <Icon name="alert" size={18} /> : sm.state === 'await' ? <Icon name="folder" size={18} /> : <Icon name="upload" size={18} />}
          </span>
          <span className="ut-tx"><b>{sm.line1}</b><span>{sm.line2}</span></span>
          {busy && sm.pct !== null && <span className="ut-pc">{sm.pct}%</span>}
          <span className="ut-chev" aria-hidden="true"><Icon name="chevd" /></span>
        </button>
        {idle && <button className="ut-x ico" onClick={dismissFinished} aria-label="Скрыть завершённые загрузки" title="Скрыть"><Icon name="close" /></button>}
      </div>
      <div className="ut-body" id="ut-body" inert={min ? '' : undefined}>
        <div className="ut-in">
          <div className="ut-scroll">
            {view.map((t) => <Task key={t.id} t={t} single={view.length === 1} page={page} onNavigate={onNavigate} />)}
          </div>
        </div>
      </div>
      <div className="vh" role="status" aria-live="polite">{announce}</div>
    </aside>
  )
}
