import { useEffect, useRef, useState } from 'react'
import Icon from '../components/Icon.jsx'
import { useThreshold } from '../threshold.jsx'
import { api, ApiError, download, urls } from '../lib/api.js'
import { useAsync, useJob, isActive } from '../lib/hooks.js'
import { ARTIFACTS } from '../lib/config.js'
import { f2, fmtBytes, fmtInt, fmtDateTime } from '../lib/format.js'

const CONCURRENCY = 4
const VEKT_POLL_MS = 1000
const RETRIES = 3

// Матчинг файлов папки с image_id
function matchFolder(imageIds, files) {
  const maps = { name: new Map(), stem: new Map(), lname: new Map(), lstem: new Map() }
  for (const f of files) {
    if (!/\.(jpe?g|png)$/i.test(f.name)) continue
    const stem = f.name.replace(/\.[^.]+$/, '')
    const put = (m, k) => { if (!m.has(k)) m.set(k, f) }
    put(maps.name, f.name); put(maps.stem, stem)
    put(maps.lname, f.name.toLowerCase()); put(maps.lstem, stem.toLowerCase())
  }
  const matched = [], unmatched = [], imageIdMap = {}
  for (const id of imageIds) {
    const low = id.toLowerCase()
    const f = maps.name.get(id) || maps.stem.get(id) || maps.lname.get(low) || maps.lstem.get(low)
    if (f) { matched.push(f); imageIdMap[f.name] = id }
    else unmatched.push(id)
  }
  return { matched, unmatched, bytes: matched.reduce((s, f) => s + f.size, 0), imageIdMap }
}

// Иконка шага — как в UploadTray
function StepIcon({ state }) {
  if (state === 'done') return (
    <svg className="ut-si" width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
      <circle cx="10" cy="10" r="9" fill="var(--accent)" />
      <path className="ut-tick" d="M6 10.4 8.8 13.1 14 7.1" pathLength="1" fill="none" stroke="var(--accent-ink)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
  if (state === 'failed') return (
    <svg className="ut-si" width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
      <circle cx="10" cy="10" r="9" fill="var(--hot)" />
      <path d="M7 7l6 6M13 7l-6 6" fill="none" stroke="var(--accent-ink)" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
  return (
    <svg className="ut-si" width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
      <circle cx="10" cy="10" r="8" fill="none" stroke={state === 'await' ? 'var(--accent-2)' : 'var(--line-strong)'} strokeWidth="1.6" />
      {state === 'await' && <circle className="ut-beat" cx="10" cy="10" r="3" fill="var(--accent-2)" />}
      {state === 'active' && <circle className="ut-spin" cx="10" cy="10" r="8" fill="none" stroke="var(--accent-2)" strokeWidth="1.8" strokeLinecap="round" strokeDasharray="13 38" />}
    </svg>
  )
}

function Step({ label, sub, state, pct, children }) {
  const ind = state === 'active' && pct === null
  return (
    <li className={`ut-step ${state}`}>
      <StepIcon state={state} />
      <div className="ut-sb">
        <div className="ut-sl">
          <span className="ut-sn" title={label}>{label}</span>
          <span className="ut-spc">{state === 'active' && pct !== null ? `${pct}%` : ''}</span>
        </div>
        <div className="ut-ss">{sub}</div>
        <div className={ind ? 'ut-bar ind' : 'ut-bar'} role="progressbar" aria-label={label}
          aria-valuemin="0" aria-valuemax="100" aria-valuenow={ind ? undefined : pct}>
          <i style={{ '--p': pct ?? 0 }} />
        </div>
        {children && <div className="ut-act" style={{ marginTop: 6 }}>{children}</div>}
      </div>
    </li>
  )
}

// Флайер загрузки для экспорта — как MetricsUploadPanel
function ExportUploadPanel({ onStart, onClose }) {
  const queryRef = useRef(null)
  const galleryRef = useRef(null)
  const dirRef = useRef(null)
  const [queryFile, setQueryFile] = useState(null)
  const [galleryFile, setGalleryFile] = useState(null)
  const [err, setErr] = useState('')
  const [phase, setPhase] = useState('select') // select → checking → missing | ready → running → done
  const [imagesCheck, setImagesCheck] = useState(null)
  const [folderFiles, setFolderFiles] = useState(null)
  const [folderMatch, setFolderMatch] = useState(null)
  const [uploadPhase, setUploadPhase] = useState('idle') // idle | picked | uploading | vect | done
  const [uploadProgress, setUploadProgress] = useState({ loaded: 0, total: 0, speed: 0 })
  const [vectStatus, setVectStatus] = useState(null)

  const allFilesReady = !!queryFile

  // Автопроверка изображений после выбора CSV
  useEffect(() => {
    if (!allFilesReady) { setImagesCheck(null); return }
    let live = true
    setImagesCheck('checking')
    setErr('')
    api.checkImages(queryFile, galleryFile).then((result) => {
      if (!live) return
      setImagesCheck(result)
      setPhase(result.all_present ? 'ready' : 'missing')
    }).catch((e) => {
      if (!live) return
      setImagesCheck(null)
      setErr(`Ошибка проверки: ${e.message}`)
    })
    return () => { live = false }
  }, [queryFile, galleryFile])

  // Поллинг векторизации для уже присутствующих в S3 изображений
  useEffect(() => {
    if (!imagesCheck || typeof imagesCheck !== 'object') return
    if (!imagesCheck.all_present) return
    if (imagesCheck.with_vectors >= imagesCheck.total) { setVectStatus(null); return }
    let live = true
    const ids = Object.keys(imagesCheck.all_bbox || {})
    setVectStatus({ total: ids.length, ready: imagesCheck.with_vectors || 0, processing: 0, uploaded: 0, failed: 0 })
    ;(async () => {
      while (live) {
        await new Promise((r) => setTimeout(r, VEKT_POLL_MS))
        if (!live) return
        try {
          const status = await api.getProcessingStatus(ids)
          if (live) { setVectStatus(status); if (status.ready + status.failed >= status.total) { setVectStatus(null); return } }
        } catch (e) {}
      }
    })()
    return () => { live = false }
  }, [imagesCheck])

  const pickFolder = () => dirRef.current && dirRef.current.click()

  const onPickFolder = (e) => {
    const files = [...e.target.files]; e.target.value = ''
    setFolderFiles(files)
    const ids = imagesCheck && typeof imagesCheck === 'object' ? imagesCheck.missing : []
    const m = matchFolder(ids, files)
    setFolderMatch(m)
    setUploadPhase('picked')
  }

  // Параллельная загрузка + векторизация
  const uploadFiles = async () => {
    if (!folderMatch || folderMatch.matched.length === 0) return
    const abort = new AbortController()
    const { signal } = abort
    setUploadPhase('uploading'); setUploadProgress({ loaded: 0, total: folderMatch.bytes, speed: 0 }); setVectStatus(null)

    const idMap = folderMatch.imageIdMap || {}
    const bboxMap = imagesCheck && typeof imagesCheck === 'object' ? (imagesCheck.missing_bbox || {}) : {}
    const vehicleIds = imagesCheck && typeof imagesCheck === 'object' ? (imagesCheck.missing_vehicle_id || {}) : {}
    const items = folderMatch.matched.map((file) => {
      const stem = idMap[file.name] || file.name.replace(/\.[^.]+$/, '')
      return { image_id: stem, file, bbox: bboxMap[stem] || null, vehicle_id: vehicleIds[stem] || null }
    })
    const allImageIds = items.map((i) => i.image_id)

    let donePoller = false
    const pollVect = (async () => {
      while (!signal.aborted) {
        await new Promise((r) => setTimeout(r, VEKT_POLL_MS))
        if (signal.aborted || donePoller) return
        try { const s = await api.getProcessingStatus(allImageIds); setVectStatus(s) } catch (e) {}
      }
    })()

    const live = new Map(); const failures = []; let doneBytes = 0
    const sample = { t: performance.now(), loaded: 0, speed: 0 }
    const updateProgress = () => {
      let loaded = doneBytes; live.forEach((v) => { loaded += v })
      const now = performance.now(); const dt = (now - sample.t) / 1000
      if (dt > 0) { const inst = (loaded - sample.loaded) / dt; sample.speed = sample.speed ? sample.speed * 0.7 + inst * 0.3 : inst }
      sample.t = now; sample.loaded = loaded
      setUploadProgress({ loaded: Math.min(loaded, folderMatch.bytes), total: folderMatch.bytes, speed: sample.speed })
    }
    const progTimer = setInterval(updateProgress, 250)

    let next = 0
    const worker = async () => {
      while (!signal.aborted) {
        const item = items[next++]; if (!item) return
        for (let attempt = 0; attempt < RETRIES && !signal.aborted; attempt++) {
          try {
            await api.uploadSingleImage(item.image_id, item.file, item.bbox, (l) => live.set(item.image_id, l), item.vehicle_id)
            doneBytes += item.file.size; break
          } catch (e) {
            if (e.name === 'AbortError') return
            live.delete(item.image_id)
            if (e.status >= 400 && e.status < 500 && e.status !== 408 && e.status !== 429) break
            if (attempt < RETRIES - 1) await new Promise((r) => setTimeout(r, 700 * (attempt + 1)))
          }
        }
        live.delete(item.image_id)
      }
    }
    await Promise.all(Array.from({ length: CONCURRENCY }, worker))
    clearInterval(progTimer); updateProgress()
    if (failures.length) { setUploadPhase('idle'); return }

    setUploadPhase('vect')
    while (!signal.aborted) {
      await new Promise((r) => setTimeout(r, VEKT_POLL_MS))
      try { const s = await api.getProcessingStatus(allImageIds); setVectStatus(s); if (s.ready + s.failed >= s.total) { donePoller = true; break } } catch (e) {}
    }
    setUploadPhase('done')
    // Перепроверяем
    api.checkImages(queryFile, galleryFile).then((r) => { setImagesCheck(r); setPhase(r.all_present ? 'ready' : 'missing') })
  }

  // Статус шага изображений
  let imgState = 'pending'; let imgSub = 'Ожидание файлов'
  if (imagesCheck === 'checking') { imgState = 'active'; imgSub = 'Проверка наличия...' }
  else if (imagesCheck && typeof imagesCheck === 'object') {
    if (imagesCheck.all_present) { imgState = 'done'; imgSub = `Все ${imagesCheck.total} изображений в хранилище` }
    else { imgState = 'await'; imgSub = `${(imagesCheck.missing || []).length} из ${imagesCheck.total} отсутствуют` }
  }

  // Запуск экспорта
  const startExport = async () => {
    setErr('')
    if (!queryFile) { setErr('Выберите CSV'); return }
    setPhase('running')
    try {
      const j = await api.startExport(queryFile, th, rerank, galleryFile)
      onStart(j)
    } catch (e) {
      setErr(e.message)
      setPhase('ready')
    }
  }

  // Используем threshold из контекста
  const { th } = useThreshold()
  const [rerank, setRerank] = useState(true)

  const uploadPct = uploadProgress.total > 0 ? Math.floor((uploadProgress.loaded / uploadProgress.total) * 100) : 0
  const vectPct = vectStatus && (vectStatus.tracked || vectStatus.total) > 0 ? Math.floor(((vectStatus.ready + vectStatus.failed) / (vectStatus.tracked || vectStatus.total)) * 100) : 0
  const isUploading = uploadPhase === 'uploading' || uploadPhase === 'vect'

  return (
    <aside className="ut metrics-upload" data-state={phase === 'done' ? 'done' : phase === 'missing' ? 'await' : 'busy'} aria-label="Загрузка CSV для экспорта">
      <div className="ut-hd">
        <div className="ut-fill" aria-hidden="true"><i style={{ '--p': 0 }} /></div>
        <div className="ut-main">
          <span className="ut-ic" aria-hidden="true">
            {phase === 'done' ? <Icon name="check" size={18} /> : phase === 'missing' ? <Icon name="folder" size={18} /> : <Icon name="upload" size={18} />}
          </span>
          <span className="ut-tx"><b>Загрузка CSV для экспорта</b><span>{allFilesReady ? 'Файлы выбраны' : 'Выберите CSV'}</span></span>
          <button className="ut-x ico" onClick={onClose}><Icon name="close" /></button>
        </div>
      </div>
      <div className="ut-body">
        <div className="ut-in">
          <div className="ut-scroll">
            <section className="ut-task">
              <ol className="ut-steps">
                <Step label="CSV запросов" sub={queryFile ? `${queryFile.name} · ${fmtBytes(queryFile.size)}` : 'image_id, x, y, w, h'}
                  state={queryFile ? 'done' : 'pending'}>
                  {!queryFile && <button className="ghost sm" onClick={() => queryRef.current.click()}><Icon name="folder" /> Выбрать</button>}
                </Step>

                <Step label="CSV галереи" sub={galleryFile ? `${galleryFile.name} · ${fmtBytes(galleryFile.size)}` : 'image_id, x, y, w, h (опционально)'}
                  state={galleryFile ? 'done' : 'pending'}>
                  {queryFile && !galleryFile && <button className="ghost sm" onClick={() => galleryRef.current.click()}><Icon name="folder" /> Выбрать</button>}
                </Step>

                <Step label="Изображения" sub={imgSub} state={imgState}>
                  {phase === 'missing' && uploadPhase === 'idle' && (
                    <button className="pri sm" onClick={pickFolder}><Icon name="folder" /> Выбрать папку с images/</button>
                  )}
                  {uploadPhase === 'picked' && folderMatch && (
                    <>
                      <div className="ut-ss" style={{ marginBottom: 4 }}>
                        Найдено {folderMatch.matched.length} из {(imagesCheck && typeof imagesCheck === 'object' && imagesCheck.missing) ? imagesCheck.missing.length : 0}
                        {folderMatch.unmatched.length > 0 ? ` · нет в папке: ${folderMatch.unmatched.slice(0, 3).join(', ')}...` : ''}
                      </div>
                      <div className="tools" style={{ gap: 6 }}>
                        <button className="ghost sm" onClick={pickFolder}><Icon name="folder" /> Другая папка</button>
                        <button className="pri sm" disabled={!folderMatch.matched.length} onClick={uploadFiles}>
                          <Icon name="upload" /> Загрузить {folderMatch.matched.length}
                        </button>
                      </div>
                    </>
                  )}
                  {isUploading && (
                    <>
                      <div className="ut-sl" style={{ fontSize: 12, marginTop: 4 }}>
                        <span className="ut-sn">Загрузка</span><span className="ut-spc">{uploadPct}%</span>
                      </div>
                      <div className="ut-ss">{fmtBytes(uploadProgress.loaded)} из {fmtBytes(uploadProgress.total)}{uploadProgress.speed > 1024 ? ` · ${(uploadProgress.speed / 1024 / 1024).toFixed(1)} МБ/с` : ''}</div>
                      <div className="ut-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow={uploadPct}><i style={{ '--p': uploadPct }} /></div>
                      {vectStatus && (
                        <div style={{ marginTop: 8 }}>
                          <div className="ut-sl" style={{ fontSize: 12 }}><span className="ut-sn">Векторизация</span><span className="ut-spc">{vectPct}%</span></div>
                          <div className="ut-ss">{fmtInt(vectStatus.ready + vectStatus.failed)} из {fmtInt((vectStatus.tracked || vectStatus.total))}{vectStatus.failed > 0 ? ` · ${vectStatus.failed} с ошибкой` : ''}</div>
                          <div className="ut-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow={vectPct}><i style={{ '--p': vectPct }} /></div>
                        </div>
                      )}
                    </>
                  )}
                  {uploadPhase === 'done' && (
                    <div className="ut-ss" style={{ color: 'var(--accent-2)' }}>Загружено и завекторизовано</div>
                  )}
                </Step>

                <Step label="Формирование файлов"
                  sub={phase === 'ready' && vectStatus ? `Ожидание векторизации: ${fmtInt(vectStatus.ready + vectStatus.failed)} из ${fmtInt((vectStatus.tracked || vectStatus.total))}` : phase === 'ready' ? 'Готово' : phase === 'running' ? 'Выполняется...' : 'Ожидание файлов'}
                  state={phase === 'running' ? 'active' : phase === 'ready' && vectStatus ? 'active' : 'pending'}
                  pct={phase === 'running' ? 50 : undefined}>
                  {vectStatus && (
                    <div className="ut-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow={vectPct}><i style={{ '--p': vectPct }} /></div>
                  )}
                  {phase === 'ready' && !vectStatus && (
                    <div className="tools" style={{ gap: 6 }}>
                      <label className="toggle" style={{ fontSize: 12 }}>
                        <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} />
                        <span>Re-ranking</span>
                      </label>
                      <button className="pri sm" onClick={startExport}><Icon name="play" /> Сформировать</button>
                    </div>
                  )}
                </Step>
              </ol>

              <input ref={queryRef} type="file" accept=".csv" hidden onChange={(e) => { setQueryFile(e.target.files[0]); e.target.value = '' }} />
              <input ref={galleryRef} type="file" accept=".csv" hidden onChange={(e) => { setGalleryFile(e.target.files[0]); e.target.value = '' }} />
              <input ref={dirRef} type="file" hidden multiple webkitdirectory="" onChange={onPickFolder} />
              {err && <div className="errline show" role="alert" style={{ marginTop: 8 }}>{err}</div>}
            </section>
          </div>
        </div>
      </div>
    </aside>
  )
}

// Основная страница экспорта
export default function ExportPage({ onNavigate }) {
  const { th } = useThreshold()
  const [job, setJob] = useJob(api.getExport)
  const [err, setErr] = useState('')
  const [showPanel, setShowPanel] = useState(false)
  const last = useAsync((signal) => api.listExports(1, signal), [])

  useEffect(() => {
    const j = last.data && last.data.items && last.data.items[0]
    if (j) setJob((cur) => cur || j)
  }, [last.data, setJob])

  const onStart = (newJob) => { setShowPanel(false); setJob(newJob) }

  const done = job && job.status === 'done'
  const files = new Map((done && job.artifacts ? job.artifacts : []).map((a) => [a.name, a]))
  const staleThreshold = done && th !== null && job.threshold !== null && Math.abs(job.threshold - th) > 0.005
  const stale = staleThreshold

  return (
    <div className="panel">
      <div className="xhead">
        <div>
          <div className="k">Порог отказа</div>
          <div className="xv">{f2(th)} <button className="link" onClick={() => onNavigate('metrics')}>изменить</button></div>
        </div>
        <span className="sp" />
        <button className="pri" onClick={() => setShowPanel((v) => !v)} disabled={isActive(job)}>
          <Icon name="upload" /> {showPanel ? 'Закрыть' : 'Загрузить CSV'}
        </button>
      </div>

      {showPanel && <ExportUploadPanel onStart={onStart} onClose={() => setShowPanel(false)} />}

      {err && <div className="errline show" role="alert">{err}</div>}
      {(last.error) && <div className="errline show" role="alert">{last.error.message}</div>}

      {stale && (
        <div className="banner warn" role="status">
          <Icon name="alert" size={18} />
          <div className="grow">
            <b>Файлы устарели</b>
            <div className="hint" style={{ color: 'inherit', opacity: 0.85 }}>
              {staleThreshold ? `Сформированы при пороге ${f2(job.threshold)}, сейчас порог ${f2(th)}. ` : ''}
            </div>
          </div>
          <button className="pri" disabled={isActive(job)} onClick={() => setShowPanel(true)}><Icon name="refresh" /> Сформировать заново</button>
        </div>
      )}

      {job && !done && (
        <div className="job">
          <div className={job.status === 'failed' ? 'err' : undefined} style={{ minWidth: 150, fontWeight: 600 }}>
            {job.status === 'failed' ? `Ошибка: ${job.error || 'экспорт не выполнен'}` : `Формирование ${job.progress ?? 0}%`}
          </div>
          {isActive(job) && <div className="pbar" role="progressbar" aria-label="Формирование файлов" aria-valuemin="0" aria-valuemax="100" aria-valuenow={job.progress ?? 0}><i style={{ width: `${job.progress ?? 0}%` }} /></div>}
        </div>
      )}

      <div className="hd" style={{ margin: '4px 0 10px' }}>
        <h2>Файлы для сдачи</h2>
        {done && <span className={`chip ${stale ? 'warn' : 'ok'}`}>{stale ? `порог ${f2(job.threshold)}` : 'актуально'}</span>}
        <span className="sp" />
        {done && <span className="meta">{fmtInt(job.row_count)} запросов · {fmtDateTime(job.finished_at)}</span>}
      </div>

      {ARTIFACTS.map(([name, desc]) => {
        const file = files.get(name)
        return (
          <div className={`art ${stale ? 'stale' : ''}`} key={name}>
            <div className="ic"><Icon name="exp" size={19} /></div>
            <div><div className="aname">{name}</div><div className="hint">{desc}</div></div>
            {file ? (
              <div className="afile"><span className="hint">{fmtBytes(file.bytes)}</span><button className="ghost" onClick={() => download(urls.exportFile(job.id, name))}><Icon name="dl" size={14} /> Скачать</button></div>
            ) : (
              <div className="hint">не создан</div>
            )}
          </div>
        )
      })}
    </div>
  )
}
