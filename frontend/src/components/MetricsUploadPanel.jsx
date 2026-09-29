import { useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../lib/api.js'
import { fmtBytes, fmtInt } from '../lib/format.js'
import Icon from './Icon.jsx'

const CONCURRENCY = 4
const VEKT_POLL_MS = 1000
const RETRIES = 3

// Матчинг как в галерее
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
  const queue = []
  for (const id of imageIds) {
    const low = id.toLowerCase()
    const f = maps.name.get(id) || maps.stem.get(id) || maps.lname.get(low) || maps.lstem.get(low)
    if (f) {
      matched.push(f)
      imageIdMap[f.name] = id
    }
    else unmatched.push(id)
  }
  return { matched, unmatched, bytes: matched.reduce((s, f) => s + f.size, 0), imageIdMap }
}

// Иконка состояния шага — идентичная UploadTray
function StepIcon({ state }) {
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

// Шаг задания — идентичный UploadTray
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
        <div
          className={ind ? 'ut-bar ind' : 'ut-bar'} role="progressbar" aria-label={label}
          aria-valuemin="0" aria-valuemax="100" aria-valuenow={ind ? undefined : pct}
        >
          <i style={{ '--p': pct ?? 0 }} />
        </div>
        {children && <div className="ut-act" style={{ marginTop: 6 }}>{children}</div>}
      </div>
    </li>
  )
}

// Плавающая панель загрузки CSV для оценки метрик — полный аналог UploadTray.
export default function MetricsUploadPanel({ onStart, onClose }) {
  const queryRef = useRef(null)
  const galleryRef = useRef(null)
  const gtRef = useRef(null)
  const dirRef = useRef(null)
  const [queryFile, setQueryFile] = useState(null)
  const [galleryFile, setGalleryFile] = useState(null)
  const [gtFile, setGtFile] = useState(null)
  const [rerank, setRerank] = useState(true)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [phase, setPhase] = useState('select')   // select → checking → missing | ready → running → done
  const [imagesCheck, setImagesCheck] = useState(null)
  const [folderFiles, setFolderFiles] = useState(null)
  const [folderMatch, setFolderMatch] = useState(null)
  const [uploadPhase, setUploadPhase] = useState('idle') // idle | picked | uploading | done | vect
  const [uploadProgress, setUploadProgress] = useState({ loaded: 0, total: 0, speed: 0 })
  const [vectStatus, setVectStatus] = useState(null)
  const [uploadResult, setUploadResult] = useState(null)
  const abortRef = useRef(null)

  const allFilesReady = queryFile && galleryFile && gtFile

  // Автопроверка изображений после выбора CSV
  useEffect(() => {
    if (!allFilesReady) { setImagesCheck(null); return }
    let live = true
    setImagesCheck('checking')
    setErr('')
    setVectStatus(null)
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
  }, [queryFile, galleryFile, gtFile])

  // Если изображения есть в S3, но не все завекторизованы — поллим статус
  useEffect(() => {
    if (!imagesCheck || typeof imagesCheck !== 'object') return
    if (!imagesCheck.all_present) return
    if (imagesCheck.with_vectors >= imagesCheck.total) {
      setVectStatus(null)
      return
    }
    let live = true
    const ids = Object.keys(imagesCheck.all_bbox || {})
    setVectStatus({ total: ids.length, ready: imagesCheck.with_vectors || 0, processing: 0, uploaded: 0, failed: 0 })
    ;(async () => {
      while (live) {
        await new Promise((r) => setTimeout(r, VEKT_POLL_MS))
        if (!live) return
        try {
          const status = await api.getProcessingStatus(ids)
          if (live) {
            setVectStatus(status)
            if (status.ready + status.failed >= status.total) {
              setVectStatus(null)
              return
            }
          }
        } catch (e) {}
      }
    })()
    return () => { live = false }
  }, [imagesCheck])

  const pickFolder = () => dirRef.current && dirRef.current.click()

  const onPickFolder = (e) => {
    const files = [...e.target.files]
    e.target.value = ''
    setFolderFiles(files)
    const m = matchFolder(imagesCheck ? imagesCheck.missing : [], files)
    setFolderMatch(m)
    setUploadPhase('picked')
    setUploadResult(null)
  }

  // Параллельная загрузка + поллинг векторизации
  const uploadFiles = async () => {
    if (!folderMatch || folderMatch.matched.length === 0) return

    const abort = new AbortController()
    abortRef.current = abort
    const { signal } = abort

    setUploadPhase('uploading')
    setUploadProgress({ loaded: 0, total: folderMatch.bytes, speed: 0 })
    setVectStatus(null)
    setUploadResult(null)

    const idMap = folderMatch.imageIdMap || {}
    const vehicleIds = imagesCheck && imagesCheck.missing_vehicle_id ? imagesCheck.missing_vehicle_id : {}
    const items = folderMatch.matched.map((file) => {
      const csvImageId = idMap[file.name]
      const stem = csvImageId || file.name.replace(/\.[^.]+$/, '')
      const bbox = imagesCheck && imagesCheck.missing_bbox ? imagesCheck.missing_bbox[stem] : null
      const vid = vehicleIds[stem] || null
      return { image_id: stem, file, bbox, vehicle_id: vid }
    })
    const allImageIds = items.map((i) => i.image_id)

    let donePoller = false
    const pollVect = (async () => {
      while (!signal.aborted) {
        await new Promise((r) => setTimeout(r, VEKT_POLL_MS))
        if (signal.aborted || donePoller) return
        try {
          const status = await api.getProcessingStatus(allImageIds)
          setVectStatus(status)
        } catch (e) {}
      }
    })()

    const live = new Map()
    const failures = []
    let doneBytes = 0
    const totalBytes = folderMatch.bytes
    const sample = { t: performance.now(), loaded: 0, speed: 0 }

    const updateProgress = () => {
      let loaded = doneBytes
      live.forEach((v) => { loaded += v })
      const now = performance.now()
      const dt = (now - sample.t) / 1000
      if (dt > 0) {
        const inst = (loaded - sample.loaded) / dt
        sample.speed = sample.speed ? sample.speed * 0.7 + inst * 0.3 : inst
      }
      sample.t = now; sample.loaded = loaded
      setUploadProgress({ loaded: Math.min(loaded, totalBytes), total: totalBytes, speed: sample.speed })
    }
    const progTimer = setInterval(updateProgress, 250)

    let next = 0
    const worker = async () => {
      while (!signal.aborted) {
        const item = items[next++]
        if (!item) return
        let ok = false
        for (let attempt = 0; attempt < RETRIES && !signal.aborted; attempt++) {
          try {
            await api.uploadSingleImage(item.image_id, item.file, item.bbox, (l) => live.set(item.image_id, l), item.vehicle_id)
            ok = true
            break
          } catch (e) {
            if (e.name === 'AbortError') return
            live.delete(item.image_id)
            if (e.status >= 400 && e.status < 500 && e.status !== 408 && e.status !== 429) break
            if (attempt < RETRIES - 1) await new Promise((r) => setTimeout(r, 700 * (attempt + 1)))
          }
        }
        live.delete(item.image_id)
        if (ok) {
          doneBytes += item.file.size
        } else if (!signal.aborted) {
          failures.push({ image_id: item.image_id, message: 'ошибка загрузки' })
        }
      }
    }

    await Promise.all(Array.from({ length: CONCURRENCY }, worker))
    clearInterval(progTimer)
    updateProgress()

    if (signal.aborted) return

    if (failures.length) {
      setUploadResult({ imported: items.length - failures.length, errors: failures.map((f) => f.message) })
      setUploadPhase('idle')
      return
    }

    setUploadPhase('vect')
    while (!signal.aborted) {
      await new Promise((r) => setTimeout(r, VEKT_POLL_MS))
      if (signal.aborted) return
      try {
        const status = await api.getProcessingStatus(allImageIds)
        setVectStatus(status)
        if (status.ready + status.failed >= status.total) {
          donePoller = true
          break
        }
      } catch (e) {}
    }

    if (signal.aborted) return

    setUploadResult({ imported: items.length, errors: [] })
    setUploadPhase('done')
    setImagesCheck('checking')
    api.checkImages(queryFile, galleryFile).then((r) => {
      setImagesCheck(r)
      setPhase(r.all_present ? 'ready' : 'missing')
      if (!r.all_present) {
        setUploadPhase('idle')
        setFolderFiles(null)
        setFolderMatch(null)
      }
    })
  }

  // Состояние шага изображений
  let imgState = 'pending'
  let imgSub = 'Ожидание файлов'
  if (imagesCheck === 'checking') {
    imgState = 'active'
    imgSub = 'Проверка наличия...'
  } else if (imagesCheck && typeof imagesCheck === 'object') {
    if (imagesCheck.all_present) {
      imgState = 'done'
      imgSub = `Все ${imagesCheck.total} изображений в хранилище`
    } else {
      imgState = 'await'
      imgSub = `${(imagesCheck.missing || []).length} из ${imagesCheck.total} отсутствуют`
    }
  }

  const start = async () => {
    setErr('')
    if (!allFilesReady) { setErr('Загрузите все три CSV'); return }
    if (imagesCheck && !imagesCheck.all_present) { setErr('Сначала загрузите изображения'); return }
    setBusy(true)
    setPhase('running')
    try {
      const job = await api.startMetricsRun(queryFile, galleryFile, gtFile, rerank)
      onStart(job)
    } catch (e) {
      setErr(e.message)
      setPhase('ready')
    } finally {
      setBusy(false)
    }
  }

  // Готовим pct для шагов
  const uploadPct = uploadProgress.total > 0 ? Math.floor((uploadProgress.loaded / uploadProgress.total) * 100) : 0
  const vectPct = vectStatus && (vectStatus.tracked || vectStatus.total) > 0 ? Math.floor(((vectStatus.ready + vectStatus.failed) / (vectStatus.tracked || vectStatus.total)) * 100) : 0
  const isUploading = uploadPhase === 'uploading' || uploadPhase === 'vect'

  return (
    <aside className="ut metrics-upload" data-state={phase === 'done' ? 'done' : phase === 'missing' ? 'await' : 'busy'} aria-label="Загрузка CSV для оценки">
      <div className="ut-hd">
        <div className="ut-fill" aria-hidden="true"><i style={{ '--p': 0 }} /></div>
        <div className="ut-main">
          <span className="ut-ic" aria-hidden="true">
            {phase === 'done' ? <Icon name="check" size={18} /> : phase === 'missing' ? <Icon name="folder" size={18} /> : <Icon name="upload" size={18} />}
          </span>
          <span className="ut-tx">
            <b>Загрузка CSV для оценки</b>
            <span>
              {phase === 'running' ? 'Запуск...' :
               phase === 'missing' ? 'Нужны изображения' :
               allFilesReady ? 'Все файлы выбраны' : 'Выберите файлы'}
            </span>
          </span>
          <button className="ut-x ico" onClick={onClose} aria-label="Закрыть"><Icon name="close" /></button>
        </div>
      </div>
      <div className="ut-body">
        <div className="ut-in">
          <div className="ut-scroll">
            <section className="ut-task">
              <ol className="ut-steps">
                {/* CSV запросов */}
                <Step label="CSV запросов"
                  sub={queryFile ? `${queryFile.name} · ${fmtBytes(queryFile.size)}` : 'image_id, x, y, w, h'}
                  state={queryFile ? 'done' : 'pending'}>
                  {!queryFile && <button className="ghost sm" onClick={() => queryRef.current.click()}><Icon name="folder" /> Выбрать</button>}
                </Step>

                {/* CSV галереи */}
                <Step label="CSV галереи"
                  sub={galleryFile ? `${galleryFile.name} · ${fmtBytes(galleryFile.size)}` : 'image_id, x, y, w, h'}
                  state={galleryFile ? 'done' : 'pending'}>
                  {queryFile && !galleryFile && <button className="ghost sm" onClick={() => galleryRef.current.click()}><Icon name="folder" /> Выбрать</button>}
                </Step>

                {/* GT CSV */}
                <Step label="GT CSV"
                  sub={gtFile ? `${gtFile.name} · ${fmtBytes(gtFile.size)}` : 'image_id, vehicle_id, camera_id, split'}
                  state={gtFile ? 'done' : 'pending'}>
                  {queryFile && galleryFile && !gtFile && <button className="ghost sm" onClick={() => gtRef.current.click()}><Icon name="folder" /> Выбрать</button>}
                </Step>

                {/* Изображения — с двумя прогресс-барами (загрузка + векторизация) */}
                <Step label="Изображения" sub={imgSub} state={imgState}>
                  {phase === 'missing' && uploadPhase === 'idle' && (
                    <button className="pri sm" onClick={pickFolder}><Icon name="folder" /> Выбрать папку с images/</button>
                  )}
                  {uploadPhase === 'picked' && folderMatch && (
                    <>
                      <div className="ut-ss" style={{ marginBottom: 4 }}>
                        Найдено {folderMatch.matched.length} из {(imagesCheck && imagesCheck.missing) ? imagesCheck.missing.length : 0}
                        {folderMatch.unmatched.length > 0 && ` · нет в папке: ${folderMatch.unmatched.slice(0, 3).join(', ')}${folderMatch.unmatched.length > 3 ? '...' : ''}`}
                      </div>
                      <div className="tools" style={{ gap: 6 }}>
                        <button className="ghost sm" onClick={pickFolder}><Icon name="folder" /> Другая папка</button>
                        <button className="pri sm" disabled={folderMatch.matched.length === 0} onClick={uploadFiles}>
                          <Icon name="upload" /> Загрузить {folderMatch.matched.length}
                        </button>
                      </div>
                    </>
                  )}

                  {isUploading && (
                    <>
                      {/* Прогресс-бар загрузки */}
                      <div className="ut-sl" style={{ fontSize: 12, marginTop: 4 }}>
                        <span className="ut-sn">Загрузка</span>
                        <span className="ut-spc">{uploadPct}%</span>
                      </div>
                      <div className="ut-ss">
                        {fmtBytes(uploadProgress.loaded)} из {fmtBytes(uploadProgress.total)}
                        {uploadProgress.speed > 1024 ? ` · ${(uploadProgress.speed / 1024 / 1024).toFixed(1)} МБ/с` : ''}
                      </div>
                      <div className="ut-bar" role="progressbar" aria-label="Загрузка изображений"
                        aria-valuemin="0" aria-valuemax="100" aria-valuenow={uploadPct}>
                        <i style={{ '--p': uploadPct }} />
                      </div>

                      {/* Прогресс-бар векторизации */}
                      {vectStatus && (
                        <div style={{ marginTop: 8 }}>
                          <div className="ut-sl" style={{ fontSize: 12 }}>
                            <span className="ut-sn">Векторизация</span>
                            <span className="ut-spc">{vectPct}%</span>
                          </div>
                          <div className="ut-ss">
                            {fmtInt(vectStatus.ready + vectStatus.failed)} из {fmtInt((vectStatus.tracked || vectStatus.total))}
                            {vectStatus.failed > 0 ? ` · ${vectStatus.failed} с ошибкой` : ''}
                          </div>
                          <div className="ut-bar" role="progressbar" aria-label="Векторизация"
                            aria-valuemin="0" aria-valuemax="100" aria-valuenow={vectPct}>
                            <i style={{ '--p': vectPct }} />
                          </div>
                        </div>
                      )}
                    </>
                  )}

                  {uploadPhase === 'done' && uploadResult && (
                    <div className="ut-ss" style={{ color: 'var(--accent-2)' }}>
                      Загружено и завекторизовано: {uploadResult.imported}
                      {vectStatus && vectStatus.failed > 0 && (
                        <span style={{ color: 'var(--hot)', marginLeft: 8 }}>{vectStatus.failed} с ошибкой</span>
                      )}
                    </div>
                  )}
                </Step>

                {/* Запуск оценки */}
                <Step label="Запуск оценки"
                  sub={
                    phase === 'ready' && vectStatus
                      ? `Ожидание векторизации: ${fmtInt(vectStatus.ready + vectStatus.failed)} из ${fmtInt((vectStatus.tracked || vectStatus.total))}`
                      : phase === 'ready'
                      ? 'Готов к запуску'
                      : phase === 'running'
                      ? 'Выполняется...'
                      : phase === 'missing'
                      ? 'Ожидание изображений'
                      : 'Ожидание файлов'
                  }
                  state={
                    phase === 'running' ? 'active'
                    : phase === 'ready' && vectStatus ? 'active'
                    : 'pending'
                  }
                  pct={phase === 'running' ? 50 : (phase === 'ready' && vectStatus ? vectPct : undefined)}>
                  {vectStatus && (
                    <div className="ut-bar" role="progressbar" aria-label="Векторизация"
                      aria-valuemin="0" aria-valuemax="100" aria-valuenow={vectPct}>
                      <i style={{ '--p': vectPct }} />
                    </div>
                  )}
                  {allFilesReady && imagesCheck && imagesCheck.all_present && phase === 'ready' && !vectStatus && (
                    <div className="tools" style={{ gap: 6 }}>
                      <label className="toggle" style={{ fontSize: 12 }}>
                        <input type="checkbox" checked={rerank} onChange={(e) => setRerank(e.target.checked)} />
                        <span>Re-ranking</span>
                      </label>
                      <button className="pri sm" disabled={busy} onClick={start}>
                        {busy ? 'Запуск...' : <><Icon name="play" /> Запустить</>}
                      </button>
                    </div>
                  )}
                </Step>
              </ol>

              <input ref={queryRef} type="file" accept=".csv" hidden onChange={(e) => { setQueryFile(e.target.files[0]); e.target.value = '' }} />
              <input ref={galleryRef} type="file" accept=".csv" hidden onChange={(e) => { setGalleryFile(e.target.files[0]); e.target.value = '' }} />
              <input ref={gtRef} type="file" accept=".csv" hidden onChange={(e) => { setGtFile(e.target.files[0]); e.target.value = '' }} />
              <input ref={dirRef} type="file" hidden multiple webkitdirectory="" onChange={onPickFolder} />

              {err && <div className="errline show" role="alert" style={{ marginTop: 8 }}>{err}</div>}
            </section>
          </div>
        </div>
      </div>
    </aside>
  )
}
