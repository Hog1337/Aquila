import { useRef, useState } from 'react'
import Icon from './Icon.jsx'
import { api } from '../lib/api.js'
import { fmtBytes, fmtInt } from '../lib/format.js'

const CONCURRENCY = 4
const VEKT_POLL_MS = 1000
const RETRIES = 3

// Диалог загрузки недостающих изображений — параллельная загрузка + векторизация.
export default function MissingImagesDialog({ missing = [], total, queryCsv, galleryCsv, missingBbox = {}, onDone, onCancel }) {
  const dirRef = useRef(null)
  const [phase, setPhase] = useState('idle') // idle | picked | uploading | vect | done | error
  const [fileList, setFileList] = useState(null)
  const [match, setMatch] = useState(null)
  const [uploadProgress, setUploadProgress] = useState({ loaded: 0, total: 0, speed: 0 })
  const [vectStatus, setVectStatus] = useState(null)
  const [result, setResult] = useState(null)

  const pickFolder = () => dirRef.current && dirRef.current.click()

  // Матчинг как в галерее
  const onPickFolder = (e) => {
    const files = [...e.target.files]
    e.target.value = ''
    setFileList(files)

    const maps = { name: new Map(), stem: new Map(), lname: new Map(), lstem: new Map() }
    for (const f of files) {
      if (!/\.(jpe?g|png)$/i.test(f.name)) continue
      const stem = f.name.replace(/\.[^.]+$/, '')
      const put = (m, k) => { if (!m.has(k)) m.set(k, f) }
      put(maps.name, f.name); put(maps.stem, stem)
      put(maps.lname, f.name.toLowerCase()); put(maps.lstem, stem.toLowerCase())
    }

    const matched = []
    const unmatched = []
    const imageIdMap = {}
    for (const id of missing) {
      const low = id.toLowerCase()
      const f = maps.name.get(id) || maps.stem.get(id) || maps.lname.get(low) || maps.lstem.get(low)
      if (f) {
        matched.push(f)
        imageIdMap[f.name] = id
      } else unmatched.push(id)
    }

    setMatch({ found: matched.length, missing: unmatched, missingSample: unmatched.slice(0, 5), matchedFiles: matched, imageIdMap })
    setPhase('picked')
  }

  // Параллельная загрузка + векторизация
  const uploadAll = async () => {
    if (!match || !match.matchedFiles || match.matchedFiles.length === 0) return

    const abort = new AbortController()
    const { signal } = abort

    setPhase('uploading')
    setUploadProgress({ loaded: 0, total: 0, speed: 0 })
    setVectStatus(null)
    setResult(null)

    const idMap = match.imageIdMap || {}
    const items = match.matchedFiles.map((file) => {
      const csvImageId = idMap[file.name]
      const stem = csvImageId || file.name.replace(/\.[^.]+$/, '')
      const bbox = missingBbox[stem] || null
      return { image_id: stem, file, bbox }
    })
    const totalBytes = items.reduce((s, i) => s + i.file.size, 0)
    const allImageIds = items.map((i) => i.image_id)
    setUploadProgress((p) => ({ ...p, total: totalBytes }))

    // Поллинг векторизации параллельно с загрузкой
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

    // Воркеры загрузки
    const live = new Map()
    const failures = []
    let doneBytes = 0
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
            await api.uploadSingleImage(item.image_id, item.file, item.bbox, (l) => live.set(item.image_id, l))
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

    if (failures.length) {
      setResult({ imported: items.length - failures.length, errors: failures.map((f) => f.message) })
      setPhase('error')
      return
    }

    // Все загружено — ждём векторизацию
    setPhase('vect')
    while (!signal.aborted) {
      await new Promise((r) => setTimeout(r, VEKT_POLL_MS))
      try {
        const status = await api.getProcessingStatus(allImageIds)
        setVectStatus(status)
        if (status.ready + status.failed >= status.total) {
          donePoller = true
          break
        }
      } catch (e) {}
    }

    setResult({ imported: items.length, errors: [] })
    setPhase('done')
    setTimeout(() => onDone(), 800)
  }

  const totalBytes = fileList ? fileList.reduce((s, f) => s + f.size, 0) : 0
  const uploadPct = uploadProgress.total > 0 ? Math.floor((uploadProgress.loaded / uploadProgress.total) * 100) : 0
  const vectPct = vectStatus && (vectStatus.tracked || vectStatus.total) > 0 ? Math.floor(((vectStatus.ready + vectStatus.failed) / (vectStatus.tracked || vectStatus.total)) * 100) : 0
  const isUploading = phase === 'uploading' || phase === 'vect'

  return (
    <div className="dlg-overlay" onClick={onCancel}>
      <div className="dlg" style={{ padding: 22, maxWidth: 520 }} onClick={(e) => e.stopPropagation()}>
        <h2 style={{ margin: '0 0 12px' }}>Не хватает изображений</h2>

        {phase === 'idle' && (
          <>
            <p className="hint" style={{ marginBottom: 16 }}>
              {missing.length} из {total} image_id из CSV <b>отсутствуют в галерее</b>.
              Выберите папку с исходными кадрами, чтобы загрузить недостающие.
            </p>
            <p className="hint" style={{ marginBottom: 16, fontSize: 12, color: 'var(--ink-3)' }}>
              Система найдёт файлы по имени (image_id.jpg или image_id.png).
            </p>
            <button className="pri" onClick={pickFolder}><Icon name="folder" /> Выбрать папку с images/</button>
            <input ref={dirRef} type="file" hidden multiple webkitdirectory="" onChange={onPickFolder} />
          </>
        )}

        {phase === 'picked' && match && (
          <>
            <div style={{ marginBottom: 12 }}>
              <div className="sum3" style={{ marginBottom: 8 }}>
                <div className="tile"><div className="k">Всего нужно</div><div className="v">{total}</div></div>
                <div className="tile"><div className="k">Найдено в папке</div><div className="v" style={{ color: 'var(--accent-2)' }}>{match.found}</div></div>
                <div className="tile"><div className="k">Не найдено</div><div className="v" style={{ color: match.missing && match.missing.length > 0 ? 'var(--hot)' : undefined }}>{match.missing ? match.missing.length : 0}</div></div>
              </div>
              {match.missing && match.missing.length > 0 && (
                <div className="hint" style={{ color: 'var(--hot)', fontSize: 12 }}>
                  Нет в папке: {(match.missingSample || []).join(', ')}{match.missing.length > 5 ? ` и ещё ${match.missing.length - 5}` : ''}. Выберите другую папку или пропустите.
                </div>
              )}
              {match.found > 0 && (
                <div className="hint" style={{ fontSize: 12, marginTop: 4 }}>
                  Файлов в папке: {fileList ? fileList.length : 0} · {fmtBytes(totalBytes)}
                </div>
              )}
            </div>
            <div className="tools" style={{ marginTop: 8 }}>
              <button className="ghost" onClick={pickFolder}><Icon name="folder" /> Другая папка</button>
              <span className="sp" />
              <button className="ghost" onClick={onCancel}>Отмена</button>
              <button className="pri" disabled={match.found === 0} onClick={uploadAll}>
                <Icon name="upload" /> Загрузить {match.found}
              </button>
            </div>
            <input ref={dirRef} type="file" hidden multiple webkitdirectory="" onChange={onPickFolder} />
          </>
        )}

        {isUploading && (
          <div style={{ marginBottom: 12 }}>
            {/* Прогресс-бар загрузки */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 2 }}>
              <span style={{ fontSize: 13, fontWeight: 600 }}>Загрузка</span>
              <span style={{ fontFamily: 'var(--disp)', fontSize: 12.5, fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>{uploadPct}%</span>
            </div>
            <div className="ut-ss" style={{ marginBottom: 4 }}>
              {fmtBytes(uploadProgress.loaded)} из {fmtBytes(uploadProgress.total)}
              {uploadProgress.speed > 1024 ? ` · ${(uploadProgress.speed / 1024 / 1024).toFixed(1)} МБ/с` : ''}
            </div>
            <div className="ut-bar" role="progressbar" aria-label="Загрузка"
              aria-valuemin="0" aria-valuemax="100" aria-valuenow={uploadPct}>
              <i style={{ '--p': uploadPct }} />
            </div>

            {/* Прогресс-бар векторизации */}
            {vectStatus && (
              <div style={{ marginTop: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 2 }}>
                  <span style={{ fontSize: 13, fontWeight: 600 }}>Векторизация</span>
                  <span style={{ fontFamily: 'var(--disp)', fontSize: 12.5, fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>{vectPct}%</span>
                </div>
                <div className="ut-ss" style={{ marginBottom: 4 }}>
                  {fmtInt(vectStatus.ready + vectStatus.failed)} из {fmtInt((vectStatus.tracked || vectStatus.total))}
                  {vectStatus.failed > 0 ? ` · ${vectStatus.failed} с ошибкой` : ''}
                </div>
                <div className="ut-bar" role="progressbar" aria-label="Векторизация"
                  aria-valuemin="0" aria-valuemax="100" aria-valuenow={vectPct}>
                  <i style={{ '--p': vectPct }} />
                </div>
              </div>
            )}
          </div>
        )}

        {phase === 'done' && result && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--accent-2)' }}>
              <Icon name="check" size={18} />
              <b>Загружено и завекторизовано: {result.imported}</b>
            </div>
            <p className="hint" style={{ marginTop: 8 }}>Запускаем формирование файлов...</p>
          </div>
        )}

        {phase === 'error' && result && (
          <div style={{ marginBottom: 12 }}>
            <div className="errline show" role="alert">
              {result && result.errors && result.errors.length > 0
                ? `Ошибки (${(result.errors || []).length}): ${(result.errors || []).slice(0, 3).join(', ')}${(result.errors || []).length > 3 ? ` и ещё ${(result.errors || []).length - 3}` : ''}`
                : 'Не удалось загрузить изображения'}
            </div>
            {result.imported > 0 && (
              <p className="hint" style={{ marginTop: 8 }}>Загружено: {result.imported}</p>
            )}
            <div className="tools" style={{ marginTop: 12 }}>
              <button className="ghost" onClick={onCancel}>Закрыть</button>
              <button className="pri" onClick={uploadAll}>Повторить</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
