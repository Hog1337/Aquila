import { useCallback, useEffect, useRef, useState } from 'react'
import { useUploads } from '../lib/uploads.jsx'
import Icon from './Icon.jsx'
import { IMAGE_TYPES, MAX_UPLOAD_MB, MIN_BOX_PX } from '../lib/config.js'
import { clearFile, loadFile, saveFile } from '../lib/persist.js'

const FILE_KEY = 'add-object'

// Ограничивает число диапазоном 0–1.
const clamp01 = (v) => Math.min(1, Math.max(0, v))

// Измеряет натуральные размеры изображения по его URL.
function measure(url) {
  return new Promise((resolve, reject) => {
    const im = new Image()
    im.onload = () => resolve({ w: im.naturalWidth, h: im.naturalHeight })
    im.onerror = () => reject(new Error('Не удалось прочитать изображение'))
    im.src = url
  })
}

// Диалог добавления одного автомобиля в галерею: кадр, рамка BBox (рисуется на изображении) и vehicle_id.
export default function AddObjectDialog({ open, onClose }) {
  const uploads = useUploads()
  const ref = useRef(null)
  const stageRef = useRef(null)
  const fileRef = useRef(null)
  const drag = useRef(null)

  const [file, setFile] = useState(null)
  const [image, setImage] = useState(null)       // { url, w, h }
  const [box, setBox] = useState(null)            // { l, t, r, b } — доли кадра
  const [vehicleId, setVehicleId] = useState('')
  const [err, setErr] = useState('')
  const [over, setOver] = useState(false)
  const [initialized, setInitialized] = useState(false)

  // Восстановление файла из IndexedDB при открытии
  useEffect(() => {
    let live = true
    loadFile(FILE_KEY).then((restored) => {
      if (!live) return
      if (restored) {
        applyFile(restored).catch(() => {})
      }
      setInitialized(true)
    })
    return () => { live = false }
  }, [])

  // Сохраняем файл при изменении
  useEffect(() => {
    if (!initialized) return
    if (file) saveFile(FILE_KEY, file)
    else clearFile(FILE_KEY)
  }, [file, initialized])

  // Загрузка изображения в стейт
  const applyFile = useCallback(async (f) => {
    const url = URL.createObjectURL(f)
    let dims
    try {
      dims = await measure(url)
    } catch (e) {
      URL.revokeObjectURL(url)
      throw e
    }
    setImage((prev) => {
      if (prev && prev.url && prev.url.startsWith('blob:')) URL.revokeObjectURL(prev.url)
      return { url, ...dims }
    })
    setBox(null)
    setErr('')
  }, [])

  // Очистка blob-URL при размонтировании
  useEffect(() => {
    return () => {
      if (image && image.url && image.url.startsWith('blob:')) URL.revokeObjectURL(image.url)
    }
  }, [])

  // Выбор файла
  const pickFile = useCallback((f) => {
    if (!f) return
    if (!IMAGE_TYPES.includes(f.type)) { setErr('Допустимы только JPEG и PNG'); return }
    if (f.size > MAX_UPLOAD_MB * 1024 * 1024) { setErr(`Файл больше ${MAX_UPLOAD_MB} МБ`); return }
    setErr('')
    setFile(f)
    applyFile(f).catch((e) => setErr(e.message))
  }, [applyFile])

  // Paste
  useEffect(() => {
    if (!open) return
    const onPaste = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
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
  }, [open, pickFile])

  // dialog open/close
  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) { setErr(''); d.showModal() }
    if (!open && d.open) d.close()
  }, [open])

  // Сброс
  const discard = useCallback(() => {
    setFile(null)
    setImage(null)
    setBox(null)
    setVehicleId('')
    setErr('')
    if (fileRef.current) fileRef.current.value = ''
  }, [])

  // ========== BBox drawing logic (как в SearchPage) ==========
  const W = image ? image.w : 0
  const H = image ? image.h : 0

  const point = (e) => {
    const r = stageRef.current.getBoundingClientRect()
    return { x: clamp01((e.clientX - r.left) / r.width), y: clamp01((e.clientY - r.top) / r.height) }
  }

  const onDown = (e) => {
    if (!image) return
    const p = point(e)
    setErr('')
    stageRef.current.setPointerCapture(e.pointerId)
    const h = e.target.dataset && e.target.dataset.h
    if (box && h) {
      // resize handle
      drag.current = { mode: 'resize', ax: h.includes('w') ? box.r : box.l, ay: h.includes('n') ? box.b : box.t }
    } else if (box && e.target.id === 'box') {
      // move
      drag.current = { mode: 'move', off: { dx: p.x - box.l, dy: p.y - box.t, w: box.r - box.l, h: box.b - box.t } }
    } else {
      // draw new
      drag.current = { mode: 'draw', ax: p.x, ay: p.y }
      setBox({ l: p.x, t: p.y, r: p.x, b: p.y })
    }
    e.preventDefault()
  }

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

  const onUp = () => {
    if (!drag.current) return
    drag.current = null
    setBox((b) => (b && ((b.r - b.l) * W < MIN_BOX_PX || (b.b - b.t) * H < MIN_BOX_PX) ? null : b))
  }

  const px = box && image && { l: Math.round(box.l * W), t: Math.round(box.t * H), w: Math.round((box.r - box.l) * W), h: Math.round((box.b - box.t) * H) }

  // ========== Submit ==========
  const submit = async (e) => {
    e.preventDefault()
    if (!file) { setErr('Выберите кадр JPEG или PNG'); return }
    if (!image) { setErr('Изображение не загружено'); return }
    if (!box || !px) { setErr('Выделите автомобиль рамкой на изображении'); return }
    if (px.w < MIN_BOX_PX || px.h < MIN_BOX_PX) { setErr(`Рамка слишком мала: минимум ${MIN_BOX_PX}×${MIN_BOX_PX} пикселей`); return }

    const form = new FormData()
    form.append('image', file)
    form.append('x', px.l)
    form.append('y', px.t)
    form.append('w', px.w)
    form.append('h', px.h)
    if (vehicleId.trim()) form.append('vehicle_id', vehicleId.trim())
    setErr('')
    uploads.addObject(form, file, vehicleId.trim())
    discard()
    onClose()
  }

  return (
    <dialog className="dlg" ref={ref} aria-labelledby="ao-t" onClose={onClose} onClick={(e) => { if (e.target === ref.current) onClose() }}>
      <form className="gc-in" onSubmit={submit}>
        <div className="hd"><h2 id="ao-t">Добавить ТС в галерею</h2></div>

        {/* Stage — загрузка и рисование bbox */}
        <div
          id="stage" ref={stageRef}
          className={`${image ? '' : 'drop-on'} ${over ? 'over' : ''}`}
          role="application" aria-label="Кадр. Обведите автомобиль рамкой"
          style={image ? {
            aspectRatio: `${image.w}/${image.h}`,
            width: `min(100%, calc(60vh * ${image.w / image.h}))`,
            margin: '0 auto'
          } : { cursor: 'default', minHeight: 200 }}
          onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}
          onDragOver={(e) => { e.preventDefault(); setOver(true) }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); pickFile(e.dataTransfer.files[0]) }}
        >
          {image && <img className="frame" src={image.url} alt="Кадр" draggable={false} />}
          {box && (
            <div id="box" style={{
              display: 'block',
              left: `${box.l * 100}%`, top: `${box.t * 100}%`,
              width: `${(box.r - box.l) * 100}%`, height: `${(box.b - box.t) * 100}%`
            }}>
              {['nw', 'ne', 'sw', 'se'].map((h) => <div key={h} className="hnd" data-h={h} />)}
            </div>
          )}
          {!image && (
            <div className="drop">
              <span className="drop-ic"><Icon name="upload" size={26} /></span>
              <div className="drop-t">Перетащите кадр сюда</div>
              <button className="pri" type="button" onClick={() => fileRef.current.click()}><Icon name="upload" /> Выбрать файл</button>
              <div className="hint">или вставьте из буфера: Ctrl+V · JPEG или PNG до {MAX_UPLOAD_MB} МБ</div>
            </div>
          )}
          {image && !box && <div id="emptyhint">Обведите автомобиль рамкой мышью</div>}
        </div>

        {/* Координаты bbox (readonly, заполняются из рамки) */}
        <div className="tools" style={{ marginTop: 8 }}>
          <div className="coords">
            {[['x', 'l'], ['y', 't'], ['w', 'w'], ['h', 'h']].map(([label, key]) => (
              <label className="coord" key={label}>
                <span>{label}</span>
                <input type="number" inputMode="numeric" value={px ? px[key] : ''} placeholder="—" disabled aria-label={`Координата ${label}, пиксели`} />
              </label>
            ))}
          </div>
          <span className="sp" />
          <button className="ghost" type="button" disabled={!box} onClick={() => setBox(null)}><Icon name="refresh" /> Сбросить</button>
        </div>

        {/* Vehicle ID */}
        <label className="fld" style={{ marginTop: 12 }}>
          <span>vehicle_id (необязательно)</span>
          <input type="text" value={vehicleId} onChange={(e) => setVehicleId(e.target.value)} maxLength={64} />
        </label>

        <input ref={fileRef} type="file" accept={IMAGE_TYPES.join(',')} hidden
          onChange={(e) => { pickFile(e.target.files[0]); e.target.value = '' }} />

        <div className={err ? 'errline show' : 'errline'} role="alert">{err}</div>

        <div className="tools">
          {image && <button className="ghost" type="button" onClick={() => fileRef.current.click()}><Icon name="upload" /> Другой кадр</button>}
          <span className="sp" />
          <button type="button" className="ghost" onClick={() => { discard(); onClose() }}>Отмена</button>
          <button type="submit" className="pri">Добавить</button>
        </div>
      </form>
    </dialog>
  )
}
