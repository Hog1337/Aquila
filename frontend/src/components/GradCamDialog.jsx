import { memo, useEffect, useRef } from 'react'
import Thumb from './Thumb.jsx'
import { api, urls } from '../lib/api.js'
import { useAsync } from '../lib/hooks.js'
import { f2 } from '../lib/format.js'
import { usePersistentState } from '../lib/persist.js'

// Отдельное окно Grad-CAM при сравнении двух автомобилей. Принимает: open — открыто ли окно, openKey — ключ повторного показа, onClose — обработчик закрытия, queryId — id запроса, cand — кандидат для сравнения.
function GradCamDialog({ open, openKey, onClose, queryId, cand }) {
  const ref = useRef(null)
  const [opacity, setOpacity] = usePersistentState('gradcam.opacity', 80, (v) => Number.isFinite(v) && v >= 0 && v <= 100)
  const gc = useAsync(
    (signal) => (open ? api.gradcam(queryId, cand.rank, signal) : Promise.resolve(null)),
    [open, queryId, cand.rank],
  )

  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) d.showModal()
    if (!open && d.open) d.close()
  }, [open, openKey])

  const data = gc.data
  const regions = data && data.regions ? data.regions : []

  return (
    <dialog
      className="dlg"
      ref={ref}
      aria-labelledby="gc-t"
      onClose={onClose}
      onClick={(e) => { if (e.target === ref.current) onClose() }}
    >
      {open && <div className="gc-in">
        <div className="hd">
          <h2 id="gc-t">CLS Attention · область внимания модели</h2>
          <span className="sp" />
          <button className="ghost" onClick={onClose} aria-label="Закрыть окно">Закрыть</button>
        </div>
        {gc.error && <div className="errline show" role="alert">{gc.error.message}</div>}
        <div className="gc-grid">
          <div className="gc-ph">
            <div className="ph" style={{ aspectRatio: '4/3' }}>
              <Thumb src={urls.queryCrop(queryId)} alt="Запрос" />
              {data && data.query_overlay_url && <img className="heat" src={data.query_overlay_url} alt="" style={{ opacity: opacity / 100 }} />}
            </div>
            <div className="gc-lab">Запрос</div>
          </div>
          <div className="gc-ph">
            <div className="ph" style={{ aspectRatio: '4/3' }}>
              <Thumb src={cand.object_id ? urls.objectCrop(cand.object_id) : null} alt="Кандидат" />
              {data && data.candidate_overlay_url && <img className="heat" src={data.candidate_overlay_url} alt="" style={{ opacity: opacity / 100 }} />}
            </div>
            <div className="gc-lab">Кандидат #{cand.rank} · {cand.image_id} · сходство {f2(cand.score)}</div>
          </div>
        </div>
        {gc.loading && <p className="hint" style={{ marginTop: 14 }} aria-live="polite">Строится карта…</p>}
        <div className="gc-reg">
          <div>
            <div className="hint" style={{ marginBottom: 7 }}>Вклад областей в сходство</div>
            {regions.length === 0 && !gc.loading && <div className="hint">Backend не вернул разбивку по областям</div>}
            {regions.map(({ name, share }) => (
              <div className="rg" key={name}>
                <span>{name}</span><b>{Math.round(share * 100)}%</b>
                <div className="track"><i style={{ width: `${share * 100}%` }} /></div>
              </div>
            ))}
          </div>
          <div>
            <div className="hint" style={{ marginBottom: 7 }}>Яркость карты</div>
            <input type="range" min="0" max="100" value={opacity} onChange={(e) => setOpacity(+e.target.value)} style={{ width: '100%' }} aria-label="Яркость карты Grad-CAM" />
            <div className="scale" style={{ marginTop: 12 }} />
            <div className="tools" style={{ justifyContent: 'space-between', marginTop: 5 }}>
              <span className="hint">меньше влияние</span><span className="hint">больше влияние</span>
            </div>
            <p className="hint" style={{ marginTop: 16 }}>
              CLS self-attention последнего слоя DINOv3, усреднённая по 16 головам. Показывает, на какие области изображения модель обращает внимание (сильнее = важнее для эмбеддинга).
            </p>
          </div>
        </div>
      </div>}
    </dialog>
  )
}

export default memo(GradCamDialog)
