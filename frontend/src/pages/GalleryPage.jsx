import { useEffect, useRef, useState } from 'react'
import Icon from '../components/Icon.jsx'
import Thumb from '../components/Thumb.jsx'
import Pager from '../components/Pager.jsx'
import AddObjectDialog from '../components/AddObjectDialog.jsx'
import { api, urls } from '../lib/api.js'
import { useAsync, useDebounced, useReloadOnActive } from '../lib/hooks.js'
import { useUploads } from '../lib/uploads.jsx'
import { usePersistentState } from '../lib/persist.js'
import { GALLERY_PAGE } from '../lib/config.js'
import { fmtInt } from '../lib/format.js'

// Экран галереи: список объектов с поиском и запуском импорта/добавления ТС. Принимает: active — активен ли экран сейчас.
export default function GalleryPage({ active }) {
  const [q, setQ] = usePersistentState('gallery.q', '', (v) => typeof v === 'string')
  const dq = useDebounced(q.trim())
  const [offset, setOffset] = usePersistentState('gallery.offset', 0, (v) => Number.isInteger(v) && v >= 0)
  const [adding, setAdding] = usePersistentState('gallery.adding', false, (v) => typeof v === 'boolean')
  const [importErr, setImportErr] = useState('')
  const [clearing, setClearing] = useState(false)
  const fileRef = useRef(null)

  const status = useAsync((signal) => api.status(signal), [])
  const list = useAsync(
    (signal) => api.listGallery({ vehicle_id: dq, limit: GALLERY_PAGE, offset }, signal),
    [dq, offset],
  )
  const uploads = useUploads()
  const { doneTick, importActive } = uploads
  const seenDone = useRef(doneTick)
  useEffect(() => {
    if (doneTick === seenDone.current) return
    seenDone.current = doneTick
    list.reload()
    status.reload()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [doneTick])

  useReloadOnActive(active, list.reload, status.reload)

  useEffect(() => {
    if (list.data && offset > 0 && offset >= list.data.total) setOffset(0)
  }, [list.data, offset, setOffset])

  // Меняет строку поиска и сбрасывает страницу. Принимает: v — новое значение поиска.
  const changeQuery = (v) => { setQ(v); setOffset(0) }

  // Проверяет выбранный файл и запускает импорт галереи по CSV. Принимает: files — список выбранных файлов.
  const startImport = (files) => {
    const csv = files[0]
    if (files.length !== 1 || !/\.csv$/i.test(csv.name)) { setImportErr('Выберите один CSV-файл с аннотациями'); return }
    setImportErr('')
    uploads.startImport(csv, 'Импорт CSV + папка')
  }

  const g = status.data && status.data.gallery
  const items = list.data ? list.data.items : []
  const total = list.data ? list.data.total : 0
  const filtered = dq !== ''

  // Человеческое описание состояния галереи
  const gallerySummary = (() => {
    if (!g) return 'Загрузка...'
    const parts = []
    if (g.objects > 0) parts.push(`${fmtInt(g.objects)} объектов`)
    else if (g.processing_pending > 0 || g.processing_active > 0) parts.push('нет объектов (обработка...)') 
    else if (g.processing_failed > 0) parts.push('нет объектов (ошибки ⚠️)')
    else parts.push('Галерея пуста')

    if (g.vectorized !== undefined && g.vectorized >= 0 && g.objects > 0) {
      if (g.vectorized === g.objects) parts.push('все завекторизованы ✅')
      else parts.push(`завекторизовано ${fmtInt(g.vectorized)} из ${fmtInt(g.objects)}`)
    }
    if (g.processing_pending > 0) parts.push(`ожидают обработки: ${fmtInt(g.processing_pending)}`)
    if (g.processing_active > 0) parts.push(`обрабатывается: ${fmtInt(g.processing_active)}`)
    if (g.processing_failed > 0) parts.push(`ошибок: ${fmtInt(g.processing_failed)} ⚠️`)
    if (g.vehicles > 0) parts.push(`${fmtInt(g.vehicles)} уникальных ТС`)
    return parts.join(' · ')
  })()

  return (
    <div className="panel">
      <div className="stats">
        <span style={{ fontSize: 13, maxWidth: 500, textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>{gallerySummary}</span>
        <span className="sp" />
        <span className="mono">{status.data ? `${status.data.model_version} · вектор ${status.data.embedding_dim}` : '—'}</span>
      </div>
      {status.error && <div className="errline show" role="alert" style={{ marginTop: 12 }}>{status.error.message}</div>}
      <div className="tools" style={{ margin: '14px 0 14px' }}>
        <label className="sf">
          <Icon name="search" />
          <input type="text" placeholder="Поиск по vehicle_id" value={q} onChange={(e) => changeQuery(e.target.value)} aria-label="Поиск по vehicle_id" />
        </label>
        <span className="sp" />
        <button className="ghost" onClick={() => setAdding(true)}><Icon name="plus" /> Добавить ТС</button>
        <input ref={fileRef} type="file" accept=".csv" hidden onChange={(e) => { startImport([...e.target.files]); e.target.value = '' }} />
        <button className="pri" disabled={importActive} title={importActive ? 'Предыдущий импорт ещё выполняется' : undefined} onClick={() => fileRef.current.click()}><Icon name="upload" /> Импорт: CSV + папка</button>
        <button className="ghost" style={{ color: 'var(--hot)' }} disabled={clearing || !g || (g.objects === 0 && g.processing_pending === 0 && g.processing_active === 0)} onClick={async () => {
          if (!confirm('Очистить всю галерею? Это удалит все объекты и эмбеддинги.')) return
          setClearing(true)
          try { await api.clearGallery(); list.reload(); status.reload() } catch (e) { setImportErr(e.message) }
          setClearing(false)
        }}><Icon name="close" size={14} /> Очистить галерею</button>
      </div>
      {importErr && <div className="errline show" role="alert">{importErr}</div>}
      {list.error && <div className="errline show" role="alert">{list.error.message}</div>}
      <div className="gg">
        {!list.loading && !list.error && items.length === 0 && (
          <div className="empty" style={{ gridColumn: '1/-1' }}>
            {filtered ? 'Ничего не найдено — измените поиск' : 'Галерея пуста. Импортируйте CSV с папкой кадров или добавьте ТС'}
          </div>
        )}
        {items.map((o) => (
          <div className="gi" key={o.id}>
            <button className="gi-del" title="Удалить объект" onClick={async () => {
              if (!confirm(`Удалить объект ${o.image_id}?`)) return
              try {
                await api.deleteObject(o.id)
                list.reload()
                status.reload()
              } catch (e) {
                setImportErr(e.message)
              }
            }}><Icon name="close" size={12} /></button>
            <div className="ph p43">
              <Thumb src={urls.objectCrop(o.id, 256)} />
            </div>
            <div className={`vid ${o.vehicle_id ? '' : 'none'}`}>{o.vehicle_id || 'id неизвестен'}</div>
            <div className="fname">{o.image_id}</div>
          </div>
        ))}
      </div>
      <Pager offset={offset} limit={GALLERY_PAGE} total={total} onChange={setOffset} />
      <AddObjectDialog open={adding} onClose={() => setAdding(false)} />
    </div>
  )
}
