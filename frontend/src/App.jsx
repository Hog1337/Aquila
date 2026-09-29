import { useEffect, useRef, useState } from 'react'
import Icon from './components/Icon.jsx'
import { ThresholdProvider, useThreshold } from './threshold.jsx'
import { UploadProvider } from './lib/uploads.jsx'
import UploadTray from './components/UploadTray.jsx'
import { PAGES } from './lib/config.js'
import { readStored, writeStored } from './lib/persist.js'
import { useScrollMemory } from './lib/scroll.js'
import { f2 } from './lib/format.js'
import SearchPage from './pages/SearchPage.jsx'
import GalleryPage from './pages/GalleryPage.jsx'
import MetricsPage from './pages/MetricsPage.jsx'
import HistoryPage from './pages/HistoryPage.jsx'
import ExportPage from './pages/ExportPage.jsx'

const VIEWS = { search: SearchPage, gallery: GalleryPage, metrics: MetricsPage, history: HistoryPage, export: ExportPage }

// Проверяет, относится ли переданный id к одному из известных экранов. Принимает: id — идентификатор экрана (string).
const isPage = (id) => PAGES.some(([p]) => p === id)
// Определяет экран для показа по hash в адресе или по последнему сохранённому экрану. Аргументов не принимает.
const pageFromUrl = () => {
  const h = window.location.hash.replace(/^#\/?/, '')
  if (isPage(h)) return h
  const last = readStored('page', null)
  return isPage(last) ? last : 'search'
}

// Показывает баннер с ошибкой загрузки единого порога отказа. Аргументов не принимает.
function ThresholdError() {
  const { error } = useThreshold()
  if (!error) return null
  return <div className="errline show" role="alert" style={{ marginBottom: 14 }}>Не удалось получить порог отказа: {error}</div>
}

// Показывает текущий порог отказа в верхней панели. Принимает: onOpen — обработчик клика по чипу.
function ThresholdChip({ onOpen }) {
  const { th, dirty } = useThreshold()
  return (
    <button className="thchip" onClick={onOpen} title={dirty ? 'Порог выбран, но не сохранён в системе. Сохраняется на экране «Метрики и порог»' : 'Порог отказа. Меняется на экране «Метрики и порог»'}>
      <span>Порог</span><b>{f2(th)}</b>{dirty && <em className="thdirty">не сохранён</em>}
    </button>
  )
}

// Корневой компонент приложения: управляет текущим экраном, адресной строкой и хранилищем. Аргументов не принимает.
export default function App() {
  const [page, setPage] = useState(pageFromUrl)
  const [visited, setVisited] = useState(() => new Set(['search', pageFromUrl()]))
  const [restore, setRestore] = useState(null)
  // Переключает текущий экран и отмечает его как посещённый. Принимает: id — идентификатор экрана.
  const show = (id) => { setPage(id); setVisited((v) => (v.has(id) ? v : new Set(v).add(id))) }
  // Меняет hash в адресе на экран и переключает его. Принимает: id — идентификатор экрана.
  const open = (id) => { if (window.location.hash !== `#${id}`) window.location.hash = id; show(id) }
  // Открывает экран «Поиск» с запросом на восстановление из истории. Принимает: id — идентификатор запроса.
  const openQuery = (id) => { setRestore((r) => ({ id, n: r ? r.n + 1 : 1 })); open('search') }
  const navRef = useRef(null)
  useScrollMemory(page)

  useEffect(() => {
    if (window.location.hash !== `#${page}`) window.history.replaceState(null, '', `#${page}`)
    writeStored('page', page)
  }, [page])
  useEffect(() => {
    const onHash = () => show(pageFromUrl())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    const track = navRef.current
    const tab = track && track.querySelector('.tab.on')
    if (!tab) return
    const trackBox = track.getBoundingClientRect()
    const tabBox = tab.getBoundingClientRect()
    if (tabBox.left < trackBox.left || tabBox.right > trackBox.right) track.scrollTo({ left: tab.offsetLeft - 24 })
  }, [page])

  return (
    <ThresholdProvider>
      <UploadProvider>
      <div className="shell">
        <nav className="topbar" aria-label="Разделы" ref={navRef}>
          <div className="brand">Wallcreepers</div>
          {PAGES.map(([id, icon, label]) => (
            <button
              key={id} className={`tab ${page === id ? 'on' : ''}`}
              aria-current={page === id ? 'page' : undefined} onClick={() => open(id)}
            >
              <Icon name={icon} />{label}
            </button>
          ))}
          <ThresholdChip onOpen={() => open('metrics')} />
          <a className="apilink" href="/docs" target="_blank" rel="noreferrer"><Icon name="api" />API · Swagger</a>
        </nav>

        <ThresholdError />

        <main>
          <h1 className="vh">{(PAGES.find(([id]) => id === page) || PAGES[0])[2]}</h1>
          {PAGES.map(([id]) => {
            if (!visited.has(id)) return null
            const View = VIEWS[id]
            return <div key={id} className={`page ${page === id ? 'on' : ''}`} id={`p-${id}`}><View restore={restore} active={page === id} onOpenQuery={openQuery} onNavigate={open} /></div>
          })}
        </main>
      </div>
      <UploadTray page={page} onNavigate={open} />
      </UploadProvider>
    </ThresholdProvider>
  )
}
