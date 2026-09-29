const ICONS = {
  search: <><circle cx="7" cy="7" r="4.6" /><path d="M10.5 10.5 14 14" /></>,
  db: <><ellipse cx="8" cy="4" rx="5.5" ry="2" /><path d="M2.5 4v9c0 1.1 2.46 2 5.5 2s5.5-.9 5.5-2V4M2.5 8.5c0 1.1 2.46 2 5.5 2s5.5-.9 5.5-2" /></>,
  chart: <path d="M2.5 15.5h13M4.5 15V9M8.5 15V5M12.5 15v-7" />,
  hist: <><circle cx="8" cy="9" r="6" /><path d="M8 6v3l2 1.5M3 3l-1 2" /></>,
  exp: <><path d="M4 2.5h6l4 4v9h-10z" /><path d="M10 2.5V7h4M8 9v5M6 12l2 2 2-2" /></>,
  api: <><rect x="2" y="4" width="14" height="10" rx="1.5" /><path d="M5.5 8v2M8 7v3M10.5 8v2M13 7v3" /></>,
  refresh: <path d="M14 8a6 6 0 1 1-1.8-4.3M14 2.5V6h-3.5" />,
  upload: <path d="M8 12V3M4.5 6.5 8 3l3.5 3.5M3 14h10" />,
  dl: <path d="M8 3v9M4.5 8.5 8 12l3.5-3.5M3 14h10" />,
  plus: <path d="M8 3v10M3 8h10" />,
  chevl: <path d="M10.5 3.5 6 8l4.5 4.5" />,
  chevr: <path d="M6.5 3.5 11 8l-4.5 4.5" />,
  play: <path d="M5 3.5v9l8-4.5z" />,
  alert: <><circle cx="8" cy="8" r="6.5" /><path d="M8 5v4" /><circle cx="8" cy="11" r=".2" fill="currentColor" stroke="none" /></>,
  info: <><circle cx="8" cy="8" r="6.5" /><path d="M8 7.5v3.5M8 5.2v.2" /></>,
  check: <path d="M3.5 8.5 6.6 11.6 12.5 4.6" />,
  close: <path d="M4 4l8 8M12 4l-8 8" />,
  chevd: <path d="M3.5 6 8 10.5 12.5 6" />,
  chevu: <path d="M3.5 10 8 5.5l4.5 4.5" />,
  file: <><path d="M4 1.8h5.2L12.5 5v9.2H4z" /><path d="M9 1.8V5.2h3.5" /></>,
  folder: <path d="M2 4.5a1 1 0 0 1 1-1h3.2l1.5 1.8H13a1 1 0 0 1 1 1V12a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1z" />,
  eye:<><path d="M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z" /><circle cx="8" cy="8" r="2" /></>,
}

// Отрисовывает SVG-иконку по имени. Принимает: name — ключ иконки в ICONS, size — размер в пикселях.
export default function Icon({ name, size = 16 }) {
  return (
    <svg className="i" width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
      {ICONS[name]}
    </svg>
  )
}
