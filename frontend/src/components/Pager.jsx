import Icon from './Icon.jsx'

// Отрисовывает постраничную навигацию по offset/limit. Принимает: offset — смещение текущей страницы, limit — размер страницы, total — общее число элементов, onChange — обработчик смены offset.
export default function Pager({ offset, limit, total, onChange }) {
  const from = total ? offset + 1 : 0
  const to = Math.min(offset + limit, total)
  if (total === 0 || (total <= limit && offset === 0)) return total ? <div className="hint" style={{ marginTop: 14 }}>Показано {total} из {total.toLocaleString('ru-RU')}</div> : null
  return (
    <div className="tools" style={{ marginTop: 14 }}>
      <span className="hint">{total ? `Показано ${from}–${to} из ${total.toLocaleString('ru-RU')}` : ''}</span>
      <span className="sp" />
      <button className="ghost ico" aria-label="Назад" disabled={offset <= 0} onClick={() => onChange(Math.max(0, offset - limit))}><Icon name="chevl" /></button>
      <button className="ghost ico" aria-label="Вперёд" disabled={offset + limit >= total} onClick={() => onChange(offset + limit)}><Icon name="chevr" /></button>
    </div>
  )
}
