// Отрисовывает сегментированный переключатель в виде пилюли. Принимает: options — список [id, текст], value — текущий выбранный id, onChange — обработчик выбора, label — подпись для доступности.
export default function Seg({ options, value, onChange, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map(([id, text]) => (
        <button key={id} className={id === value ? 'on' : ''} aria-pressed={id === value} onClick={() => onChange(id)}>
          {text}
        </button>
      ))}
    </div>
  )
}
