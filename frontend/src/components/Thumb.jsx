import { useEffect, useState } from 'react'

// Отрисовывает миниатюру из API, показывая пустую рамку без ссылки или при ошибке загрузки. Принимает: src — URL изображения, alt — альтернативный текст.
export default function Thumb({ src, alt = '' }) {
  const [bad, setBad] = useState(false)
  useEffect(() => setBad(false), [src])
  if (!src || bad) return <div className="noimg" aria-hidden="true" />
  return <img className="im" src={src} alt={alt} loading="lazy" decoding="async" onError={() => setBad(true)} />
}
