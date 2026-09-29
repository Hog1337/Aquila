import { useEffect, useLayoutEffect, useRef } from 'react'
import { readStored, writeStored } from './persist.js'

const KEY = 'scroll'
const GIVE_UP_MS = 4000
const USER_EVENTS = ['wheel', 'touchstart', 'keydown', 'mousedown']

// Прокручивает окно к заданной позиции, повторяя попытки, пока высота страницы меняется. Принимает: y — целевая позиция прокрутки, done — колбэк, вызываемый один раз по завершении.
function restoreScroll(y, done) {
  let over = false
  const ro = new ResizeObserver(() => apply())
  const timer = setTimeout(() => finish(), GIVE_UP_MS)
  const finish = () => {
    if (over) return
    over = true
    ro.disconnect()
    clearTimeout(timer)
    USER_EVENTS.forEach((e) => window.removeEventListener(e, finish))
    done()
  }
  function apply() {
    if (over) return
    window.scrollTo(0, y)
    if (Math.abs(window.scrollY - y) < 2) finish()
  }
  USER_EVENTS.forEach((e) => window.addEventListener(e, finish, { passive: true }))
  ro.observe(document.body)
  apply()
  return finish
}

// Запоминает и восстанавливает позицию прокрутки отдельно для каждого экрана. Принимает: page — идентификатор текущего экрана.
export function useScrollMemory(page) {
  const pos = useRef(null)
  if (pos.current === null) {
    const s = readStored(KEY, {})
    pos.current = s && typeof s === 'object' ? s : {}
  }
  const pageRef = useRef(page)
  const restoring = useRef(false)
  const saveTimer = useRef(null)

  useEffect(() => {
    const prev = window.history.scrollRestoration
    window.history.scrollRestoration = 'manual'
    const flush = () => { clearTimeout(saveTimer.current); writeStored(KEY, pos.current) }
    const onScroll = () => {
      if (restoring.current) return
      pos.current[pageRef.current] = Math.round(window.scrollY)
      clearTimeout(saveTimer.current)
      saveTimer.current = setTimeout(flush, 200)
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('pagehide', flush)
    return () => {
      window.history.scrollRestoration = prev
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('pagehide', flush)
      flush()
    }
  }, [])

  useLayoutEffect(() => {
    pageRef.current = page
    restoring.current = true
    return restoreScroll(pos.current[page] || 0, () => { restoring.current = false })
  }, [page])
}
