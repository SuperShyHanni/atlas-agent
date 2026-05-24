import axios from 'axios'
import type { CalendarEvent, StudentProfile, Task } from '../types'

const BASE_URL = '/api'

export const api = axios.create({ baseURL: BASE_URL })

// Inject auth token on every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('atlas_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// ── Auth ──────────────────────────────────────────────────────────────────────

export const register = (email: string, password: string) =>
  api.post<{ token: string; user_id: string; email: string }>('/auth/register', {
    email,
    password,
  })

export const login = (email: string, password: string) =>
  api.post<{ token: string; user_id: string; email: string }>('/auth/login', {
    email,
    password,
  })

export const getMe = () => api.get<{ id: string; email: string }>('/auth/me')

// ── Profile ───────────────────────────────────────────────────────────────────

export const getProfile = () => api.get<StudentProfile>('/profile')

export const updateProfile = (profile: Partial<StudentProfile>) =>
  api.put('/profile', { profile })

// ── Calendar ──────────────────────────────────────────────────────────────────

export const getCalendar = () => api.get<{ events: CalendarEvent[] }>('/calendar')

export const addCalendarEvent = (event: {
  title: string
  start_datetime: string
  end_datetime: string
  description?: string
  course?: string
}) => api.post('/calendar/events', event)

export const deleteCalendarEvent = (eventId: string) =>
  api.delete(`/calendar/events/${eventId}`)

// ── Tasks ─────────────────────────────────────────────────────────────────────

export const getTasks = () => api.get<{ tasks: Task[] }>('/tasks')

export const createTask = (task: {
  title: string
  description?: string
  due_date?: string
  priority?: string
  course?: string
}) => api.post('/tasks', task)

export const updateTask = (taskId: string, updates: Partial<Task>) =>
  api.put(`/tasks/${taskId}`, updates)

export const deleteTask = (taskId: string) => api.delete(`/tasks/${taskId}`)

// ── History ───────────────────────────────────────────────────────────────────

export const getHistory = () =>
  api.get<{ messages: Array<{ role: string; content: string }> }>('/history')

export const clearHistory = () => api.delete('/history')

// ── Streaming Chat ────────────────────────────────────────────────────────────

export async function* streamChatFetch(
  message: string,
): AsyncGenerator<{ event: string; data: Record<string, unknown> }> {
  const token = localStorage.getItem('atlas_token') ?? ''

  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({ message }),
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(`${response.status}: ${text}`)
  }

  if (!response.body) throw new Error('No response body')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let currentEvent = ''
  let currentData = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.replace(/\r$/, '')

      if (trimmed.startsWith('event: ')) {
        currentEvent = trimmed.slice(7).trim()
      } else if (trimmed.startsWith('data: ')) {
        currentData = trimmed.slice(6).trim()
      } else if (trimmed === '' && currentEvent && currentData) {
        try {
          yield { event: currentEvent, data: JSON.parse(currentData) }
        } catch {
          yield { event: currentEvent, data: { raw: currentData } }
        }
        currentEvent = ''
        currentData = ''
      }
    }
  }
}
