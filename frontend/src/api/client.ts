import axios from 'axios'
import type { CalendarEvent, StudentProfile, Task } from '../types'

const BASE_URL = '/api'

export const api = axios.create({ baseURL: BASE_URL })

// Session
export const createSession = () => api.post<{ session_id: string }>('/session')

// Profile
export const getProfile = (sessionId: string) =>
  api.get<StudentProfile>(`/profile/${sessionId}`)

export const updateProfile = (sessionId: string, profile: Partial<StudentProfile>) =>
  api.put(`/profile/${sessionId}`, { profile })

// Calendar
export const getCalendar = (sessionId: string) =>
  api.get<{ events: CalendarEvent[] }>(`/calendar/${sessionId}`)

export const addCalendarEvent = (
  sessionId: string,
  event: {
    title: string
    start_datetime: string
    end_datetime: string
    description?: string
    course?: string
  },
) => api.post(`/calendar/${sessionId}/events`, event)

export const deleteCalendarEvent = (sessionId: string, eventId: string) =>
  api.delete(`/calendar/${sessionId}/events/${eventId}`)

// Tasks
export const getTasks = (sessionId: string) =>
  api.get<{ tasks: Task[] }>(`/tasks/${sessionId}`)

export const createTask = (
  sessionId: string,
  task: {
    title: string
    description?: string
    due_date?: string
    priority?: string
    course?: string
  },
) => api.post(`/tasks/${sessionId}`, task)

export const updateTask = (
  sessionId: string,
  taskId: string,
  updates: Partial<Task>,
) => api.put(`/tasks/${sessionId}/${taskId}`, updates)

export const deleteTask = (sessionId: string, taskId: string) =>
  api.delete(`/tasks/${sessionId}/${taskId}`)

// History
export const getHistory = (sessionId: string) =>
  api.get<{ messages: Array<{ role: string; content: string }> }>(`/history/${sessionId}`)

export const clearHistory = (sessionId: string) =>
  api.delete(`/history/${sessionId}`)

// Streaming chat — returns an EventSource
export const streamChat = (message: string, sessionId: string): EventSource => {
  // SSE requires GET or POST. We POST via fetch + ReadableStream (manual SSE parsing)
  return new EventSource(`/api/chat/stream?_placeholder=1`)
}

// Manual SSE via fetch for POST requests
export async function* streamChatFetch(
  message: string,
  sessionId: string,
): AsyncGenerator<{ event: string; data: Record<string, unknown> }> {
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId }),
  })

  if (!response.body) throw new Error('No response body')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  // Declared OUTSIDE the while loop so state persists across chunk boundaries
  let currentEvent = ''
  let currentData = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      // Trim \r to handle both \n and \r\n line endings
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
