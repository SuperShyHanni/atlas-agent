import { useEffect, useState } from 'react'
import { Plus, Trash2, Calendar } from 'lucide-react'
import { getCalendar, addCalendarEvent, deleteCalendarEvent } from '../../api/client'
import type { CalendarEvent } from '../../types'

function formatDateTime(dt: string) {
  try {
    return new Date(dt).toLocaleString([], {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return dt
  }
}

interface Props {
  sessionId: string
}

export default function CalendarPanel({ sessionId }: Props) {
  const [events, setEvents] = useState<CalendarEvent[]>([])
  const [showForm, setShowForm] = useState(false)
  const [newEvent, setNewEvent] = useState({
    title: '',
    start_datetime: '',
    end_datetime: '',
    description: '',
    course: '',
  })

  const load = async () => {
    try {
      const res = await getCalendar(sessionId)
      const sorted = (res.data.events ?? []).sort(
        (a, b) =>
          new Date(a.start.dateTime).getTime() - new Date(b.start.dateTime).getTime(),
      )
      setEvents(sorted)
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    load()
  }, [sessionId])

  const handleCreate = async () => {
    if (!newEvent.title.trim() || !newEvent.start_datetime) return
    const endDt = newEvent.end_datetime || newEvent.start_datetime
    await addCalendarEvent(sessionId, { ...newEvent, end_datetime: endDt })
    setNewEvent({ title: '', start_datetime: '', end_datetime: '', description: '', course: '' })
    setShowForm(false)
    load()
  }

  const handleDelete = async (id: string) => {
    await deleteCalendarEvent(sessionId, id)
    load()
  }

  const now = new Date()
  const upcoming = events.filter((e) => new Date(e.start.dateTime) >= now)
  const past = events.filter((e) => new Date(e.start.dateTime) < now)

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-700 text-sm">
          Calendar ({upcoming.length} upcoming)
        </h3>
        <button
          onClick={() => setShowForm((v) => !v)}
          className="flex items-center gap-1 text-xs text-indigo-600 hover:text-indigo-800 font-medium"
        >
          <Plus size={14} />
          Add
        </button>
      </div>

      {showForm && (
        <div className="bg-indigo-50 rounded-xl p-3 mb-3 space-y-2 border border-indigo-100">
          <input
            placeholder="Event title *"
            value={newEvent.title}
            onChange={(e) => setNewEvent((p) => ({ ...p, title: e.target.value }))}
            className="w-full text-sm border border-gray-200 rounded-lg px-3 py-1.5 outline-none focus:border-indigo-400"
          />
          <input
            placeholder="Course (optional)"
            value={newEvent.course}
            onChange={(e) => setNewEvent((p) => ({ ...p, course: e.target.value }))}
            className="w-full text-sm border border-gray-200 rounded-lg px-3 py-1.5 outline-none focus:border-indigo-400"
          />
          <label className="text-xs text-gray-500">Start</label>
          <input
            type="datetime-local"
            value={newEvent.start_datetime}
            onChange={(e) => setNewEvent((p) => ({ ...p, start_datetime: e.target.value }))}
            className="w-full text-sm border border-gray-200 rounded-lg px-3 py-1.5 outline-none focus:border-indigo-400"
          />
          <label className="text-xs text-gray-500">End (optional)</label>
          <input
            type="datetime-local"
            value={newEvent.end_datetime}
            onChange={(e) => setNewEvent((p) => ({ ...p, end_datetime: e.target.value }))}
            className="w-full text-sm border border-gray-200 rounded-lg px-3 py-1.5 outline-none focus:border-indigo-400"
          />
          <div className="flex gap-2">
            <button
              onClick={handleCreate}
              className="flex-1 text-xs bg-indigo-600 text-white rounded-lg py-1.5 hover:bg-indigo-700"
            >
              Add Event
            </button>
            <button
              onClick={() => setShowForm(false)}
              className="flex-1 text-xs bg-gray-200 text-gray-700 rounded-lg py-1.5 hover:bg-gray-300"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto space-y-1.5">
        {upcoming.map((event) => (
          <EventItem key={event.id} event={event} onDelete={handleDelete} />
        ))}

        {past.length > 0 && (
          <>
            <p className="text-xs text-gray-400 font-medium pt-2 pb-1">Past events</p>
            {past.map((event) => (
              <EventItem key={event.id} event={event} onDelete={handleDelete} past />
            ))}
          </>
        )}

        {events.length === 0 && (
          <div className="flex flex-col items-center py-6 text-gray-400">
            <Calendar size={24} className="mb-2 opacity-40" />
            <p className="text-xs">No events scheduled</p>
          </div>
        )}
      </div>
    </div>
  )
}

function EventItem({
  event,
  onDelete,
  past,
}: {
  event: CalendarEvent
  onDelete: (id: string) => void
  past?: boolean
}) {
  return (
    <div
      className={`flex items-start gap-2 p-2 rounded-lg border bg-white hover:shadow-sm transition-shadow ${past ? 'opacity-50' : ''}`}
    >
      <div className="w-1.5 h-full min-h-[32px] bg-indigo-400 rounded-full flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <p className="text-xs font-medium text-gray-800 truncate">{event.title}</p>
        <p className="text-xs text-gray-400">{formatDateTime(event.start.dateTime)}</p>
        {event.course && (
          <span className="text-xs bg-gray-100 text-gray-600 rounded px-1">{event.course}</span>
        )}
      </div>
      <button
        onClick={() => onDelete(event.id)}
        className="text-gray-300 hover:text-red-400 flex-shrink-0 mt-0.5"
      >
        <Trash2 size={13} />
      </button>
    </div>
  )
}
