import { useEffect, useState } from 'react'
import { Plus, Trash2, CheckCircle, Circle } from 'lucide-react'
import { getTasks, createTask, updateTask, deleteTask } from '../../api/client'
import type { Task } from '../../types'

const PRIORITY_COLORS = {
  high: 'text-red-600 bg-red-50 border-red-200',
  medium: 'text-yellow-700 bg-yellow-50 border-yellow-200',
  low: 'text-green-700 bg-green-50 border-green-200',
}

interface Props {
  sessionId: string
}

export default function TasksPanel({ sessionId }: Props) {
  const [tasks, setTasks] = useState<Task[]>([])
  const [showForm, setShowForm] = useState(false)
  const [newTask, setNewTask] = useState({
    title: '',
    description: '',
    due_date: '',
    priority: 'medium',
    course: '',
  })

  const load = async () => {
    try {
      const res = await getTasks(sessionId)
      setTasks(res.data.tasks ?? [])
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    load()
  }, [sessionId])

  const handleCreate = async () => {
    if (!newTask.title.trim()) return
    await createTask(sessionId, newTask)
    setNewTask({ title: '', description: '', due_date: '', priority: 'medium', course: '' })
    setShowForm(false)
    load()
  }

  const handleToggle = async (task: Task) => {
    await updateTask(sessionId, task.id, { completed: !task.completed })
    load()
  }

  const handleDelete = async (id: string) => {
    await deleteTask(sessionId, id)
    load()
  }

  const pending = tasks.filter((t) => !t.completed)
  const done = tasks.filter((t) => t.completed)

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-700 text-sm">
          Tasks ({pending.length} pending)
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
            placeholder="Task title *"
            value={newTask.title}
            onChange={(e) => setNewTask((p) => ({ ...p, title: e.target.value }))}
            className="w-full text-sm border border-gray-200 rounded-lg px-3 py-1.5 outline-none focus:border-indigo-400"
          />
          <input
            placeholder="Course (optional)"
            value={newTask.course}
            onChange={(e) => setNewTask((p) => ({ ...p, course: e.target.value }))}
            className="w-full text-sm border border-gray-200 rounded-lg px-3 py-1.5 outline-none focus:border-indigo-400"
          />
          <div className="flex gap-2">
            <input
              type="date"
              value={newTask.due_date}
              onChange={(e) => setNewTask((p) => ({ ...p, due_date: e.target.value }))}
              className="flex-1 text-sm border border-gray-200 rounded-lg px-3 py-1.5 outline-none focus:border-indigo-400"
            />
            <select
              value={newTask.priority}
              onChange={(e) => setNewTask((p) => ({ ...p, priority: e.target.value }))}
              className="flex-1 text-sm border border-gray-200 rounded-lg px-2 py-1.5 outline-none focus:border-indigo-400"
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleCreate}
              className="flex-1 text-xs bg-indigo-600 text-white rounded-lg py-1.5 hover:bg-indigo-700"
            >
              Create
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
        {pending.map((task) => (
          <TaskItem
            key={task.id}
            task={task}
            onToggle={handleToggle}
            onDelete={handleDelete}
          />
        ))}

        {done.length > 0 && (
          <>
            <p className="text-xs text-gray-400 font-medium pt-2 pb-1">Completed</p>
            {done.map((task) => (
              <TaskItem
                key={task.id}
                task={task}
                onToggle={handleToggle}
                onDelete={handleDelete}
              />
            ))}
          </>
        )}

        {tasks.length === 0 && (
          <p className="text-xs text-gray-400 text-center py-4">No tasks yet</p>
        )}
      </div>
    </div>
  )
}

function TaskItem({
  task,
  onToggle,
  onDelete,
}: {
  task: Task
  onToggle: (t: Task) => void
  onDelete: (id: string) => void
}) {
  return (
    <div
      className={`flex items-start gap-2 p-2 rounded-lg border bg-white hover:shadow-sm transition-shadow ${
        task.completed ? 'opacity-50' : ''
      }`}
    >
      <button onClick={() => onToggle(task)} className="mt-0.5 flex-shrink-0">
        {task.completed ? (
          <CheckCircle size={16} className="text-green-500" />
        ) : (
          <Circle size={16} className="text-gray-400" />
        )}
      </button>
      <div className="flex-1 min-w-0">
        <p
          className={`text-xs font-medium truncate ${task.completed ? 'line-through text-gray-400' : 'text-gray-800'}`}
        >
          {task.title}
        </p>
        <div className="flex items-center gap-1 mt-0.5 flex-wrap">
          <span
            className={`text-xs border rounded-full px-1.5 py-0.5 ${PRIORITY_COLORS[task.priority as keyof typeof PRIORITY_COLORS] ?? PRIORITY_COLORS.medium}`}
          >
            {task.priority}
          </span>
          {task.due_date && (
            <span className="text-xs text-gray-400">{task.due_date}</span>
          )}
        </div>
      </div>
      <button
        onClick={() => onDelete(task.id)}
        className="text-gray-300 hover:text-red-400 flex-shrink-0 mt-0.5"
      >
        <Trash2 size={13} />
      </button>
    </div>
  )
}
