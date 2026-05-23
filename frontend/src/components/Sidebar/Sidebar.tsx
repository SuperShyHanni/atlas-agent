import { useState } from 'react'
import { User, Calendar, CheckSquare, BookOpen } from 'lucide-react'
import ProfilePanel from '../Profile/ProfilePanel'
import CalendarPanel from '../Calendar/CalendarPanel'
import TasksPanel from '../Tasks/TasksPanel'

type Tab = 'profile' | 'calendar' | 'tasks'

const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
  { id: 'profile', label: 'Profile', icon: <User size={16} /> },
  { id: 'calendar', label: 'Calendar', icon: <Calendar size={16} /> },
  { id: 'tasks', label: 'Tasks', icon: <CheckSquare size={16} /> },
]

interface Props {
  sessionId: string
}

export default function Sidebar({ sessionId }: Props) {
  const [activeTab, setActiveTab] = useState<Tab>('profile')

  return (
    <div className="w-72 bg-white border-r border-gray-200 flex flex-col flex-shrink-0">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <BookOpen size={20} className="text-indigo-600" />
          <div>
            <h1 className="font-bold text-gray-900 text-base leading-tight">ATLAS</h1>
            <p className="text-xs text-gray-500">Academic Learning Agent</p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-200">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 flex flex-col items-center gap-0.5 py-2.5 text-xs font-medium transition-colors border-b-2 ${
              activeTab === tab.id
                ? 'border-indigo-600 text-indigo-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden p-3">
        {activeTab === 'profile' && <ProfilePanel sessionId={sessionId} />}
        {activeTab === 'calendar' && <CalendarPanel sessionId={sessionId} />}
        {activeTab === 'tasks' && <TasksPanel sessionId={sessionId} />}
      </div>
    </div>
  )
}
