import { useEffect, useState } from 'react'
import AuthPage from './components/Auth/AuthPage'
import Sidebar from './components/Sidebar/Sidebar'
import Chat from './components/Chat/Chat'
import AgentStatusPanel from './components/AgentStatus/AgentStatus'
import type { AgentName, AgentStatus } from './types'

const INITIAL_AGENT_STATUSES: AgentStatus[] = [
  { name: 'coordinator', status: 'idle' },
  { name: 'planner', status: 'idle' },
  { name: 'notewriter', status: 'idle' },
  { name: 'advisor', status: 'idle' },
]

export default function App() {
  const [token, setToken] = useState<string | null>(null)
  const [userId, setUserId] = useState<string>('')
  const [agentStatuses, setAgentStatuses] = useState<AgentStatus[]>(INITIAL_AGENT_STATUSES)
  const [requiredAgents, setRequiredAgents] = useState<AgentName[]>([])
  const [isLoading, setIsLoading] = useState(false)

  // Restore auth from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem('atlas_token')
    const storedUserId = localStorage.getItem('atlas_user_id')
    if (stored && storedUserId) {
      setToken(stored)
      setUserId(storedUserId)
    }
  }, [])

  const handleAuth = (newToken: string, newUserId: string) => {
    localStorage.setItem('atlas_token', newToken)
    localStorage.setItem('atlas_user_id', newUserId)
    setToken(newToken)
    setUserId(newUserId)
  }

  const handleLogout = () => {
    localStorage.removeItem('atlas_token')
    localStorage.removeItem('atlas_user_id')
    setToken(null)
    setUserId('')
  }

  if (!token) {
    return <AuthPage onAuth={handleAuth} />
  }

  return (
    <div className="h-screen flex bg-gray-50 overflow-hidden">
      <Sidebar userId={userId} onLogout={handleLogout} />

      <div className="flex-1 flex flex-col min-w-0">
        <Chat
          onAgentStatusChange={setAgentStatuses}
          onRequiredAgentsChange={setRequiredAgents}
          onLoadingChange={setIsLoading}
        />
      </div>

      <AgentStatusPanel
        agents={agentStatuses}
        requiredAgents={requiredAgents}
        isLoading={isLoading}
      />
    </div>
  )
}
