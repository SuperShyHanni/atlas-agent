import { useEffect, useState } from 'react'
import Sidebar from './components/Sidebar/Sidebar'
import Chat from './components/Chat/Chat'
import AgentStatusPanel from './components/AgentStatus/AgentStatus'
import type { AgentName, AgentStatus } from './types'
import { createSession } from './api/client'

const INITIAL_AGENT_STATUSES: AgentStatus[] = [
  { name: 'coordinator', status: 'idle' },
  { name: 'planner', status: 'idle' },
  { name: 'notewriter', status: 'idle' },
  { name: 'advisor', status: 'idle' },
]

export default function App() {
  const [sessionId, setSessionId] = useState<string>('')
  const [agentStatuses, setAgentStatuses] = useState<AgentStatus[]>(INITIAL_AGENT_STATUSES)
  const [requiredAgents, setRequiredAgents] = useState<AgentName[]>([])
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    const stored = localStorage.getItem('atlas_session_id')
    if (stored) {
      setSessionId(stored)
    } else {
      createSession().then((res) => {
        const id = res.data.session_id
        setSessionId(id)
        localStorage.setItem('atlas_session_id', id)
      })
    }
  }, [])

  if (!sessionId) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-50">
        <div className="flex items-center gap-3 text-gray-500">
          <div className="w-5 h-5 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          Initializing ATLAS…
        </div>
      </div>
    )
  }

  return (
    <div className="h-screen flex bg-gray-50 overflow-hidden">
      <Sidebar sessionId={sessionId} />

      <div className="flex-1 flex flex-col min-w-0">
        <Chat
          sessionId={sessionId}
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
