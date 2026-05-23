import type { AgentName, AgentStatus as AgentStatusType } from '../../types'

const AGENT_LABELS: Record<AgentName, string> = {
  coordinator: 'Coordinator',
  planner: 'Planner',
  notewriter: 'NoteWriter',
  advisor: 'Advisor',
  idle: 'Idle',
}

const AGENT_COLORS: Record<AgentName, string> = {
  coordinator: 'bg-purple-500',
  planner: 'bg-blue-500',
  notewriter: 'bg-green-500',
  advisor: 'bg-orange-500',
  idle: 'bg-gray-400',
}

const AGENT_ICONS: Record<AgentName, string> = {
  coordinator: '🎯',
  planner: '📅',
  notewriter: '✍️',
  advisor: '💡',
  idle: '⭕',
}

interface Props {
  agents: AgentStatusType[]
  requiredAgents: AgentName[]
  isLoading: boolean
}

export default function AgentStatus({ agents, requiredAgents, isLoading }: Props) {
  return (
    <div className="bg-white border-l border-gray-200 w-64 flex-shrink-0 flex flex-col">
      <div className="p-4 border-b border-gray-200">
        <h2 className="font-semibold text-gray-700 text-sm uppercase tracking-wide">
          Agent Activity
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {(['coordinator', 'planner', 'notewriter', 'advisor'] as AgentName[]).map((agentName) => {
          const agentStatus = agents.find((a) => a.name === agentName)
          const status = agentStatus?.status ?? 'idle'
          const isRequired = requiredAgents.includes(agentName)
          const isActive = status === 'running'
          const isDone = status === 'done'

          return (
            <div
              key={agentName}
              className={`rounded-lg p-3 border transition-all duration-300 ${
                isActive
                  ? 'border-indigo-300 bg-indigo-50 shadow-sm'
                  : isDone
                  ? 'border-green-200 bg-green-50'
                  : isRequired
                  ? 'border-gray-200 bg-gray-50'
                  : 'border-gray-100 bg-white opacity-60'
              }`}
            >
              <div className="flex items-center gap-2">
                <div
                  className={`w-2 h-2 rounded-full flex-shrink-0 ${
                    isActive
                      ? `${AGENT_COLORS[agentName]} animate-pulse`
                      : isDone
                      ? 'bg-green-500'
                      : 'bg-gray-300'
                  }`}
                />
                <span className="text-base">{AGENT_ICONS[agentName]}</span>
                <span className="font-medium text-sm text-gray-800">{AGENT_LABELS[agentName]}</span>
                {isActive && (
                  <span className="ml-auto text-xs text-indigo-600 font-medium animate-pulse">
                    Running…
                  </span>
                )}
                {isDone && (
                  <span className="ml-auto text-xs text-green-600 font-medium">✓ Done</span>
                )}
              </div>

              {isActive && (
                <div className="mt-2 flex gap-1">
                  {[0, 1, 2].map((i) => (
                    <div
                      key={i}
                      className={`h-1 flex-1 rounded-full ${AGENT_COLORS[agentName]} opacity-60 animate-pulse`}
                      style={{ animationDelay: `${i * 0.15}s` }}
                    />
                  ))}
                </div>
              )}
            </div>
          )
        })}
      </div>

      {!isLoading && agents.every((a) => a.status === 'idle') && (
        <div className="p-4 text-xs text-gray-400 text-center">
          Send a message to activate agents
        </div>
      )}
    </div>
  )
}
