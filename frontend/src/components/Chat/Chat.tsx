import { useEffect, useRef, useState } from 'react'
import { Send, Trash2 } from 'lucide-react'
import { streamChatFetch, clearHistory, getHistory } from '../../api/client'
import type { AgentName, AgentStatus, Message } from '../../types'
import MessageBubble from './MessageBubble'

const SUGGESTED_PROMPTS = [
  'Help me plan my study schedule for this week',
  'Create study notes for Data Structures',
  'I have a database exam in 2 days, what should I focus on?',
  'How can I improve my time management?',
]

interface Props {
  onAgentStatusChange: (agents: AgentStatus[]) => void
  onRequiredAgentsChange: (agents: AgentName[]) => void
  onLoadingChange: (loading: boolean) => void
}

export default function Chat({
  onAgentStatusChange,
  onRequiredAgentsChange,
  onLoadingChange,
}: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [streamingId, setStreamingId] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    getHistory()
      .then(({ data }) => {
        const loaded: Message[] = data.messages.map((m, i) => ({
          id: `history-${i}`,
          role: m.role as Message['role'],
          content: m.content,
          timestamp: new Date(),
        }))
        setMessages(loaded)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const resetAgentStatuses = () => {
    onAgentStatusChange([
      { name: 'coordinator', status: 'idle' },
      { name: 'planner', status: 'idle' },
      { name: 'notewriter', status: 'idle' },
      { name: 'advisor', status: 'idle' },
    ])
    onRequiredAgentsChange([])
  }

  const sendMessage = async (text: string) => {
    if (!text.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: text.trim(),
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    onLoadingChange(true)

    const assistantId = (Date.now() + 1).toString()
    const assistantMessage: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      agents: [],
    }

    setMessages((prev) => [...prev, assistantMessage])
    setStreamingId(assistantId)

    const agentStatuses: AgentStatus[] = [
      { name: 'coordinator', status: 'idle' },
      { name: 'planner', status: 'idle' },
      { name: 'notewriter', status: 'idle' },
      { name: 'advisor', status: 'idle' },
    ]

    const updateAgent = (name: AgentName, status: AgentStatus['status']) => {
      const idx = agentStatuses.findIndex((a) => a.name === name)
      if (idx >= 0) agentStatuses[idx] = { name, status }
      onAgentStatusChange([...agentStatuses])
    }

    try {
      for await (const { event, data } of streamChatFetch(text.trim())) {
        console.log('[SSE]', event, data)
        if (event === 'agent_start') {
          const agent = data.agent as AgentName
          updateAgent(agent, 'running')
        } else if (event === 'coordinator_done') {
          updateAgent('coordinator', 'done')
          const required = (data.required_agents as string[] ?? []).map((a) =>
            a.toLowerCase() as AgentName,
          )
          onRequiredAgentsChange(required)
        } else if (event === 'chunk') {
          const content = data.content as string
          const agent = data.agent as AgentName
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    content: m.content + content,
                    agents: m.agents?.includes(agent) ? m.agents : [...(m.agents ?? []), agent],
                  }
                : m,
            ),
          )
        } else if (event === 'agent_end') {
          const agent = data.agent as AgentName
          updateAgent(agent, 'done')
        } else if (event === 'error') {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: `⚠️ Error: ${data.message as string}` }
                : m,
            ),
          )
        }
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: '⚠️ Connection error. Please check the backend server.' }
            : m,
        ),
      )
    } finally {
      setStreamingId(null)
      setIsLoading(false)
      onLoadingChange(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  const handleClear = async () => {
    await clearHistory()
    setMessages([])
    resetAgentStatuses()
  }

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200 bg-white flex-shrink-0">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-500" />
          <span className="text-sm text-gray-600">ATLAS Academic Agent</span>
        </div>
        <button
          onClick={handleClear}
          className="flex items-center gap-1 text-xs text-gray-500 hover:text-red-500 transition-colors"
        >
          <Trash2 size={14} />
          Clear chat
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 bg-gray-50">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center">
            <div className="text-5xl mb-4">🎓</div>
            <h2 className="text-xl font-semibold text-gray-700 mb-2">
              Welcome to ATLAS
            </h2>
            <p className="text-gray-500 text-sm mb-8 max-w-md">
              Your Academic Task and Learning Agent System. I coordinate specialized
              AI agents to help you study smarter.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
              {SUGGESTED_PROMPTS.map((prompt) => (
                <button
                  key={prompt}
                  onClick={() => sendMessage(prompt)}
                  className="text-left text-sm bg-white border border-gray-200 rounded-xl p-3 hover:border-indigo-300 hover:bg-indigo-50 transition-colors text-gray-700 shadow-sm"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                isStreaming={streamingId === msg.id}
              />
            ))}
          </>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div className="flex-shrink-0 border-t border-gray-200 bg-white px-4 py-3">
        <div className="flex items-end gap-2 bg-gray-100 rounded-2xl px-4 py-2">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about schedules, study notes, academic advice…"
            rows={1}
            disabled={isLoading}
            className="flex-1 bg-transparent resize-none outline-none text-sm text-gray-800 placeholder-gray-400 max-h-32 py-1 disabled:opacity-50"
            style={{ minHeight: '24px' }}
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={!input.trim() || isLoading}
            className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white disabled:opacity-40 hover:bg-indigo-700 transition-colors flex-shrink-0 mb-0.5"
          >
            {isLoading ? (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <Send size={14} />
            )}
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-1 text-center">
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
