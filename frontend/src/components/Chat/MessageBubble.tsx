import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from '../../types'

const AGENT_LABELS: Record<string, string> = {
  planner: '📅 Planner',
  notewriter: '✍️ NoteWriter',
  advisor: '💡 Advisor',
}

interface Props {
  message: Message
  isStreaming?: boolean
}

export default function MessageBubble({ message, isStreaming }: Props) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white text-sm font-bold mr-3 flex-shrink-0 mt-1">
          A
        </div>
      )}

      <div className={`max-w-[75%] ${isUser ? 'order-1' : ''}`}>
        {!isUser && message.agents && message.agents.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-1">
            {message.agents.map((a) => (
              <span
                key={a}
                className="text-xs bg-indigo-100 text-indigo-700 rounded-full px-2 py-0.5 font-medium"
              >
                {AGENT_LABELS[a] ?? a}
              </span>
            ))}
          </div>
        )}

        <div
          className={`rounded-2xl px-4 py-3 ${
            isUser
              ? 'bg-indigo-600 text-white rounded-tr-sm'
              : 'bg-white border border-gray-200 text-gray-800 rounded-tl-sm shadow-sm'
          }`}
        >
          {isUser ? (
            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose prose-sm max-w-none prose-headings:text-gray-900 prose-p:text-gray-700 prose-code:bg-gray-100 prose-code:rounded prose-code:px-1 prose-pre:bg-gray-900 prose-pre:text-gray-100">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
              {isStreaming && (
                <span className="inline-block w-2 h-4 bg-indigo-600 animate-pulse ml-0.5 align-middle" />
              )}
            </div>
          )}
        </div>

        <p className="text-xs text-gray-400 mt-1 px-1">
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </p>
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center text-white text-sm font-bold ml-3 flex-shrink-0 mt-1">
          U
        </div>
      )}
    </div>
  )
}
