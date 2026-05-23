export type AgentName = 'coordinator' | 'planner' | 'notewriter' | 'advisor' | 'idle'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  agents?: AgentName[]
}

export interface AgentStatus {
  name: AgentName
  status: 'idle' | 'running' | 'done' | 'error'
  startTime?: Date
}

export interface CalendarEvent {
  id: string
  title: string
  start: { dateTime: string }
  end: { dateTime: string }
  description?: string
  course?: string
}

export interface Task {
  id: string
  title: string
  description?: string
  due_date?: string
  priority: 'low' | 'medium' | 'high'
  course?: string
  completed: boolean
}

export interface StudentProfile {
  personal_info?: {
    name?: string
    major?: string
    academic_year?: string
    gpa?: number
  }
  learning_preferences?: {
    learning_style?: string
    study_environment?: string
    peak_study_hours?: string
    study_hours_per_day?: number
    note_taking_method?: string
    break_interval_minutes?: number
  }
  academic_info?: {
    current_courses?: Array<{ name: string; credits: number; professor: string }>
    performance?: Record<string, string>
    strengths?: string[]
    challenges?: string[]
  }
  goals?: string[]
}

export interface SSEEvent {
  type: 'session' | 'agent_start' | 'coordinator_done' | 'chunk' | 'agent_end' | 'done' | 'error'
  data: Record<string, unknown>
}
