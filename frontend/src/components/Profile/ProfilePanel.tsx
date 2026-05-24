import { useEffect, useState } from 'react'
import { Save, User } from 'lucide-react'
import { getProfile, updateProfile } from '../../api/client'
import type { StudentProfile } from '../../types'

export default function ProfilePanel() {
  const [profile, setProfile] = useState<StudentProfile>({})
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    getProfile().then((res) => setProfile(res.data)).catch(() => {})
  }, [])

  const set = (path: string[], value: string | number) => {
    setProfile((prev) => {
      const next = structuredClone(prev) as Record<string, Record<string, unknown>>
      let cur: Record<string, unknown> = next
      for (let i = 0; i < path.length - 1; i++) {
        if (!cur[path[i]]) cur[path[i]] = {}
        cur = cur[path[i]] as Record<string, unknown>
      }
      cur[path[path.length - 1]] = value
      return next as StudentProfile
    })
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      await updateProfile(profile)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } finally {
      setSaving(false)
    }
  }

  const pi = profile.personal_info ?? {}
  const lp = profile.learning_preferences ?? {}
  const ai = profile.academic_info ?? {}

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 mb-3">
        <User size={16} className="text-gray-500" />
        <h3 className="font-semibold text-gray-700 text-sm">Student Profile</h3>
      </div>

      <div className="flex-1 overflow-y-auto space-y-4">
        {/* Personal Info */}
        <section>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            Personal Info
          </p>
          <div className="space-y-2">
            <Field
              label="Name"
              value={pi.name ?? ''}
              onChange={(v) => set(['personal_info', 'name'], v)}
            />
            <Field
              label="Major"
              value={pi.major ?? ''}
              onChange={(v) => set(['personal_info', 'major'], v)}
            />
            <SelectField
              label="Year"
              value={pi.academic_year ?? ''}
              options={['Freshman', 'Sophomore', 'Junior', 'Senior', 'Graduate']}
              onChange={(v) => set(['personal_info', 'academic_year'], v)}
            />
          </div>
        </section>

        {/* Learning Preferences */}
        <section>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            Learning Preferences
          </p>
          <div className="space-y-2">
            <SelectField
              label="Learning Style"
              value={lp.learning_style ?? ''}
              options={['Visual', 'Auditory', 'Reading/Writing', 'Kinesthetic']}
              onChange={(v) => set(['learning_preferences', 'learning_style'], v)}
            />
            <Field
              label="Peak Study Hours"
              value={lp.peak_study_hours ?? ''}
              placeholder="e.g. 9 AM - 12 PM"
              onChange={(v) => set(['learning_preferences', 'peak_study_hours'], v)}
            />
            <Field
              label="Study Hours/Day"
              value={String(lp.study_hours_per_day ?? '')}
              placeholder="e.g. 4"
              onChange={(v) => set(['learning_preferences', 'study_hours_per_day'], Number(v))}
            />
          </div>
        </section>

        {/* Goals */}
        <section>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            Goals
          </p>
          <div className="space-y-1">
            {(profile.goals ?? []).map((goal, i) => (
              <div key={i} className="flex items-center gap-1">
                <span className="text-indigo-500 text-xs">→</span>
                <span className="text-xs text-gray-700">{goal}</span>
              </div>
            ))}
            {(!profile.goals || profile.goals.length === 0) && (
              <p className="text-xs text-gray-400">No goals set</p>
            )}
          </div>
        </section>

        {/* Courses */}
        <section>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            Current Courses
          </p>
          <div className="space-y-1">
            {(ai.current_courses ?? []).map((course, i) => (
              <div
                key={i}
                className="flex items-center justify-between bg-gray-50 rounded-lg px-2 py-1 border border-gray-100"
              >
                <span className="text-xs font-medium text-gray-700 truncate">{course.name}</span>
                <span className="text-xs text-gray-400 ml-2 flex-shrink-0">{course.credits}cr</span>
              </div>
            ))}
            {(!ai.current_courses || ai.current_courses.length === 0) && (
              <p className="text-xs text-gray-400">No courses listed</p>
            )}
          </div>
        </section>
      </div>

      <button
        onClick={handleSave}
        disabled={saving}
        className="mt-3 flex items-center justify-center gap-2 w-full py-2 bg-indigo-600 text-white rounded-xl text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 transition-colors"
      >
        <Save size={14} />
        {saved ? '✓ Saved!' : saving ? 'Saving…' : 'Save Profile'}
      </button>
    </div>
  )
}

function Field({
  label,
  value,
  placeholder,
  onChange,
}: {
  label: string
  value: string
  placeholder?: string
  onChange: (v: string) => void
}) {
  return (
    <div>
      <label className="text-xs text-gray-500 block mb-0.5">{label}</label>
      <input
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className="w-full text-xs border border-gray-200 rounded-lg px-2 py-1.5 outline-none focus:border-indigo-400 bg-white"
      />
    </div>
  )
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: string
  options: string[]
  onChange: (v: string) => void
}) {
  return (
    <div>
      <label className="text-xs text-gray-500 block mb-0.5">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full text-xs border border-gray-200 rounded-lg px-2 py-1.5 outline-none focus:border-indigo-400 bg-white"
      >
        <option value="">Select…</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </div>
  )
}
