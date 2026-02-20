import { useState, useEffect } from 'react'
import Card from '../components/ui/Card'

export default function Docs() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('http://localhost:5000/api/docs')
      .then(res => {
        if (!res.ok) throw new Error('Failed to load Documentation data')
        return res.json()
      })
      .then(data => {
        setData(data)
        setLoading(false)
      })
      .catch(err => {
        console.error("Docs load error:", err)
        setError(err.message)
        setLoading(false)
      })
  }, [])

  if (loading) {
    return (
      <div className="w-full min-h-screen bg-white dark:bg-neutral-950 p-6 flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-neutral-200 dark:border-neutral-700 border-t-neutral-900 dark:border-t-neutral-100 rounded-full animate-spin" />
          <p className="text-neutral-600 dark:text-neutral-400">Loading documentation...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="w-full min-h-screen bg-white dark:bg-neutral-950 p-6 flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="text-5xl">⚠️</div>
          <h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">Unable to load documentation</h2>
          <p className="text-neutral-600 dark:text-neutral-400 max-w-md mx-auto">{error}</p>
          <p className="text-sm text-neutral-500">Make sure the backend server (app.py) is running.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full min-h-screen bg-white dark:bg-neutral-950 p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100 mb-2">
            {data?.title || 'Documentation'}
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400">
            Technical documentation and system explanation
          </p>
        </div>

        <div className="space-y-6">
          {data?.sections?.map((section) => (
            <Card key={section.id}>
              <div className="p-1">
                <h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100 mb-3">
                  {section.title}
                </h2>
                <div className="text-neutral-600 dark:text-neutral-300 leading-relaxed whitespace-pre-line">
                  {section.content}
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}