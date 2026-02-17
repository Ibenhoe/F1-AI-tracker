import Card from '../components/ui/Card'

export default function Wiki() {
  return (
    <div className="w-full min-h-screen bg-white dark:bg-neutral-950 p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100 mb-2">
            Wiki
          </h1>
          <p className="text-neutral-600 dark:text-neutral-400">
            Community knowledge base
          </p>
        </div>
        
        <Card>
          <div className="p-8 text-center text-neutral-500">
            <p>This page is currently empty.</p>
          </div>
        </Card>
      </div>
    </div>
  )
}
