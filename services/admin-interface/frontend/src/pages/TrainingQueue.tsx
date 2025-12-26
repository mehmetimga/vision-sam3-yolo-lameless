import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { trainingApi } from '@/api/client'

export default function TrainingQueue() {
  const [queue, setQueue] = useState<any[]>([])
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      const [queueData, statsData] = await Promise.all([
        trainingApi.getQueue(),
        trainingApi.getStats(),
      ])
      setQueue(queueData.videos || [])
      setStats(statsData)
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="text-center py-8">Loading...</div>
  }

  return (
    <div>
      <h2 className="text-3xl font-bold mb-6">Training Queue</h2>

      {stats && (
        <div className="grid gap-4 md:grid-cols-4 mb-6">
          <div className="border rounded-lg p-4">
            <p className="text-sm text-muted-foreground">Total Labels</p>
            <p className="text-2xl font-bold">{stats.total_labels}</p>
          </div>
          <div className="border rounded-lg p-4">
            <p className="text-sm text-muted-foreground">Sound</p>
            <p className="text-2xl font-bold text-green-600">{stats.sound_count}</p>
          </div>
          <div className="border rounded-lg p-4">
            <p className="text-sm text-muted-foreground">Lame</p>
            <p className="text-2xl font-bold text-red-600">{stats.lame_count}</p>
          </div>
          <div className="border rounded-lg p-4">
            <p className="text-sm text-muted-foreground">Balance Ratio</p>
            <p className="text-2xl font-bold">{stats.balance_ratio.toFixed(2)}</p>
          </div>
        </div>
      )}

      <div className="border rounded-lg">
        <div className="p-4 border-b">
          <h3 className="font-semibold">Videos Needing Labels ({queue.length})</h3>
        </div>
        <div className="divide-y">
          {queue.length === 0 ? (
            <div className="p-8 text-center text-muted-foreground">
              No videos in queue
            </div>
          ) : (
            queue.map((video) => (
              <Link
                key={video.video_id}
                to={`/video/${video.video_id}`}
                className="block p-4 hover:bg-accent transition-colors"
              >
                <div className="flex justify-between items-center">
                  <div>
                    <p className="font-medium">Video {video.video_id.slice(0, 8)}...</p>
                    <p className="text-sm text-muted-foreground">
                      Predicted: {(video.predicted_probability * 100).toFixed(1)}% | 
                      Uncertainty: {(video.uncertainty * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="text-sm text-muted-foreground">→</div>
                </div>
              </Link>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

