import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { videosApi, analysisApi } from '@/api/client'

export default function Dashboard() {
  const [videos, setVideos] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadVideos()
  }, [])

  const loadVideos = async () => {
    try {
      const data = await videosApi.list()
      setVideos(data.videos || [])
    } catch (error) {
      console.error('Failed to load videos:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="text-center py-8">Loading...</div>
  }

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-3xl font-bold">Dashboard</h2>
        <Link
          to="/upload"
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
        >
          Upload Video
        </Link>
      </div>

      <div className="grid gap-4">
        {videos.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            No videos uploaded yet. <Link to="/upload" className="text-primary">Upload one now</Link>
          </div>
        ) : (
          videos.map((video) => (
            <Link
              key={video.video_id}
              to={`/video/${video.video_id}`}
              className="block p-4 border rounded-lg hover:bg-accent transition-colors"
            >
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="font-semibold">{video.filename}</h3>
                  <p className="text-sm text-muted-foreground">
                    {video.has_analysis ? 'Analyzed' : 'Pending analysis'}
                  </p>
                </div>
                <div className="text-sm">
                  {(video.file_size / 1024 / 1024).toFixed(2)} MB
                </div>
              </div>
            </Link>
          ))
        )}
      </div>
    </div>
  )
}

