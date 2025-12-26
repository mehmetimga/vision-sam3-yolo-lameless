import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { videosApi, analysisApi, trainingApi, shapApi } from '@/api/client'

export default function VideoAnalysis() {
  const { videoId } = useParams()
  const [video, setVideo] = useState<any>(null)
  const [analysis, setAnalysis] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [labeling, setLabeling] = useState(false)

  useEffect(() => {
    if (videoId) {
      loadData()
    }
  }, [videoId])

  const loadData = async () => {
    try {
      const [videoData, analysisData] = await Promise.all([
        videosApi.get(videoId!),
        analysisApi.getSummary(videoId!).catch(() => null),
      ])
      setVideo(videoData)
      setAnalysis(analysisData)
    } catch (error) {
      console.error('Failed to load data:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleLabel = async (label: number) => {
    if (!videoId) return

    setLabeling(true)
    try {
      await trainingApi.label(videoId, label)
      alert('Label saved successfully!')
    } catch (error) {
      alert('Failed to save label')
    } finally {
      setLabeling(false)
    }
  }

  if (loading) {
    return <div className="text-center py-8">Loading...</div>
  }

  if (!video) {
    return <div className="text-center py-8">Video not found</div>
  }

  const probability = analysis?.final_probability || 0.5
  const prediction = analysis?.final_prediction || 0

  return (
    <div>
      <h2 className="text-3xl font-bold mb-6">Video Analysis</h2>

      <div className="grid gap-6 md:grid-cols-2">
        <div className="border rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4">Video Information</h3>
          <p className="text-sm text-muted-foreground mb-2">Filename: {video.filename}</p>
          <p className="text-sm text-muted-foreground mb-2">
            Size: {(video.file_size / 1024 / 1024).toFixed(2)} MB
          </p>
          <p className="text-sm text-muted-foreground">Status: {video.status}</p>
        </div>

        <div className="border rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4">Prediction</h3>
          <div className="mb-4">
            <div className="flex justify-between mb-2">
              <span>Lameness Probability:</span>
              <span className="font-bold">{(probability * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-secondary rounded-full h-2">
              <div
                className="bg-primary h-2 rounded-full"
                style={{ width: `${probability * 100}%` }}
              />
            </div>
          </div>
          <p className="text-lg font-semibold">
            Prediction: {prediction === 1 ? 'Lame' : 'Sound'}
          </p>
        </div>
      </div>

      {analysis && (
        <div className="mt-6 border rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4">Pipeline Contributions</h3>
          <div className="grid gap-2">
            {Object.entries(analysis.pipeline_contributions || {}).map(([pipeline, value]) => (
              <div key={pipeline} className="flex justify-between">
                <span className="capitalize">{pipeline}:</span>
                <span>{value !== null ? (Number(value) * 100).toFixed(1) + '%' : 'N/A'}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="mt-6 border rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4">Label Video</h3>
        <div className="flex gap-4">
          <button
            onClick={() => handleLabel(0)}
            disabled={labeling}
            className="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
          >
            {labeling ? 'Saving...' : 'Label as Sound'}
          </button>
          <button
            onClick={() => handleLabel(1)}
            disabled={labeling}
            className="px-6 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50"
          >
            {labeling ? 'Saving...' : 'Label as Lame'}
          </button>
        </div>
      </div>
    </div>
  )
}

