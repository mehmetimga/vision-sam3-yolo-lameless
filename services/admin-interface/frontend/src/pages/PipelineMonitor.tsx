/**
 * Pipeline Monitor Page
 * Real-time status and control of all ML pipeline services
 */
import { useState, useEffect, useCallback } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { usePipelineWebSocket } from '../hooks/useWebSocket'
import {
  Activity,
  CheckCircle,
  XCircle,
  AlertTriangle,
  HelpCircle,
  Play,
  RefreshCw,
  Loader2,
  ChevronDown,
  ChevronUp,
  Terminal,
  Clock
} from 'lucide-react'

interface PipelineStatus {
  service_name: string
  description: string
  status: 'healthy' | 'degraded' | 'down' | 'unknown'
  last_heartbeat: string | null
  active_jobs: number
  success_count: number
  error_count: number
  success_rate: number
  last_error: string | null
}

interface LogEntry {
  timestamp: string
  level: string
  service: string
  message: string
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function PipelineMonitor() {
  const { getAccessToken, hasRole } = useAuth()
  const [pipelines, setPipelines] = useState<PipelineStatus[]>([])
  const [selectedPipeline, setSelectedPipeline] = useState<string | null>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isLoadingLogs, setIsLoadingLogs] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [triggerVideoId, setTriggerVideoId] = useState('')
  const [triggerDialogOpen, setTriggerDialogOpen] = useState(false)
  const [triggerPipeline, setTriggerPipeline] = useState<string | null>(null)
  const [isTriggering, setIsTriggering] = useState(false)

  // WebSocket for real-time updates
  const { lastMessage, isConnected } = usePipelineWebSocket({
    onMessage: (message) => {
      if (message.type === 'pipeline_status') {
        setPipelines(prev => prev.map(p =>
          p.service_name === message.service
            ? { ...p, status: message.status as PipelineStatus['status'] }
            : p
        ))
      }
    }
  })

  const fetchPipelines = useCallback(async () => {
    try {
      const token = getAccessToken()
      const response = await fetch(`${API_URL}/api/pipeline/status`, {
        headers: { Authorization: `Bearer ${token}` }
      })

      if (!response.ok) throw new Error('Failed to fetch pipeline status')

      const data = await response.json()
      setPipelines(data)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load pipelines')
    } finally {
      setIsLoading(false)
    }
  }, [getAccessToken])

  const fetchLogs = useCallback(async (serviceName: string) => {
    setIsLoadingLogs(true)
    try {
      const token = getAccessToken()
      const response = await fetch(`${API_URL}/api/pipeline/${serviceName}/logs?limit=50`, {
        headers: { Authorization: `Bearer ${token}` }
      })

      if (!response.ok) throw new Error('Failed to fetch logs')

      const data = await response.json()
      setLogs(data)
    } catch (err) {
      console.error('Failed to fetch logs:', err)
      setLogs([])
    } finally {
      setIsLoadingLogs(false)
    }
  }, [getAccessToken])

  useEffect(() => {
    fetchPipelines()
    const interval = setInterval(fetchPipelines, 30000) // Refresh every 30s
    return () => clearInterval(interval)
  }, [fetchPipelines])

  useEffect(() => {
    if (selectedPipeline) {
      fetchLogs(selectedPipeline)
    }
  }, [selectedPipeline, fetchLogs])

  const handleTrigger = async () => {
    if (!triggerPipeline || !triggerVideoId) return

    setIsTriggering(true)
    try {
      const token = getAccessToken()
      const response = await fetch(
        `${API_URL}/api/pipeline/${triggerPipeline}/trigger/${triggerVideoId}`,
        {
          method: 'POST',
          headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      )

      if (!response.ok) throw new Error('Failed to trigger pipeline')

      setTriggerDialogOpen(false)
      setTriggerVideoId('')
      setTriggerPipeline(null)
      fetchPipelines()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to trigger pipeline')
    } finally {
      setIsTriggering(false)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'degraded':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />
      case 'down':
        return <XCircle className="h-5 w-5 text-red-500" />
      default:
        return <HelpCircle className="h-5 w-5 text-gray-400" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-50 border-green-200'
      case 'degraded':
        return 'bg-yellow-50 border-yellow-200'
      case 'down':
        return 'bg-red-50 border-red-200'
      default:
        return 'bg-gray-50 border-gray-200'
    }
  }

  const formatTime = (timestamp: string | null) => {
    if (!timestamp) return 'Never'
    const date = new Date(timestamp)
    return date.toLocaleTimeString()
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Pipeline Monitor</h1>
          <p className="text-sm text-gray-500">
            Real-time status of all ML pipeline services
          </p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
            <span className="text-gray-500">
              {isConnected ? 'Live updates' : 'Disconnected'}
            </span>
          </div>
          <button
            onClick={() => fetchPipelines()}
            className="flex items-center gap-2 px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm"
          >
            <RefreshCw className="h-4 w-4" />
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
          {error}
        </div>
      )}

      {/* Pipeline Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        {pipelines.map((pipeline) => (
          <div
            key={pipeline.service_name}
            className={`p-4 rounded-lg border-2 transition-all cursor-pointer hover:shadow-md ${getStatusColor(pipeline.status)} ${
              selectedPipeline === pipeline.service_name ? 'ring-2 ring-blue-500' : ''
            }`}
            onClick={() => setSelectedPipeline(
              selectedPipeline === pipeline.service_name ? null : pipeline.service_name
            )}
          >
            <div className="flex items-start justify-between mb-2">
              {getStatusIcon(pipeline.status)}
              {hasRole(['admin', 'researcher']) && (
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    setTriggerPipeline(pipeline.service_name)
                    setTriggerDialogOpen(true)
                  }}
                  className="p-1 hover:bg-white/50 rounded"
                  title="Trigger pipeline"
                >
                  <Play className="h-4 w-4 text-gray-600" />
                </button>
              )}
            </div>

            <h3 className="font-semibold text-gray-900 text-sm truncate">
              {pipeline.service_name}
            </h3>
            <p className="text-xs text-gray-500 mt-1 line-clamp-2">
              {pipeline.description}
            </p>

            <div className="mt-3 pt-3 border-t border-gray-200/50 space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">Active Jobs</span>
                <span className="font-medium">{pipeline.active_jobs}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">Success Rate</span>
                <span className="font-medium">
                  {(pipeline.success_rate * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">Last Heartbeat</span>
                <span className="font-medium">{formatTime(pipeline.last_heartbeat)}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Logs Panel */}
      {selectedPipeline && (
        <div className="bg-white rounded-lg border shadow-sm">
          <div className="flex items-center justify-between p-4 border-b">
            <div className="flex items-center gap-2">
              <Terminal className="h-5 w-5 text-gray-500" />
              <h3 className="font-semibold">{selectedPipeline} Logs</h3>
            </div>
            <button
              onClick={() => setSelectedPipeline(null)}
              className="text-gray-400 hover:text-gray-600"
            >
              <ChevronUp className="h-5 w-5" />
            </button>
          </div>

          <div className="p-4 bg-gray-900 rounded-b-lg max-h-80 overflow-auto font-mono text-sm">
            {isLoadingLogs ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-gray-400" />
              </div>
            ) : logs.length > 0 ? (
              logs.map((log, index) => (
                <div key={index} className="py-1">
                  <span className="text-gray-500">
                    {new Date(log.timestamp).toLocaleTimeString()}
                  </span>
                  <span className={`ml-2 ${
                    log.level === 'ERROR' ? 'text-red-400' :
                    log.level === 'WARNING' ? 'text-yellow-400' :
                    'text-green-400'
                  }`}>
                    [{log.level}]
                  </span>
                  <span className="ml-2 text-gray-300">{log.message}</span>
                </div>
              ))
            ) : (
              <p className="text-gray-500 text-center py-8">No logs available</p>
            )}
          </div>
        </div>
      )}

      {/* Trigger Dialog */}
      {triggerDialogOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">
              Trigger Pipeline: {triggerPipeline}
            </h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Video ID
                </label>
                <input
                  type="text"
                  value={triggerVideoId}
                  onChange={(e) => setTriggerVideoId(e.target.value)}
                  placeholder="Enter video ID"
                  className="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div className="flex justify-end gap-3">
                <button
                  onClick={() => {
                    setTriggerDialogOpen(false)
                    setTriggerVideoId('')
                    setTriggerPipeline(null)
                  }}
                  className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-md"
                >
                  Cancel
                </button>
                <button
                  onClick={handleTrigger}
                  disabled={!triggerVideoId || isTriggering}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {isTriggering ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4" />
                  )}
                  Trigger
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Stats Summary */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-lg border">
          <div className="flex items-center gap-2 text-green-600 mb-1">
            <CheckCircle className="h-4 w-4" />
            <span className="text-sm font-medium">Healthy</span>
          </div>
          <p className="text-2xl font-bold">
            {pipelines.filter(p => p.status === 'healthy').length}
          </p>
        </div>
        <div className="bg-white p-4 rounded-lg border">
          <div className="flex items-center gap-2 text-yellow-600 mb-1">
            <AlertTriangle className="h-4 w-4" />
            <span className="text-sm font-medium">Degraded</span>
          </div>
          <p className="text-2xl font-bold">
            {pipelines.filter(p => p.status === 'degraded').length}
          </p>
        </div>
        <div className="bg-white p-4 rounded-lg border">
          <div className="flex items-center gap-2 text-red-600 mb-1">
            <XCircle className="h-4 w-4" />
            <span className="text-sm font-medium">Down</span>
          </div>
          <p className="text-2xl font-bold">
            {pipelines.filter(p => p.status === 'down').length}
          </p>
        </div>
        <div className="bg-white p-4 rounded-lg border">
          <div className="flex items-center gap-2 text-gray-600 mb-1">
            <Activity className="h-4 w-4" />
            <span className="text-sm font-medium">Active Jobs</span>
          </div>
          <p className="text-2xl font-bold">
            {pipelines.reduce((sum, p) => sum + p.active_jobs, 0)}
          </p>
        </div>
      </div>
    </div>
  )
}
