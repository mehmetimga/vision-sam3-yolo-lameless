/**
 * System Health Page
 * Infrastructure monitoring dashboard for Docker, NATS, databases, and disk usage
 */
import { useState, useEffect, useCallback } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { useHealthWebSocket } from '../hooks/useWebSocket'
import {
  Server,
  Database,
  HardDrive,
  Activity,
  CheckCircle,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Loader2,
  Cpu,
  MemoryStick,
  Clock,
  Zap,
  TrendingUp
} from 'lucide-react'

interface HealthOverview {
  status: 'healthy' | 'degraded' | 'critical'
  timestamp: string
  components: Record<string, string>
  issues: string[]
}

interface ContainerHealth {
  name: string
  status: string
  cpu_percent: number | null
  memory_mb: number | null
  memory_percent: number | null
  uptime: string | null
}

interface NATSHealth {
  status: string
  connections: number
  subscriptions: number
  messages_in: number
  messages_out: number
  bytes_in: number
  bytes_out: number
}

interface DatabaseHealth {
  status: string
  connection_count: number
  database_size_mb: number
  response_time_ms: number
}

interface DiskUsage {
  path: string
  total_gb: number
  used_gb: number
  free_gb: number
  percent_used: number
  status: string
}

interface ThroughputMetrics {
  videos_processed_24h: number
  videos_processed_7d: number
  avg_processing_time_s: number
  success_rate: number
  queue_depth: number
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function SystemHealth() {
  const { getAccessToken } = useAuth()
  const [overview, setOverview] = useState<HealthOverview | null>(null)
  const [containers, setContainers] = useState<ContainerHealth[]>([])
  const [nats, setNats] = useState<NATSHealth | null>(null)
  const [postgres, setPostgres] = useState<DatabaseHealth | null>(null)
  const [qdrant, setQdrant] = useState<DatabaseHealth | null>(null)
  const [disk, setDisk] = useState<DiskUsage[]>([])
  const [throughput, setThroughput] = useState<ThroughputMetrics | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // WebSocket for real-time health updates
  const { isConnected } = useHealthWebSocket({
    onMessage: (message) => {
      if (message.type === 'health_update') {
        // Handle real-time health updates
        fetchAll()
      }
    }
  })

  const fetchAll = useCallback(async () => {
    const token = getAccessToken()
    const headers = { Authorization: `Bearer ${token}` }

    try {
      const [overviewRes, containersRes, natsRes, postgresRes, qdrantRes, diskRes, throughputRes] = await Promise.all([
        fetch(`${API_URL}/api/health/overview`, { headers }),
        fetch(`${API_URL}/api/health/docker`, { headers }),
        fetch(`${API_URL}/api/health/nats`, { headers }),
        fetch(`${API_URL}/api/health/postgres`, { headers }),
        fetch(`${API_URL}/api/health/qdrant`, { headers }),
        fetch(`${API_URL}/api/health/disk`, { headers }),
        fetch(`${API_URL}/api/health/throughput`, { headers })
      ])

      if (overviewRes.ok) setOverview(await overviewRes.json())
      if (containersRes.ok) setContainers(await containersRes.json())
      if (natsRes.ok) setNats(await natsRes.json())
      if (postgresRes.ok) setPostgres(await postgresRes.json())
      if (qdrantRes.ok) setQdrant(await qdrantRes.json())
      if (diskRes.ok) setDisk(await diskRes.json())
      if (throughputRes.ok) setThroughput(await throughputRes.json())

      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load health data')
    } finally {
      setIsLoading(false)
    }
  }, [getAccessToken])

  useEffect(() => {
    fetchAll()
    const interval = setInterval(fetchAll, 30000)
    return () => clearInterval(interval)
  }, [fetchAll])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'warning':
      case 'degraded':
        return <AlertTriangle className="h-5 w-5 text-yellow-500" />
      case 'critical':
      case 'down':
        return <XCircle className="h-5 w-5 text-red-500" />
      default:
        return <Activity className="h-5 w-5 text-gray-400" />
    }
  }

  const getStatusBg = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-500'
      case 'warning':
      case 'degraded':
        return 'bg-yellow-500'
      case 'critical':
      case 'down':
        return 'bg-red-500'
      default:
        return 'bg-gray-400'
    }
  }

  const formatBytes = (bytes: number) => {
    if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(2)} GB`
    if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(2)} MB`
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)} KB`
    return `${bytes} B`
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
          <h1 className="text-2xl font-bold text-gray-900">System Health</h1>
          <p className="text-sm text-gray-500">
            Infrastructure monitoring and system status
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
            onClick={() => fetchAll()}
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

      {/* Overview Card */}
      {overview && (
        <div className={`p-6 rounded-lg border-2 ${
          overview.status === 'healthy' ? 'bg-green-50 border-green-200' :
          overview.status === 'degraded' ? 'bg-yellow-50 border-yellow-200' :
          'bg-red-50 border-red-200'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {getStatusIcon(overview.status)}
              <div>
                <h2 className="text-lg font-semibold capitalize">{overview.status}</h2>
                <p className="text-sm text-gray-500">
                  Last updated: {new Date(overview.timestamp).toLocaleString()}
                </p>
              </div>
            </div>
            <div className="flex gap-4">
              {Object.entries(overview.components).map(([name, status]) => (
                <div key={name} className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${getStatusBg(status)}`} />
                  <span className="text-sm capitalize">{name}</span>
                </div>
              ))}
            </div>
          </div>
          {overview.issues.length > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Issues</h4>
              <ul className="space-y-1">
                {overview.issues.map((issue, i) => (
                  <li key={i} className="text-sm text-gray-600 flex items-center gap-2">
                    <AlertTriangle className="h-3 w-3 text-yellow-500" />
                    {issue}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Services Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* NATS Card */}
        <div className="bg-white p-6 rounded-lg border shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-purple-500" />
              <h3 className="font-semibold">NATS Messaging</h3>
            </div>
            {nats && getStatusIcon(nats.status)}
          </div>
          {nats && (
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Connections</span>
                <span className="font-medium">{nats.connections}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Subscriptions</span>
                <span className="font-medium">{nats.subscriptions}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Messages In</span>
                <span className="font-medium">{nats.messages_in.toLocaleString()}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Messages Out</span>
                <span className="font-medium">{nats.messages_out.toLocaleString()}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Bytes In</span>
                <span className="font-medium">{formatBytes(nats.bytes_in)}</span>
              </div>
            </div>
          )}
        </div>

        {/* PostgreSQL Card */}
        <div className="bg-white p-6 rounded-lg border shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Database className="h-5 w-5 text-blue-500" />
              <h3 className="font-semibold">PostgreSQL</h3>
            </div>
            {postgres && getStatusIcon(postgres.status)}
          </div>
          {postgres && (
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Connections</span>
                <span className="font-medium">{postgres.connection_count}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Database Size</span>
                <span className="font-medium">{postgres.database_size_mb.toFixed(2)} MB</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Response Time</span>
                <span className="font-medium">{postgres.response_time_ms.toFixed(2)} ms</span>
              </div>
            </div>
          )}
        </div>

        {/* Qdrant Card */}
        <div className="bg-white p-6 rounded-lg border shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Database className="h-5 w-5 text-orange-500" />
              <h3 className="font-semibold">Qdrant Vector DB</h3>
            </div>
            {qdrant && getStatusIcon(qdrant.status)}
          </div>
          {qdrant && (
            <div className="space-y-3">
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Status</span>
                <span className="font-medium capitalize">{qdrant.status}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Est. Size</span>
                <span className="font-medium">{qdrant.database_size_mb.toFixed(2)} MB</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Response Time</span>
                <span className="font-medium">{qdrant.response_time_ms.toFixed(2)} ms</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Disk Usage */}
      <div className="bg-white p-6 rounded-lg border shadow-sm">
        <div className="flex items-center gap-2 mb-4">
          <HardDrive className="h-5 w-5 text-gray-500" />
          <h3 className="font-semibold">Disk Usage</h3>
        </div>
        <div className="space-y-4">
          {disk.map((d) => (
            <div key={d.path}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium">{d.path}</span>
                <span className="text-sm text-gray-500">
                  {d.used_gb.toFixed(2)} / {d.total_gb.toFixed(2)} GB ({d.percent_used.toFixed(1)}%)
                </span>
              </div>
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${
                    d.status === 'critical' ? 'bg-red-500' :
                    d.status === 'warning' ? 'bg-yellow-500' :
                    'bg-green-500'
                  }`}
                  style={{ width: `${Math.min(d.percent_used, 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Throughput Metrics */}
      {throughput && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white p-4 rounded-lg border">
            <div className="flex items-center gap-2 text-blue-600 mb-1">
              <TrendingUp className="h-4 w-4" />
              <span className="text-sm font-medium">24h Processed</span>
            </div>
            <p className="text-2xl font-bold">{throughput.videos_processed_24h}</p>
          </div>
          <div className="bg-white p-4 rounded-lg border">
            <div className="flex items-center gap-2 text-purple-600 mb-1">
              <TrendingUp className="h-4 w-4" />
              <span className="text-sm font-medium">7d Processed</span>
            </div>
            <p className="text-2xl font-bold">{throughput.videos_processed_7d}</p>
          </div>
          <div className="bg-white p-4 rounded-lg border">
            <div className="flex items-center gap-2 text-green-600 mb-1">
              <CheckCircle className="h-4 w-4" />
              <span className="text-sm font-medium">Success Rate</span>
            </div>
            <p className="text-2xl font-bold">{(throughput.success_rate * 100).toFixed(1)}%</p>
          </div>
          <div className="bg-white p-4 rounded-lg border">
            <div className="flex items-center gap-2 text-orange-600 mb-1">
              <Clock className="h-4 w-4" />
              <span className="text-sm font-medium">Queue Depth</span>
            </div>
            <p className="text-2xl font-bold">{throughput.queue_depth}</p>
          </div>
        </div>
      )}

      {/* Containers Table */}
      <div className="bg-white rounded-lg border shadow-sm overflow-hidden">
        <div className="p-4 border-b">
          <div className="flex items-center gap-2">
            <Server className="h-5 w-5 text-gray-500" />
            <h3 className="font-semibold">Docker Containers</h3>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Container</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">CPU</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Memory</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Uptime</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {containers.map((container) => (
                <tr key={container.name} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm font-medium">{container.name}</td>
                  <td className="px-4 py-3 text-sm">
                    <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${
                      container.status === 'running' ? 'bg-green-100 text-green-700' :
                      container.status === 'paused' ? 'bg-yellow-100 text-yellow-700' :
                      'bg-red-100 text-red-700'
                    }`}>
                      <div className={`w-1.5 h-1.5 rounded-full ${
                        container.status === 'running' ? 'bg-green-500' :
                        container.status === 'paused' ? 'bg-yellow-500' :
                        'bg-red-500'
                      }`} />
                      {container.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-500">
                    {container.cpu_percent !== null ? `${container.cpu_percent.toFixed(1)}%` : '-'}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-500">
                    {container.memory_mb !== null ? `${container.memory_mb.toFixed(0)} MB` : '-'}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-500">
                    {container.uptime || '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
