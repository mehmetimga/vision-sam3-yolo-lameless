/**
 * WebSocket Hook
 * Provides real-time connection to backend WebSocket channels
 */
import { useState, useEffect, useRef, useCallback } from 'react'

type WebSocketStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

interface WebSocketMessage {
  type: string
  timestamp?: string
  [key: string]: unknown
}

interface UseWebSocketOptions {
  onMessage?: (message: WebSocketMessage) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
  reconnectAttempts?: number
  reconnectInterval?: number
  autoConnect?: boolean
}

interface UseWebSocketReturn {
  status: WebSocketStatus
  lastMessage: WebSocketMessage | null
  connect: () => void
  disconnect: () => void
  send: (message: object) => void
  isConnected: boolean
}

const WS_BASE_URL = import.meta.env.VITE_WS_URL ||
  (window.location.protocol === 'https:' ? 'wss://' : 'ws://') +
  (import.meta.env.VITE_API_URL?.replace(/^https?:\/\//, '') || 'localhost:8000')

export function useWebSocket(
  channel: 'pipeline' | 'health' | 'queue' | 'rater',
  options: UseWebSocketOptions = {}
): UseWebSocketReturn {
  const {
    onMessage,
    onConnect,
    onDisconnect,
    onError,
    reconnectAttempts = 5,
    reconnectInterval = 3000,
    autoConnect = true
  } = options

  const [status, setStatus] = useState<WebSocketStatus>('disconnected')
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectCountRef = useRef(0)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const clearTimers = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current)
      pingIntervalRef.current = null
    }
  }, [])

  const disconnect = useCallback(() => {
    clearTimers()
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setStatus('disconnected')
  }, [clearTimers])

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    clearTimers()
    setStatus('connecting')

    try {
      const wsUrl = `${WS_BASE_URL}/api/ws/${channel}`
      wsRef.current = new WebSocket(wsUrl)

      wsRef.current.onopen = () => {
        setStatus('connected')
        reconnectCountRef.current = 0
        onConnect?.()

        // Set up ping interval
        pingIntervalRef.current = setInterval(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send('ping')
          }
        }, 25000)
      }

      wsRef.current.onmessage = (event) => {
        try {
          // Handle pong response
          if (event.data === 'pong') {
            return
          }

          const message: WebSocketMessage = JSON.parse(event.data)

          // Handle ping from server
          if (message.type === 'ping') {
            wsRef.current?.send('pong')
            return
          }

          setLastMessage(message)
          onMessage?.(message)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      wsRef.current.onclose = () => {
        setStatus('disconnected')
        onDisconnect?.()
        clearTimers()

        // Attempt reconnection
        if (reconnectCountRef.current < reconnectAttempts) {
          reconnectCountRef.current += 1
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectInterval)
        }
      }

      wsRef.current.onerror = (error) => {
        setStatus('error')
        onError?.(error)
      }
    } catch (error) {
      console.error('WebSocket connection error:', error)
      setStatus('error')
    }
  }, [channel, onConnect, onDisconnect, onError, onMessage, reconnectAttempts, reconnectInterval, clearTimers])

  const send = useCallback((message: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    }
  }, [])

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, [autoConnect, connect, disconnect])

  return {
    status,
    lastMessage,
    connect,
    disconnect,
    send,
    isConnected: status === 'connected'
  }
}

// Convenience hooks for specific channels
export function usePipelineWebSocket(options?: UseWebSocketOptions) {
  return useWebSocket('pipeline', options)
}

export function useHealthWebSocket(options?: UseWebSocketOptions) {
  return useWebSocket('health', options)
}

export function useQueueWebSocket(options?: UseWebSocketOptions) {
  return useWebSocket('queue', options)
}

export function useRaterWebSocket(options?: UseWebSocketOptions) {
  return useWebSocket('rater', options)
}
