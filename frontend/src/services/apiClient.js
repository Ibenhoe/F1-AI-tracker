import { io } from 'socket.io-client'

const BACKEND_URL = 'http://localhost:5000'

class APIClient {
  constructor() {
    this.listeners = {}
    this.pollingInterval = null
    this.currentRaceId = null
    this.isRaceRunning = false
    this.socket = null
  }

  async connect() {
    // Reuse socket if it already exists and is connected – don't tear it down
    // on every race change.  A single persistent socket is cheaper and avoids
    // the unnecessary disconnect/reconnect cycle that produced the misleading
    // "[SOCKETIO] ERROR: CLIENT DISCONNECTED" messages.
    if (this.socket && this.socket.connected) {
      console.log('[API] Socket already connected, reusing existing connection')
      return Promise.resolve()
    }

    // If there is a stale socket (disconnected), clean it up first
    if (this.socket) {
      this.socket.removeAllListeners()
      this.socket.disconnect()
      this.socket = null
    }

    console.log('[API] Connecting to backend via SocketIO...')

    this.socket = io(BACKEND_URL, {
      // Flask-SocketIO with async_mode='threading' runs on werkzeug which does
      // not support the WebSocket upgrade.  Force polling-only so the client
      // never attempts the WS handshake that produces "Invalid frame header".
      transports: ['polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: 10,
    })

    // Handle connection lifecycle
    this.socket.on('connect', () => {
      console.log('[API] ✅ Connected to backend via SocketIO')
    })

    this.socket.on('disconnect', (reason) => {
      console.log('[API] ⚠️ Disconnected from backend:', reason)
    })

    this.socket.on('connect_error', (err) => {
      console.warn('[API] Connection error:', err.message)
    })

    // Forward all race events into the application listener system
    const forwardEvents = [
      'lap/update',
      'race/ready',
      'race/finished',
      'race/error',
      'race/init-error',
      'init/progress',
    ]
    forwardEvents.forEach((event) => {
      this.socket.on(event, (data) => {
        console.log(`[API-SOCKETIO] Received ${event}`)
        this.triggerListener(event, data)
      })
    })

    // Wait for the actual connect event (not an arbitrary timeout)
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Socket connection timed out after 8 s'))
      }, 8000)

      if (this.socket.connected) {
        clearTimeout(timeout)
        resolve()
      } else {
        this.socket.once('connect', () => {
          clearTimeout(timeout)
          resolve()
        })
        this.socket.once('connect_error', (err) => {
          clearTimeout(timeout)
          reject(err)
        })
      }
    })
  }

  startPolling() {
    // No longer needed with SocketIO, but keep for compatibility
    console.log('[API] Polling disabled (using SocketIO instead)')
  }

  triggerListener(event, data) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(cb => {
        try {
          cb(data)
        } catch (error) {
          console.error(`[API] Error in ${event} listener:`, error)
        }
      })
    }
  }

  // REST API calls
  async healthCheck() {
    const response = await fetch(`${BACKEND_URL}/api/health`)
    return response.json()
  }

  async getRaces() {
    const response = await fetch(`${BACKEND_URL}/api/races`)
    return response.json()
  }

  async initRace(raceNumber) {
    const response = await fetch(`${BACKEND_URL}/api/race/init?race=${raceNumber}`)
    const data = await response.json()
    if (data.race_id) {
      this.currentRaceId = data.race_id
    }
    return data
  }

  on(event, callback) {
    if (!this.listeners[event]) {
      this.listeners[event] = []
    }
    this.listeners[event].push(callback)
  }

  off(event, callback) {
    if (!this.listeners[event]) return
    this.listeners[event] = this.listeners[event].filter(cb => cb !== callback)
  }

  async startRace(speed = 1.0) {
    this.isRaceRunning = true
    const response = await fetch(`${BACKEND_URL}/api/race/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ race_id: this.currentRaceId, speed })
    })
    return response.json()
  }

  async pauseRace() {
    this.isRaceRunning = false
    const response = await fetch(`${BACKEND_URL}/api/race/pause`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ race_id: this.currentRaceId })
    })
    return response.json()
  }

  async resumeRace() {
    this.isRaceRunning = true
    const response = await fetch(`${BACKEND_URL}/api/race/resume`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ race_id: this.currentRaceId })
    })
    return response.json()
  }

  async setSimulationSpeed(speed) {
    const response = await fetch(`${BACKEND_URL}/api/race/speed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ race_id: this.currentRaceId, speed })
    })
    return response.json()
  }

  // Remove all application-level listeners for a given event (or all events)
  clearListeners(event) {
    if (event) {
      delete this.listeners[event]
    } else {
      this.listeners = {}
    }
  }

  disconnect() {
    // Clear application-level listeners so they don't accumulate if the caller
    // re-registers them on the next connect/race-init cycle
    this.clearListeners()

    // Keep the underlying socket alive – only fully tear it down when the app
    // itself unmounts (not on every race change).
    // If a hard disconnect is truly needed, the caller can pass force=true.
    console.log('[API] Listeners cleared (socket kept alive for reconnection)')
  }

  // Hard disconnect – tears down the socket entirely (call on app unmount)
  destroy() {
    this.clearListeners()
    if (this.socket) {
      this.socket.removeAllListeners()
      this.socket.disconnect()
      this.socket = null
    }
    console.log('[API] Socket destroyed')
  }
}

export default new APIClient()
