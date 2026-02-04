const BACKEND_URL = 'http://localhost:5000'

// Import Socket.IO client
import { io } from 'socket.io-client'

class APIClient {
  constructor() {
    this.listeners = {}
    this.pollingInterval = null
    this.currentRaceId = null
    this.isRaceRunning = false
    this.socket = null
  }

  async connect() {
    console.log('[API] Connecting to backend via SocketIO...')
    try {
      // Connect via SocketIO
      this.socket = io(BACKEND_URL, {
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: 5
      })
      
      // Handle connection
      this.socket.on('connect', () => {
        console.log('[API] ✅ Connected to backend via SocketIO')
      })
      
      // Handle disconnect
      this.socket.on('disconnect', () => {
        console.log('[API] ⚠️ Disconnected from backend')
      })
      
      // Listen for all race events
      this.socket.on('lap/update', (data) => {
        console.log('[API-SOCKETIO] Received lap/update:', data)
        this.triggerListener('lap/update', data)
      })
      
      this.socket.on('race/ready', (data) => {
        console.log('[API-SOCKETIO] Received race/ready:', data)
        this.triggerListener('race/ready', data)
      })
      
      this.socket.on('race/finished', (data) => {
        console.log('[API-SOCKETIO] Received race/finished:', data)
        this.triggerListener('race/finished', data)
      })
      
      this.socket.on('race/error', (data) => {
        console.log('[API-SOCKETIO] Received race/error:', data)
        this.triggerListener('race/error', data)
      })
      
      return new Promise((resolve) => {
        setTimeout(resolve, 500)
      })
    } catch (error) {
      console.error('[API] Connection failed:', error)
      throw error
    }
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

  disconnect() {
    if (this.socket) {
      this.socket.disconnect()
      this.socket = null
    }
    console.log('[API] Disconnected from backend')
  }
}

export default new APIClient()
