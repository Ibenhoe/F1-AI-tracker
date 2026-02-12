/**
 * Simple HTTP polling client instead of Socket.IO
 * More reliable for development with werkzeug server
 */

class SimpleAPIClient {
  constructor() {
    this.listeners = {}
    this.pollingInterval = null
    this.isConnected = false
    this.lastEventId = 0
  }

  connect() {
    return new Promise((resolve) => {
      console.log('[API] Connecting via HTTP polling...')
      
      this.isConnected = true
      
      // Start polling for events
      this.startPolling()
      
      // Simulate immediate connection
      setTimeout(() => {
        console.log('[API] ✅ Connected to backend (via polling)')
        resolve()
      }, 500)
    })
  }

  startPolling() {
    // Poll every 1 second for events
    this.pollingInterval = setInterval(async () => {
      try {
        const response = await fetch('http://localhost:5000/api/events')
        if (response.ok) {
          const data = await response.json()
          if (data.events && data.events.length > 0) {
            data.events.forEach(event => {
              this.triggerListeners(event.type, event.data)
            })
          }
        }
      } catch (error) {
        console.warn('[API] Polling request failed:', error.message)
      }
    }, 1000)
  }

  disconnect() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval)
      this.pollingInterval = null
    }
    this.isConnected = false
    console.log('[API] Disconnected from backend')
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

  triggerListeners(event, data) {
    if (this.listeners[event]) {
      this.listeners[event].forEach(callback => {
        try {
          callback(data)
        } catch (error) {
          console.error(`[API] Error in listener for ${event}:`, error)
        }
      })
    }
  }

  emit(event, data) {
    console.log('[API] Emitting:', event, data)
    // Send to backend endpoint
    fetch('http://localhost:5000/api/emit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event, data })
    }).catch(error => {
      console.error('[API] Emit failed:', error)
    })
  }

  // API methods
  async healthCheck() {
    try {
      const response = await fetch('http://localhost:5000/api/health')
      return await response.json()
    } catch (error) {
      console.error('[API] Health check failed:', error)
      return { status: 'error' }
    }
  }

  async getRaces() {
    try {
      const response = await fetch('http://localhost:5000/api/races')
      return await response.json()
    } catch (error) {
      console.error('[API] Get races failed:', error)
      return {}
    }
  }

  async initRace(raceNumber) {
    try {
      const response = await fetch(`http://localhost:5000/api/race/init?race=${raceNumber}`)
      return await response.json()
    } catch (error) {
      console.error('[API] Init race failed:', error)
      return { success: false }
    }
  }

  // Race control - HTTP REST endpoints
  async startRace(speed = 1.0) {
    try {
      const response = await fetch('http://localhost:5000/api/race/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed })
      })
      const data = await response.json()
      this.triggerListeners('race/start', data)
      return data
    } catch (error) {
      console.error('[API] Start race failed:', error)
      return { error: error.message }
    }
  }

  async pauseRace() {
    try {
      const response = await fetch('http://localhost:5000/api/race/pause', {
        method: 'POST'
      })
      const data = await response.json()
      this.triggerListeners('race/pause', data)
      return data
    } catch (error) {
      console.error('[API] Pause race failed:', error)
      return { error: error.message }
    }
  }

  async resumeRace() {
    try {
      const response = await fetch('http://localhost:5000/api/race/resume', {
        method: 'POST'
      })
      const data = await response.json()
      this.triggerListeners('race/resume', data)
      return data
    } catch (error) {
      console.error('[API] Resume race failed:', error)
      return { error: error.message }
    }
  }

  async setSimulationSpeed(speed) {
    try {
      const response = await fetch('http://localhost:5000/api/race/speed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed })
      })
      const data = await response.json()
      this.triggerListeners('race/speed', data)
      return data
    } catch (error) {
      console.error('[API] Set speed failed:', error)
      return { error: error.message }
    }
  }

  async getRaceState() {
    try {
      const response = await fetch('http://localhost:5000/api/race/state')
      return await response.json()
    } catch (error) {
      console.error('[API] Get race state failed:', error)
      return { drivers: [], current_lap: 0 }
    }
  }
}

export default new SimpleAPIClient()
