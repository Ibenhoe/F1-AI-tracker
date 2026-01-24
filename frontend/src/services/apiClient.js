const BACKEND_URL = 'http://localhost:5000'

class APIClient {
  constructor() {
    this.listeners = {}
    this.pollingInterval = null
    this.currentRaceId = null
    this.isRaceRunning = false
  }

  async connect() {
    console.log('[API] Connecting to backend...')
    try {
      const response = await fetch(`${BACKEND_URL}/api/health`)
      if (!response.ok) throw new Error('Health check failed')
      console.log('[API] ✅ Connected to backend')
      this.startPolling()
      return Promise.resolve()
    } catch (error) {
      console.error('[API] Connection failed:', error)
      throw error
    }
  }

  startPolling() {
    // Poll every 1 second when race is running
    this.pollingInterval = setInterval(async () => {
      if (this.currentRaceId) {
        try {
          const response = await fetch(`${BACKEND_URL}/api/race/state`)
          if (response.ok) {
            const data = await response.json()
            // Always send update if we have race state
            if (data.drivers !== undefined) {
              this.triggerListener('lap/update', data)
            }
          }
        } catch (error) {
          // Silent retry
        }
      }
    }, 1000)  // Poll every 1 second
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
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval)
      this.pollingInterval = null
    }
  }
}

export default new APIClient()
