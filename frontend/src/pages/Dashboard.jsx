import { useState, useEffect } from 'react'

import WeatherWidget from '../components/WeatherWidget'
import RaceInfo from '../components/RaceInfo'
import DriversList from '../components/DriversList'
import PredictionsPanel from '../components/PredictionsPanel'
import NotificationsPanel from '../components/NotificationsPanel'
import RaceControls from '../components/RaceControls'
import RaceSelector from '../components/RaceSelector'
import apiClient from '../services/apiClient'

import Card from '../components/ui/Card'
import Badge from '../components/ui/Badge'

export default function Dashboard() {
  const [raceData, setRaceData] = useState(null)
  const [weatherData, setWeatherData] = useState(null)
  const [predictions, setPredictions] = useState([])
  const [modelMetrics, setModelMetrics] = useState(null)
  const [notifications, setNotifications] = useState([])
  const [currentLap, setCurrentLap] = useState(0)
  const [raceRunning, setRaceRunning] = useState(false)
  const [connected, setConnected] = useState(false)
  const [raceInitialized, setRaceInitialized] = useState(false)
  const [selectedRace, setSelectedRace] = useState(21)

  // Initialize connection and race
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Connect to backend
        await apiClient.connect()
        setConnected(true)
        console.log('[DASHBOARD] Connected to backend')

        // Initialize race with selected race number
        const result = await apiClient.initRace(selectedRace)
        console.log('[DASHBOARD] Race initialized:', result)
        
        setRaceData({
          race: result.race_name || `Race ${selectedRace}`,
          totalLaps: result.total_laps || 58,
          currentLap: 0,
          drivers: result.drivers || []
        })
        
        // Set initial weather data
        if (result.weather) {
          setWeatherData({
            temp: result.weather.air_temp || 25,
            humidity: result.weather.humidity || 50,
            trackTemp: result.weather.track_temp || 35,
            windSpeed: result.weather.wind_speed || 0,
            condition: result.weather.conditions || 'Dry'
          })
        }
        
        setRaceInitialized(true)

        // Setup WebSocket listeners
        apiClient.on('lap/update', (data) => {
          console.log('[DASHBOARD] Lap update:', data.lap_number)
          setCurrentLap(data.lap_number)
          setRaceData(prev => ({
            ...prev,
            currentLap: data.lap_number,
            drivers: data.drivers
          }))
          setPredictions(data.predictions)
          
          // Update model metrics if available
          if (data.model_metrics) {
            setModelMetrics(data.model_metrics)
            console.log('[DASHBOARD] Model metrics updated:', data.model_metrics)
          }
          
          // Update weather data if available
          if (data.weather) {
            setWeatherData({
              temp: data.weather.air_temp || 25,
              humidity: data.weather.humidity || 50,
              trackTemp: data.weather.track_temp || 35,
              windSpeed: data.weather.wind_speed || 0,
              condition: data.weather.conditions || 'Dry'
            })
          }
          
          // Add notifications for events
          if (data.events && data.events.length > 0) {
            setNotifications(prev => [
              ...data.events.map((e, i) => ({
                id: Date.now() + i,
                type: 'info',
                message: e.message,
                time: new Date().toLocaleTimeString()
              })),
              ...prev.slice(0, 4)
            ])
          }
        })

        apiClient.on('race/started', (data) => {
          setRaceRunning(true)
          setNotifications(prev => [{
            id: Date.now(),
            type: 'success',
            message: 'Race started!',
            time: new Date().toLocaleTimeString()
          }, ...prev])
        })

        apiClient.on('race/paused', () => {
          setRaceRunning(false)
        })

        apiClient.on('race/resumed', () => {
          setRaceRunning(true)
        })

        apiClient.on('race/finished', (data) => {
          setRaceRunning(false)
          setNotifications(prev => [{
            id: Date.now(),
            type: 'success',
            message: 'Race finished!',
            time: new Date().toLocaleTimeString()
          }, ...prev])
        })

        apiClient.on('race/error', (data) => {
          setNotifications(prev => [{
            id: Date.now(),
            type: 'error',
            message: data.error,
            time: new Date().toLocaleTimeString()
          }, ...prev])
        })

      } catch (error) {
        console.error('[DASHBOARD] Initialization error:', error)
        setNotifications([{
          id: Date.now(),
          type: 'error',
          message: `Connection error: ${error.message}`,
          time: new Date().toLocaleTimeString()
        }])
      }
    }

    initializeApp()

    return () => {
      apiClient.disconnect()
    }
  }, [selectedRace])

  const handleRaceChange = async (newRaceNumber) => {
    setSelectedRace(newRaceNumber)
    setRaceInitialized(false)
    setCurrentLap(0)
    setRaceRunning(false)
  }

    return (
    <div className="space-y-6">
      {/* HEADER */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold tracking-tight">Dashboard</h1>
          <p className="text-sm text-neutral-400">
            Live race tracking, model predictions and telemetry summaries.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={connected ? "success" : "danger"}>
            {connected ? "Connected" : "Disconnected"}
          </Badge>
          <Badge variant={raceRunning ? "warning" : "neutral"}>
            {raceRunning ? "Race running" : "Paused"}
          </Badge>
          <Badge variant="neutral">Lap {currentLap}</Badge>
        </div>
      </div>

      {/* TOOLBAR (selector + controls) */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="lg:col-span-5">
          <Card className="p-4">
            <RaceSelector
              selectedRace={selectedRace}
              onRaceChange={handleRaceChange}
              disabled={raceRunning}
            />
          </Card>
        </div>

        <div className="lg:col-span-7">
          <Card className="p-4">
            <RaceControls
              raceInitialized={raceInitialized}
              raceRunning={raceRunning}
              connected={connected}
            />
          </Card>
        </div>
      </div>

      {/* TOP WIDGETS */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card>
          <WeatherWidget data={weatherData} />
        </Card>
        <Card>
          <RaceInfo data={raceData} />
        </Card>
      </div>

      {/* MAIN GRID */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
        <div className="xl:col-span-4">
          <Card className="h-full">
            <DriversList
              drivers={raceData?.drivers || []}
              currentLap={raceData?.currentLap}
            />
          </Card>
        </div>

        <div className="xl:col-span-5">
          <Card className="h-full">
            <PredictionsPanel
              predictions={predictions}
              currentLap={raceData?.currentLap}
              modelMetrics={modelMetrics}
              totalLaps={raceData?.totalLaps}
            />
          </Card>
        </div>

        <div className="xl:col-span-3">
          <Card className="h-full">
            <NotificationsPanel notifications={notifications} />
          </Card>
        </div>
      </div>
    </div>
  );
}
