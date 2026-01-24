import { useState } from 'react'
import './App.css'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import PreRaceAnalysis from './pages/PreRaceAnalysis'

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard')

  return (
    <div className="app-container">
      <Sidebar currentPage={currentPage} setCurrentPage={setCurrentPage} />
      <main className="main-content">
        {currentPage === 'dashboard' && <Dashboard />}
        {currentPage === 'pre-race' && <PreRaceAnalysis />}
      </main>
    </div>
  )
}

export default App
