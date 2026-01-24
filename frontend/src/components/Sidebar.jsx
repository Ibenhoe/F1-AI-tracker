import './Sidebar.css'

export default function Sidebar({ currentPage, setCurrentPage }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1>F1 AI Tracker</h1>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-section">
          <h2 className="nav-title">Algemeen</h2>
          <button
            className={`nav-button ${currentPage === 'dashboard' ? 'active' : ''}`}
            onClick={() => setCurrentPage('dashboard')}
          >
            <span className="nav-icon">📊</span>
            Dashboard
          </button>
        </div>

        <div className="nav-section">
          <h2 className="nav-title">Analyse</h2>
          <button
            className={`nav-button ${currentPage === 'pre-race' ? 'active' : ''}`}
            onClick={() => setCurrentPage('pre-race')}
          >
            <span className="nav-icon">📈</span>
            Pre-Race Analyse
          </button>
        </div>
      </nav>

      <div className="sidebar-footer">
        <p>F1 AI Tracker v1.0</p>
      </div>
    </aside>
  )
}
