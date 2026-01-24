import './NotificationsPanel.css'

export default function NotificationsPanel({ notifications }) {
  return (
    <div className="notifications-container">
      <div className="notifications-header">
        <h2>🔔 Meldingen</h2>
      </div>
      <div className="notifications-list">
        {notifications.map((notif) => (
          <div key={notif.id} className={`notification-item ${notif.type}`}>
            <div className="notification-icon">
              {notif.type === 'info' && '📌'}
              {notif.type === 'warning' && '⚠️'}
              {notif.type === 'success' && '✅'}
              {notif.type === 'error' && '❌'}
            </div>
            <div className="notification-content">
              <div className="notification-message">{notif.message}</div>
              <div className="notification-time">{notif.time}</div>
            </div>
          </div>
        ))}
      </div>
      {notifications.length === 0 && (
        <div className="empty-state">
          <p>Geen meldingen</p>
        </div>
      )}
    </div>
  )
}
