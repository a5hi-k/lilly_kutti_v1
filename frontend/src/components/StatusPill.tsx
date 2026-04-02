export function StatusPill({ connected }: { connected: boolean }) {
  return (
    <div className="app-status">
      <span className={`status-indicator ${connected ? 'online' : 'offline'}`} />
      <span>{connected ? 'Online' : 'Offline'}</span>
    </div>
  )
}

