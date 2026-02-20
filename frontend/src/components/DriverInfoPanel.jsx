import React from 'react';

const TEAM_COLORS = {
  'Mercedes': '#00D7B6',
  'Red Bull Racing': '#4781D7',
  'Ferrari': '#ED1131',
  'McLaren': '#F47600',
  'Alpine': '#00A1E8',
  'Racing Bulls': '#6C98FF',
  'Aston Martin': '#229971',
  'Williams': '#1868DB',
  'Kick Sauber': '#01C00E',
  'Haas': '#9C9FA2',
  'Red Bull': '#4781D7',
};

function tireMeta(compound) {
  const c = String(compound ?? '').toUpperCase();
  if (c === 'SOFT')         return { label: 'S', color: '#ef4444', bg: 'bg-red-500/10',    text: 'text-red-600 dark:text-red-400' };
  if (c === 'MEDIUM')       return { label: 'M', color: '#f59e0b', bg: 'bg-amber-500/10',  text: 'text-amber-600 dark:text-amber-400' };
  if (c === 'HARD')         return { label: 'H', color: '#a3a3a3', bg: 'bg-neutral-300/30', text: 'text-neutral-700 dark:text-neutral-300' };
  if (c === 'INTERMEDIATE') return { label: 'I', color: '#10b981', bg: 'bg-emerald-500/10', text: 'text-emerald-600 dark:text-emerald-400' };
  if (c === 'WET')          return { label: 'W', color: '#3b82f6', bg: 'bg-blue-500/10',   text: 'text-blue-600 dark:text-blue-400' };
  return { label: '?', color: '#888', bg: 'bg-neutral-200/40', text: 'text-neutral-500' };
}

function TelBar({ label, value, max = 100, color = '#ef4444' }) {
  const pct = Math.min(100, Math.max(0, value));
  return (
    <div className="flex items-center gap-2">
      <span className="w-16 flex-shrink-0 text-[11px] font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-400">
        {label}
      </span>
      <div className="flex-1 h-1.5 rounded-full bg-neutral-200 dark:bg-neutral-800 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-100"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="w-10 text-right text-xs tabular-nums font-semibold text-neutral-800 dark:text-neutral-200">
        {Math.round(pct)}%
      </span>
    </div>
  );
}

/**
 * DriverInfoPanel — Tailwind-native driver telemetry card
 */
const DriverInfoPanel = ({ driver, driverData, frame, showTelemetry }) => {
  if (!driver || !driverData) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-neutral-500 dark:text-neutral-400">
        Select a driver to view details
      </div>
    );
  }

  const team     = driverData.team || '';
  const teamColor = TEAM_COLORS[team] || '#888';
  const tire     = tireMeta(driverData.tire_compound);
  const isDNF    = driverData.status === 'OUT';
  const drsOn    = Boolean(driverData.drs);

  return (
    <div className="flex flex-col gap-4">
      {/* ── Header ── */}
      <div
        className="relative overflow-hidden rounded-xl border border-neutral-200/80 bg-white/90 p-4 dark:border-white/10 dark:bg-neutral-950/60"
        style={{ boxShadow: `0 0 0 2px ${teamColor}25, 0 4px 16px ${teamColor}15` }}
      >
        {/* team accent bar */}
        <div className="absolute inset-x-0 top-0 h-1 rounded-t-xl" style={{ backgroundColor: teamColor }} />
        <div className="mt-1 flex items-start justify-between gap-3">
          <div>
            <div className="text-lg font-black tracking-tight text-neutral-900 dark:text-neutral-50">
              {driverData.driver_name || driver}
            </div>
            <div className="mt-0.5 text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-400">
              {team || '—'}
            </div>
          </div>
          <div className="text-right">
            <div className="text-2xl font-black tabular-nums text-neutral-900 dark:text-neutral-50">
              P{Math.round(driverData.position ?? 0)}
            </div>
            {isDNF && (
              <div className="mt-0.5 text-xs font-semibold text-red-500">DNF</div>
            )}
          </div>
        </div>
      </div>

      {/* ── Telemetry ── */}
      {showTelemetry && (
        <div className="rounded-xl border border-neutral-200/80 bg-white/70 p-4 dark:border-white/10 dark:bg-neutral-950/50">
          <div className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-400">Telemetry</div>
          <div className="flex flex-col gap-3">
            {/* Speed */}
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-400">Speed</span>
              <span className="text-sm font-bold tabular-nums text-neutral-900 dark:text-neutral-100">
                {Number(driverData.speed ?? 0).toFixed(0)} <span className="font-normal text-xs text-neutral-500">km/h</span>
              </span>
            </div>
            {/* Gear */}
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-400">Gear</span>
              <span className="text-sm font-bold tabular-nums text-neutral-900 dark:text-neutral-100">
                {driverData.gear ?? '—'}
              </span>
            </div>
            {/* Throttle */}
            <TelBar label="Throttle" value={(driverData.throttle ?? 0) * 100} color="#22c55e" />
            {/* Brake */}
            <TelBar label="Brake" value={(driverData.brake ?? 0) * 100} color="#ef4444" />
            {/* DRS */}
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-400">DRS</span>
              <span className={`rounded-full px-2 py-0.5 text-xs font-bold ${drsOn ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400' : 'bg-neutral-200/60 text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400'}`}>
                {drsOn ? 'ON' : 'OFF'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* ── Tyre ── */}
      <div className="rounded-xl border border-neutral-200/80 bg-white/70 p-4 dark:border-white/10 dark:bg-neutral-950/50">
        <div className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-400">Tyre</div>
        <div className="flex items-center gap-3">
          <div
            className={`flex h-10 w-10 items-center justify-center rounded-full text-base font-black ring-2 ${tire.bg} ${tire.text}`}
            style={{ ringColor: teamColor }}
            title={driverData.tire_compound || '?'}
          >
            {tire.label}
          </div>
          <div>
            <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
              {driverData.tire_compound || '—'}
            </div>
            <div className="mt-0.5 text-xs text-neutral-500 dark:text-neutral-400">
              {driverData.tire_age ?? 0} laps on set
            </div>
          </div>
        </div>
      </div>

      {/* ── Race info ── */}
      <div className="rounded-xl border border-neutral-200/80 bg-white/70 p-4 dark:border-white/10 dark:bg-neutral-950/50">
        <div className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-neutral-500 dark:text-neutral-400">Race info</div>
        <div className="flex flex-col gap-2 text-sm">
          <div className="flex items-center justify-between">
            <span className="text-neutral-500 dark:text-neutral-400">Gap</span>
            <span className="tabular-nums font-medium text-neutral-900 dark:text-neutral-100">
              {Math.round(driverData.position) === 1 ? (
                <span className="font-bold text-yellow-500">LEAD</span>
              ) : (
                driverData.gap ?? '—'
              )}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-neutral-500 dark:text-neutral-400">Pit stops</span>
            <span className="tabular-nums font-medium text-neutral-900 dark:text-neutral-100">
              {driverData.pit_stops ?? 0}
            </span>
          </div>
          {driverData.lap_time && (
            <div className="flex items-center justify-between">
              <span className="text-neutral-500 dark:text-neutral-400">Last lap</span>
              <span className="tabular-nums font-medium text-neutral-900 dark:text-neutral-100">
                {driverData.lap_time}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DriverInfoPanel;
