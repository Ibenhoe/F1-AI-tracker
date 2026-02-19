import React from 'react';
import Badge from './ui/Badge.jsx';

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
};

function tireMeta(compound) {
  const c = String(compound ?? '').toUpperCase();
  if (c === 'SOFT')         return { label: 'S', bg: 'bg-red-500/10 dark:bg-red-500/10',     text: 'text-red-700 dark:text-red-200',         ring: 'ring-red-500/30' };
  if (c === 'MEDIUM')       return { label: 'M', bg: 'bg-amber-500/10 dark:bg-amber-500/10', text: 'text-amber-800 dark:text-amber-200',       ring: 'ring-amber-500/30' };
  if (c === 'HARD')         return { label: 'H', bg: 'bg-neutral-200/60 dark:bg-neutral-200/10', text: 'text-neutral-800 dark:text-neutral-200', ring: 'ring-neutral-400/30' };
  if (c === 'INTERMEDIATE') return { label: 'I', bg: 'bg-emerald-500/10 dark:bg-emerald-500/10', text: 'text-emerald-800 dark:text-emerald-200', ring: 'ring-emerald-500/30' };
  if (c === 'WET')          return { label: 'W', bg: 'bg-blue-500/10 dark:bg-blue-500/10',   text: 'text-blue-800 dark:text-blue-200',         ring: 'ring-blue-500/30' };
  return { label: '?', bg: 'bg-neutral-100 dark:bg-neutral-900', text: 'text-neutral-700 dark:text-neutral-300', ring: 'ring-neutral-300 dark:ring-neutral-700' };
}

function gapDisplay(gap, isLeader) {
  if (isLeader) return <span className="text-xs font-bold text-yellow-500 dark:text-yellow-400">LEAD</span>;
  // null/undefined = no data yet
  if (gap == null) return <span className="text-xs text-neutral-600 dark:text-neutral-600">—</span>;
  const num = parseFloat(String(gap).replace('+', ''));
  if (isNaN(num)) return <span className="text-xs text-neutral-600 dark:text-neutral-600">—</span>;
  // Show millisecond precision like real F1 timing: +0.123s
  return (
    <span className="text-xs tabular-nums text-neutral-300 dark:text-neutral-300">
      +{num.toFixed(3)}s
    </span>
  );
}

/**
 * ReplayLeaderboard – compact left-panel standings
 * Styled to match the DriversList design language.
 */
const ReplayLeaderboard = ({
  drivers,
  selectedDriver,
  onDriverSelect,
  currentLap,
}) => {
  const sorted = Object.entries(drivers || {})
    .map(([code, d]) => ({ code, ...d }))
    .sort((a, b) => (a.position ?? 999) - (b.position ?? 999));

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Header */}
      <div className="flex items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Standings</h2>
          <p className="mt-0.5 text-xs text-neutral-500 dark:text-neutral-400">
            Live classification
          </p>
        </div>
        {currentLap != null && (
          <Badge variant="neutral">Lap {currentLap}</Badge>
        )}
      </div>

      {/* Table */}
      <div className="overflow-hidden rounded-xl border border-neutral-200 dark:border-neutral-800 flex-1 overflow-y-auto">
        {/* Column header */}
        <div className="grid grid-cols-[28px_1fr_36px_56px] items-center gap-2 px-3 py-2 bg-neutral-50 dark:bg-neutral-950/40 text-xs font-medium text-neutral-500 dark:text-neutral-400 sticky top-0 z-10">
          <div className="text-center">P</div>
          <div>Driver</div>
          <div className="text-center">Tyre</div>
          <div className="text-right">Gap</div>
        </div>

        <div className="divide-y divide-neutral-200 dark:divide-neutral-900">
          {sorted.length === 0 ? (
            <div className="px-4 py-8 text-center text-xs text-neutral-500 dark:text-neutral-500">
              Waiting for data…
            </div>
          ) : (
            sorted.map((driver) => {
              const isSelected = selectedDriver === driver.code;
              const isLeader = Math.round(driver.position) === 1;
              const meta = tireMeta(driver.tire_compound);
              const team = driver.team || '';
              const teamColor = TEAM_COLORS[team] || '#555';
              const isDNF = driver.status === 'OUT';

              return (
                <div
                  key={driver.code}
                  onClick={() => onDriverSelect(isSelected ? null : driver.code)}
                  className={[
                    'relative grid grid-cols-[28px_1fr_36px_56px] items-center gap-2 px-3 py-2 cursor-pointer transition-colors',
                    isSelected
                      ? 'bg-neutral-100 dark:bg-neutral-800/60'
                      : 'bg-white dark:bg-neutral-950/30 hover:bg-neutral-50 dark:hover:bg-neutral-950/50',
                    isDNF ? 'opacity-50' : '',
                  ].join(' ')}
                >
                  {/* Team colour bar */}
                  <div
                    className="absolute left-0 top-0 h-full w-1 rounded-l"
                    style={{ backgroundColor: teamColor }}
                  />

                  {/* Position */}
                  <div className={[
                    'text-center text-xs font-bold tabular-nums',
                    isLeader ? 'text-yellow-500 dark:text-yellow-400' : 'text-neutral-900 dark:text-neutral-100',
                  ].join(' ')}>
                    {Math.round(driver.position)}
                  </div>

                  {/* Driver name + team */}
                  <div className="min-w-0">
                    <div className={[
                      'truncate text-xs font-semibold',
                      isSelected ? 'text-neutral-900 dark:text-white' : 'text-neutral-900 dark:text-neutral-100',
                    ].join(' ')}>
                      {driver.code}
                    </div>
                    <div className="truncate text-[10px] text-neutral-500 dark:text-neutral-500 leading-tight">
                      {team || '—'}
                    </div>
                    {driver.pit_stops > 0 && (
                      <div className="text-[10px] text-neutral-400 dark:text-neutral-600 leading-tight">
                        🛠 {driver.pit_stops}
                      </div>
                    )}
                  </div>

                  {/* Tyre badge */}
                  <div className="flex justify-center">
                    <span
                      className={[
                        'inline-flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold ring-1',
                        meta.bg, meta.text, meta.ring,
                      ].join(' ')}
                      title={`${driver.tire_compound || '?'} – ${driver.tire_age ?? 0} laps`}
                    >
                      {meta.label}
                    </span>
                  </div>

                  {/* Gap */}
                  <div className="text-right">
                    {gapDisplay(driver.gap, isLeader)}
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};

export default ReplayLeaderboard;
