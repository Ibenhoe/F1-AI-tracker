import React, { useState } from 'react';
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
  'Red Bull': '#4781D7',
};

// Fallback team map by driver code (works even when backend returns "Unknown")
const DRIVER_TEAMS = {
  VER: 'Red Bull Racing', PER: 'Red Bull Racing',
  HAM: 'Mercedes',        RUS: 'Mercedes',
  LEC: 'Ferrari',         SAI: 'Ferrari',
  NOR: 'McLaren',         PIA: 'McLaren',
  ALO: 'Aston Martin',    STR: 'Aston Martin',
  GAS: 'Alpine',          OCO: 'Alpine',
  TSU: 'Racing Bulls',    RIC: 'Racing Bulls',    LAW: 'Racing Bulls',
  ALB: 'Williams',        SAR: 'Williams',        COL: 'Williams',
  HUL: 'Haas',            MAG: 'Haas',            BEA: 'Haas',
  BOT: 'Kick Sauber',     ZHO: 'Kick Sauber',
};

const DRIVER_LAST = {
  VER: 'Verstappen', LEC: 'Leclerc',    SAI: 'Sainz',
  PIA: 'Piastri',    NOR: 'Norris',     HAM: 'Hamilton',
  RUS: 'Russell',    ALO: 'Alonso',     STR: 'Stroll',
  GAS: 'Gasly',      OCO: 'Ocon',       MAG: 'Magnussen',
  HUL: 'Hülkenberg', BOT: 'Bottas',     ZHO: 'Zhou',
  TSU: 'Tsunoda',    ALB: 'Albon',      SAR: 'Sargeant',
  PER: 'Pérez',      RIC: 'Ricciardo',  LAW: 'Lawson',
  BEA: 'Bearman',    COL: 'Colapinto',
};

function tireMeta(compound) {
  const c = String(compound ?? '').toUpperCase();
  if (c.startsWith('S'))                     return { label: 'S', bg: 'bg-red-500/15',    text: 'text-red-500 dark:text-red-400',         ring: 'ring-red-500/40' };
  if (c.startsWith('M'))                     return { label: 'M', bg: 'bg-amber-500/15',  text: 'text-amber-600 dark:text-amber-400',     ring: 'ring-amber-500/40' };
  if (c.startsWith('H'))                     return { label: 'H', bg: 'bg-neutral-400/20', text: 'text-neutral-600 dark:text-neutral-300', ring: 'ring-neutral-400/40' };
  if (c.startsWith('I') || c.startsWith('INT')) return { label: 'I', bg: 'bg-emerald-500/15', text: 'text-emerald-600 dark:text-emerald-400', ring: 'ring-emerald-500/40' };
  if (c.startsWith('W'))                     return { label: 'W', bg: 'bg-blue-500/15',   text: 'text-blue-600 dark:text-blue-400',       ring: 'ring-blue-500/40' };
  return { label: '?', bg: 'bg-neutral-200/40', text: 'text-neutral-500', ring: 'ring-neutral-400/30' };
}

function formatGap(gap) {
  if (gap == null) return '—';
  const num = parseFloat(String(gap).replace('+', ''));
  if (isNaN(num)) return '—';
  if (num >= 60) {
    const m = Math.floor(num / 60);
    const s = (num % 60).toFixed(1).padStart(4, '0');
    return `+${m}:${s}`;
  }
  return `+${num.toFixed(3)}`;
}

/**
 * ReplayLeaderboard – race standings panel with team colors, tire age, and interval toggle
 */
const ReplayLeaderboard = ({ drivers, selectedDriver, onDriverSelect, currentLap, totalLaps }) => {
  const [gapMode, setGapMode] = useState('leader'); // 'leader' | 'interval'

  const sorted = Object.entries(drivers || {})
    .map(([code, d]) => ({ code, ...d }))
    .sort((a, b) => (a.position ?? 999) - (b.position ?? 999));

  // Compute interval (gap to car directly ahead)
  const withInterval = sorted.map((d, i) => {
    if (i === 0) return { ...d, interval: null };
    const myGap   = parseFloat(String(d.gap ?? '').replace('+', '')) || 0;
    const prevGap = parseFloat(String(sorted[i - 1].gap ?? '').replace('+', '')) || 0;
    const iv = Math.max(0, myGap - prevGap);
    return { ...d, interval: `+${iv.toFixed(3)}` };
  });

  const running = sorted.filter(d => d.status !== 'OUT').length;
  const dnfCount = sorted.length - running;

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Header */}
      <div className="flex items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Standings</h2>
          <p className="mt-0.5 text-[11px] text-neutral-500 dark:text-neutral-400">
            {running} running{dnfCount > 0 ? ` · ${dnfCount} DNF` : ''}
          </p>
        </div>
        {currentLap != null && (
          <Badge variant="neutral">Lap {currentLap}{totalLaps ? `/${totalLaps}` : ''}</Badge>
        )}
      </div>

      {/* Gap mode toggle */}
      <div className="flex items-center gap-1 self-start rounded-full border border-neutral-200 bg-neutral-100/60 px-1 py-0.5 dark:border-neutral-800 dark:bg-neutral-900/60">
        {[['leader', 'To leader'], ['interval', 'Interval']].map(([mode, label]) => (
          <button
            key={mode}
            type="button"
            onClick={() => setGapMode(mode)}
            className={[
              'rounded-full px-2.5 py-0.5 text-[11px] font-semibold transition',
              gapMode === mode
                ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900'
                : 'text-neutral-600 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100',
            ].join(' ')}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="flex-1 min-h-0 overflow-y-auto overflow-hidden rounded-xl border border-neutral-200 dark:border-neutral-800">
        {/* Column headers */}
        <div className="grid grid-cols-[26px_1fr_30px_52px] items-center gap-1.5 px-3 py-2 bg-neutral-50 dark:bg-neutral-950/40 text-[10px] font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-400 sticky top-0 z-10 border-b border-neutral-100 dark:border-neutral-800">
          <div className="text-center">P</div>
          <div>Driver</div>
          <div className="text-center">Tyre</div>
          <div className="text-right">{gapMode === 'leader' ? 'Gap' : 'Int.'}</div>
        </div>

        <div className="divide-y divide-neutral-100 dark:divide-neutral-900/80">
          {sorted.length === 0 ? (
            <div className="px-4 py-8 text-center text-xs text-neutral-500">Waiting for data…</div>
          ) : (
            withInterval.map((driver, i) => {
              const isSelected = selectedDriver === driver.code;
              const isLeader   = i === 0;
              const isDNF      = driver.status === 'OUT';
              const tire       = tireMeta(driver.tire_compound);
              const team       = (driver.team && driver.team !== 'Unknown') ? driver.team : (DRIVER_TEAMS[driver.code] || '');
              const teamColor  = TEAM_COLORS[team] || '#666';
              const lastName   = DRIVER_LAST[driver.code] || '';
              const gapStr     = gapMode === 'leader' ? formatGap(driver.gap) : (isLeader ? null : formatGap(driver.interval));

              return (
                <div
                  key={driver.code}
                  onClick={() => onDriverSelect(isSelected ? null : driver.code)}
                  className={[
                    'relative grid grid-cols-[26px_1fr_30px_52px] items-center gap-1.5 px-3 py-2 cursor-pointer transition-colors select-none',
                    isSelected
                      ? 'bg-neutral-100 dark:bg-neutral-800/70'
                      : 'bg-white dark:bg-neutral-950/20 hover:bg-neutral-50 dark:hover:bg-neutral-950/50',
                    isDNF ? 'opacity-40' : '',
                  ].join(' ')}
                >
                  {/* Left accent bar */}
                  <div
                    className="absolute left-0 top-0 h-full w-0.5"
                    style={{ backgroundColor: isSelected ? teamColor : `${teamColor}99` }}
                  />

                  {/* Position */}
                  <div className={[
                    'text-center text-xs font-black tabular-nums',
                    isLeader && !isDNF ? 'text-yellow-500 dark:text-yellow-400' : 'text-neutral-700 dark:text-neutral-300',
                  ].join(' ')}>
                    {isDNF ? <span className="text-[9px] font-bold text-red-500">OUT</span> : Math.round(driver.position)}
                  </div>

                  {/* Driver */}
                  <div className="min-w-0">
                    <div className="flex items-baseline gap-1 min-w-0">
                      <span className={['text-xs font-black tracking-tight', isSelected ? 'text-neutral-900 dark:text-white' : 'text-neutral-900 dark:text-neutral-100'].join(' ')}>
                        {driver.code}
                      </span>
                      {lastName && (
                        <span className="truncate text-[10px] text-neutral-500 dark:text-neutral-400">
                          {lastName}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      {team && <span className="text-[9px] text-neutral-500 dark:text-neutral-500 truncate leading-none">{team.replace(' Racing', '').replace(' F1 Team', '')}</span>}
                      {driver.pit_stops > 0 && <span className="text-[9px] text-neutral-400 dark:text-neutral-600 leading-none">· P{driver.pit_stops}</span>}
                    </div>
                  </div>

                  {/* Tyre badge + age */}
                  <div className="flex flex-col items-center gap-0.5">
                    <span
                      className={['inline-flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-black ring-1', tire.bg, tire.text, tire.ring].join(' ')}
                      title={`${driver.tire_compound || '?'} – ${driver.tire_age ?? 0} laps old`}
                    >
                      {tire.label}
                    </span>
                    {driver.tire_age != null && (
                      <span className="text-[9px] tabular-nums text-neutral-400 dark:text-neutral-600 leading-none">
                        {driver.tire_age}L
                      </span>
                    )}
                  </div>

                  {/* Gap / Interval */}
                  <div className="text-right">
                    {isLeader ? (
                      <span className="text-[11px] font-black text-yellow-500 dark:text-yellow-400">LEAD</span>
                    ) : gapStr && gapStr !== '—' ? (
                      <span className="text-[11px] tabular-nums font-medium text-neutral-600 dark:text-neutral-300">
                        {gapStr}
                      </span>
                    ) : (
                      <span className="text-[11px] text-neutral-400">—</span>
                    )}
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
