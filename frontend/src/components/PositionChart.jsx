/**
 * PositionChart – SVG multi-line chart of driver positions across laps.
 * No external chart library required.
 */

const TEAM_COLORS = {
  "Mercedes":         "#00D7B6",
  "Red Bull Racing":  "#4781D7",
  "Ferrari":          "#ED1131",
  "McLaren":          "#F47600",
  "Alpine":           "#00A1E8",
  "Racing Bulls":     "#6C98FF",
  "Aston Martin":     "#229971",
  "Williams":         "#1868DB",
  "Kick Sauber":      "#01C00E",
  "Haas":             "#9C9FA2",
};

const FALLBACK_COLORS = [
  "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
  "#1abc9c","#e67e22","#e91e63","#00bcd4","#8bc34a",
  "#ff5722","#607d8b","#795548","#ffc107","#673ab7",
  "#03a9f4","#4caf50","#ff9800","#f44336","#9c27b0",
];

function getDriverColor(driver, index) {
  return TEAM_COLORS[driver.team] || FALLBACK_COLORS[index % FALLBACK_COLORS.length];
}

export default function PositionChart({ lapHistory = [], totalLaps = 60, highlightedDrivers = null }) {
  if (!lapHistory || lapHistory.length < 2) {
    return (
      <div className="flex h-48 items-center justify-center text-sm text-neutral-500 dark:text-neutral-400">
        Position history will appear once the race begins…
      </div>
    );
  }

  // Collect all driver codes in the order they first appear (for stable color assignment)
  const driverMeta = {};
  for (const frame of lapHistory) {
    for (const d of frame.drivers) {
      if (!driverMeta[d.code]) {
        driverMeta[d.code] = {
          team: d.team,
          colorIndex: Object.keys(driverMeta).length,
        };
      }
    }
  }

  const driverCodes = Object.keys(driverMeta);

  // SVG layout constants
  const W = 600;
  const H = 220;
  const PAD = { top: 10, right: 52, bottom: 24, left: 28 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const maxPos = 20;
  const minLap = lapHistory[0].lap;
  const maxLap = Math.max(lapHistory[lapHistory.length - 1].lap, minLap + 1);

  const xScale = (lap) => PAD.left + ((lap - minLap) / (maxLap - minLap)) * innerW;
  const yScale = (pos) => PAD.top + ((pos - 1) / (maxPos - 1)) * innerH;

  // Build polyline points per driver
  const lines = driverCodes.map((code, i) => {
    const pts = lapHistory
      .map((frame) => {
        const d = frame.drivers.find((x) => x.code === code);
        if (!d) return null;
        return `${xScale(frame.lap).toFixed(1)},${yScale(d.position).toFixed(1)}`;
      })
      .filter(Boolean)
      .join(" ");

    const lastFrame = [...lapHistory].reverse().find((f) => f.drivers.find((x) => x.code === code));
    const lastDriver = lastFrame?.drivers.find((x) => x.code === code);
    const color = getDriverColor({ team: driverMeta[code].team }, i);
    const isHighlighted = !highlightedDrivers || highlightedDrivers.includes(code);

    return { code, pts, color, lastDriver, lastFrame, isHighlighted };
  });

  // Y-axis tick positions: 1, 5, 10, 15, 20
  const yTicks = [1, 5, 10, 15, 20];

  // X-axis: show every ~10 laps
  const lapSpan = maxLap - minLap;
  const xTickStep = lapSpan <= 20 ? 5 : lapSpan <= 40 ? 10 : 15;
  const xTicks = [];
  for (let l = minLap; l <= maxLap; l += xTickStep) xTicks.push(l);
  if (!xTicks.includes(maxLap)) xTicks.push(maxLap);

  return (
    <div className="w-full overflow-x-auto">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ minWidth: 320, maxHeight: 260 }}
        aria-label="Driver position history chart"
      >
        {/* subtle grid lines */}
        {yTicks.map((pos) => (
          <line
            key={pos}
            x1={PAD.left}
            y1={yScale(pos)}
            x2={W - PAD.right}
            y2={yScale(pos)}
            stroke="currentColor"
            strokeWidth={0.5}
            className="text-neutral-200 dark:text-neutral-800"
            strokeDasharray="3 3"
          />
        ))}

        {/* Y-axis labels */}
        {yTicks.map((pos) => (
          <text
            key={pos}
            x={PAD.left - 6}
            y={yScale(pos) + 4}
            textAnchor="end"
            fontSize={9}
            className="fill-neutral-500 dark:fill-neutral-500 font-mono"
          >
            P{pos}
          </text>
        ))}

        {/* X-axis labels */}
        {xTicks.map((lap) => (
          <text
            key={lap}
            x={xScale(lap)}
            y={H - 6}
            textAnchor="middle"
            fontSize={9}
            className="fill-neutral-500 dark:fill-neutral-500 font-mono"
          >
            {lap}
          </text>
        ))}

        {/* X-axis label "Lap" */}
        <text
          x={PAD.left + innerW / 2}
          y={H - 0}
          textAnchor="middle"
          fontSize={8}
          className="fill-neutral-400 dark:fill-neutral-600"
        >
          LAPS
        </text>

        {/* Driver lines — dimmed ones first, highlighted on top */}
        {[...lines.filter((l) => !l.isHighlighted), ...lines.filter((l) => l.isHighlighted)].map((l) => (
          <g key={l.code}>
            <polyline
              points={l.pts}
              fill="none"
              stroke={l.color}
              strokeWidth={l.isHighlighted ? 2 : 1}
              strokeOpacity={l.isHighlighted ? 0.9 : 0.25}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            {/* driver label at end of line */}
            {l.lastDriver && l.isHighlighted && (
              <text
                x={xScale(l.lastFrame.lap) + 4}
                y={yScale(l.lastDriver.position) + 4}
                fontSize={9}
                fontWeight="600"
                fill={l.color}
                opacity={0.95}
              >
                {l.code}
              </text>
            )}
          </g>
        ))}

        {/* Dot at current position for highlighted drivers */}
        {lines
          .filter((l) => l.isHighlighted && l.lastDriver)
          .map((l) => (
            <circle
              key={`dot-${l.code}`}
              cx={xScale(l.lastFrame.lap)}
              cy={yScale(l.lastDriver.position)}
              r={3}
              fill={l.color}
              opacity={0.95}
            />
          ))}
      </svg>
    </div>
  );
}
