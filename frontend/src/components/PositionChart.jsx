import { TEAM_COLORS } from "../utils/teamColors";

function getDriverColor(team) {
  return TEAM_COLORS?.[team] || "currentColor";
}

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

function snapHalf(v) {
  return Math.round(v * 2) / 2;
}

export default function PositionChart({
  lapHistory = [],
  highlightedDrivers = null,
}) {
  if (!lapHistory || lapHistory.length < 2) {
    return (
      <div className="flex h-48 items-center justify-center text-sm text-neutral-500 dark:text-neutral-400">
        Position history will appear once the race begins…
      </div>
    );
  }

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

  const driverCodes = Object.keys(driverMeta).slice(0, 20);

  const W = 1400;
  const H = 680;
  const pillW = 120;
  const pillGutter = 10;
  const PAD = { top: 20, right: pillW + pillGutter * 2, bottom: 48, left: 48 };

  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const maxPos = 20;
  const minLap = lapHistory[0].lap;
  const maxLap = Math.max(lapHistory[lapHistory.length - 1].lap, minLap + 1);

  const xScale = (lap) =>
    snapHalf(PAD.left + ((lap - minLap) / (maxLap - minLap)) * innerW);

  const SAFE_Y = 14;

  const yScale = (pos) =>
    snapHalf(
      PAD.top +
      SAFE_Y +
      ((pos - 1) / (maxPos - 1)) * (innerH - 2 * SAFE_Y)
    );

  const yTicks = [1, 5, 10, 15, 20];

  const lapSpan = maxLap - minLap;
  const xTickStep = lapSpan <= 20 ? 5 : lapSpan <= 40 ? 10 : 15;
  const xTicks = [];
  for (let l = minLap; l <= maxLap; l += xTickStep) xTicks.push(l);
  if (!xTicks.includes(maxLap)) xTicks.push(maxLap);

  const endX = xScale(maxLap);

  const lines = driverCodes.map((code, i) => {
    const pts = lapHistory
      .map((frame) => {
        const d = frame.drivers.find((x) => x.code === code);
        if (!d) return null;
        const x = xScale(frame.lap);
        const y = yScale(d.position);
        return `${x},${y}`;
      })
      .filter(Boolean)
      .join(" ");

    const lastFrame = [...lapHistory]
      .reverse()
      .find((f) => f.drivers.find((x) => x.code === code));

    const lastDriver = lastFrame?.drivers.find((x) => x.code === code);

    const team = driverMeta[code]?.team;
    const color = getDriverColor(team);

    const isHighlighted =
      Array.isArray(highlightedDrivers) && highlightedDrivers.length > 0
        ? highlightedDrivers.includes(code)
        : true;

    const endY = lastDriver ? yScale(lastDriver.position) : null;

    return {
      code,
      pts,
      color,
      lastDriver,
      isHighlighted,
      endY,
    };
  });

  const orderedLines = [
    ...lines.filter((l) => !l.isHighlighted),
    ...lines.filter((l) => l.isHighlighted),
  ];

  const rawLabels = lines
    .filter((l) => l.lastDriver && typeof l.endY === "number")
    .map((l) => ({
      code: l.code,
      color: l.color,
      isHighlighted: l.isHighlighted,
      y: l.endY,
    }))
    .sort((a, b) => a.y - b.y);

  const count = rawLabels.length || 1;
  const available = innerH;
  const targetGap = 6;

  let pillH = 26;
  let gap = targetGap;

  const needed = count * pillH + (count - 1) * gap;
  if (needed > available) {
    const scale = available / needed;
    pillH = clamp(Math.floor(pillH * scale), 16, 26);
    gap = clamp(Math.floor(gap * scale), 2, targetGap);
  }

  const fontSize = pillH <= 18 ? 11 : pillH <= 22 ? 12 : 16;

  const alignedLabels = rawLabels.map((l) => ({
    ...l,
    y: l.y,
  }));

  const pillX = W - PAD.right + pillGutter;

  return (
    <div className="w-full overflow-x-auto">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        style={{ minWidth: 560, maxHeight: 520 }}
        aria-label="Driver position history chart"
        shapeRendering="geometricPrecision"
      >
        {yTicks.map((pos) => (
          <line
            key={`grid-${pos}`}
            x1={PAD.left}
            y1={yScale(pos)}
            x2={W - PAD.right}
            y2={yScale(pos)}
            stroke="currentColor"
            strokeWidth={0.8}
            className="text-neutral-200 dark:text-white/10"
            strokeDasharray="4 7"
          />
        ))}

        {yTicks.map((pos) => (
          <text
            key={`y-${pos}`}
            x={PAD.left - 10}
            y={yScale(pos) + 4}
            textAnchor="end"
            fontSize={10}
            className="fill-neutral-500 dark:fill-neutral-500 font-mono"
          >
            P{pos}
          </text>
        ))}

        {xTicks.map((lap) => (
          <text
            key={`x-${lap}`}
            x={xScale(lap)}
            y={H - 10}
            textAnchor="middle"
            fontSize={10}
            className="fill-neutral-500 dark:fill-neutral-500 font-mono"
          >
            {lap}
          </text>
        ))}

        <text
          x={PAD.left + innerW / 2}
          y={H - 2}
          textAnchor="middle"
          fontSize={9}
          className="fill-neutral-400 dark:fill-neutral-600"
        >
          LAPS
        </text>

        {orderedLines.map((l) => (
          <g key={`line-${l.code}`}>
            <polyline
              points={l.pts}
              fill="none"
              stroke={l.color}
              strokeWidth={l.isHighlighted ? 2.6 : 1.35}
              strokeOpacity={l.isHighlighted ? 0.92 : 0.28}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </g>
        ))}

        {lines
          .filter((l) => l.lastDriver && typeof l.endY === "number")
          .map((l) => (
            <circle
              key={`dot-${l.code}`}
              cx={endX}
              cy={l.endY}
              r={3.1}
              fill={l.color}
              opacity={l.isHighlighted ? 0.9 : 0.28}
            />
          ))}

        {alignedLabels.map((lab) => {
          const muted = !lab.isHighlighted;

          return (
            <g key={`pill-${lab.code}`}>
              <foreignObject
                x={pillX}
                y={lab.y - pillH / 2}
                width={pillW}
                height={pillH}
              >
                <div
                  xmlns="http://www.w3.org/1999/xhtml"
                  className={[
                    "pointer-events-none select-none",
                    "inline-flex items-center gap-2",
                    "w-full justify-start",
                    "rounded-full px-2.5",
                    "ring-1 ring-black/5 dark:ring-white/10",
                    muted
                      ? "bg-white/70 opacity-70 dark:bg-neutral-950/40"
                      : "bg-white/90 dark:bg-neutral-950/55",
                  ].join(" ")}
                  style={{ height: `${pillH}px` }}
                >
                  <span
                    className="rounded-full"
                    style={{
                      height: `${Math.max(10, pillH - 12)}px`,
                      width: "3px",
                      backgroundColor: lab.color,
                    }}
                  />
                  <span
                    className={[
                      "font-medium tracking-tight",
                      muted
  ? "text-neutral-500 dark:text-neutral-400"
  : "text-neutral-700 dark:text-neutral-200"
                    ].join(" ")}
                    style={{ fontSize: `${fontSize}px`, lineHeight: "1" }}
                  >
                    {lab.code}
                  </span>
                </div>
              </foreignObject>
            </g>
          );
        })}
      </svg>
    </div>
  );
}