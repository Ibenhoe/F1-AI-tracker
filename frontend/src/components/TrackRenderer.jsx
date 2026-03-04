import React, { useEffect, useRef, forwardRef, useImperativeHandle } from "react";
import { getTeamColor } from "../utils/teamColors";
import { normalizeDriver } from "../components/racereplay/utils/telemetry";

function isDarkTheme() {
  if (typeof document === "undefined") return true;
  return document.documentElement.classList.contains("dark");
}

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

function withAlpha(color, alpha) {
  const c = String(color || "").trim();
  if (/^#([0-9a-f]{6})$/i.test(c)) {
    const a = Math.round(clamp(alpha, 0, 1) * 255)
      .toString(16)
      .padStart(2, "0");
    return `${c}${a}`;
  }
  const rgb = c.match(/^rgb\(\s*([^)]+)\s*\)$/i);
  if (rgb) return `rgba(${rgb[1]}, ${alpha})`;
  const rgba = c.match(/^rgba\(\s*([^)]+)\s*\)$/i);
  if (rgba) {
    const parts = rgba[1].split(",").map((p) => p.trim());
    return `rgba(${parts.slice(0, 3).join(", ")}, ${alpha})`;
  }
  return c;
}

const DRIVER_TEAMS_FALLBACK = {
  VER: "Red Bull Racing",
  PER: "Red Bull Racing",
  HAM: "Mercedes",
  RUS: "Mercedes",
  LEC: "Ferrari",
  SAI: "Ferrari",
  NOR: "McLaren",
  PIA: "McLaren",
  ALO: "Aston Martin",
  STR: "Aston Martin",
  GAS: "Alpine",
  OCO: "Alpine",
  TSU: "RB",
  RIC: "RB",
  LAW: "RB",
  ALB: "Williams",
  SAR: "Williams",
  COL: "Williams",
  HUL: "Haas F1 Team",
  MAG: "Haas F1 Team",
  BEA: "Haas F1 Team",
  BOT: "Kick Sauber",
  ZHO: "Kick Sauber",
};

function getDriverTeamName(code, driver) {
  const hasConstructorKey = Object.prototype.hasOwnProperty.call(
    driver || {},
    "constructor"
  );
  const raw =
    driver?.team ||
    driver?.team_name ||
    (hasConstructorKey ? driver?.constructor : null) ||
    driver?.constructor_name;

  if (raw && raw !== "Unknown") return raw;
  return DRIVER_TEAMS_FALLBACK[code] || "";
}

function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

const TrackRenderer = forwardRef(
  (
    {
      frames,
      frameIndex,
      isPlaying = false,
      playbackSpeed = 1,
      currentFrame,
      trackData,
      drsZones,
      showDRS,
      selectedDriver,
      onDriverSelect,
      onFrameChange,
      focusMode,
      rotation = 0,
      smoothedGaps = {},
    },
    ref
  ) => {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);

    const scaleRef = useRef(1);
    const offsetRef = useRef({ x: 0, y: 0 });
    const sizeRef = useRef({ w: 1, h: 1, dpr: 1 });

    useImperativeHandle(ref, () => canvasRef.current);

    const frameIdxRef = useRef(frameIndex || 0);
    const lastRafTimeRef = useRef(null);
    const lastNotifiedFrameRef = useRef(-1);

    useEffect(() => {
      frameIdxRef.current = frameIndex || 0;
      lastNotifiedFrameRef.current = Math.floor(frameIndex || 0);
    }, [frameIndex]);

    const drawPropsRef = useRef({});
    drawPropsRef.current = {
      frames,
      trackData,
      drsZones,
      showDRS,
      selectedDriver,
      onDriverSelect,
      focusMode,
      rotation,
      smoothedGaps,
      isPlaying,
      playbackSpeed,
      onFrameChange,
    };

    useEffect(() => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container) return;

      const resizeCanvas = () => {
        const rect = container.getBoundingClientRect();
        const width = Math.max(1, rect.width);
        const height = Math.max(1, rect.height);

        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.floor(width * dpr);
        canvas.height = Math.floor(height * dpr);

        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        sizeRef.current = { w: width, h: height, dpr };
        scaleRef.current = 1;
        offsetRef.current = { x: width / 2, y: height / 2 };
      };

      resizeCanvas();
      window.addEventListener("resize", resizeCanvas);
      return () => window.removeEventListener("resize", resizeCanvas);
    }, []);

    useEffect(() => {
      let rafId;

      const loop = (rafTimestamp) => {
        const canvas = canvasRef.current;
        const props = drawPropsRef.current;

        const {
          frames: fr,
          trackData: td,
          drsZones: dz,
          showDRS: sDRS,
          selectedDriver: sel,
          onDriverSelect: onSel,
          focusMode: fm,
          rotation: rotDeg,
          isPlaying: play,
          playbackSpeed: speed,
          onFrameChange: onFc,
        } = props;

        if (play && fr && fr.length > 0) {
          if (lastRafTimeRef.current !== null) {
            const rawDelta = (rafTimestamp - lastRafTimeRef.current) / 1000;
            const delta = Math.min(rawDelta, 0.05);
            const advance = delta * 120 * speed;
            frameIdxRef.current = frameIdxRef.current + advance;

            const lastIdx = fr.length - 1;
            if (frameIdxRef.current >= fr.length) {
              frameIdxRef.current = lastIdx;
              if (onFc && lastNotifiedFrameRef.current !== lastIdx) {
                lastNotifiedFrameRef.current = lastIdx;
                onFc(lastIdx);
              }
            } else {
              const intIdx = Math.floor(frameIdxRef.current);
              if (onFc && intIdx - lastNotifiedFrameRef.current >= 12) {
                lastNotifiedFrameRef.current = intIdx;
                onFc(intIdx);
              }
            }
          }
          lastRafTimeRef.current = rafTimestamp;
        } else {
          lastRafTimeRef.current = null;
        }

        const idx = Math.min(
          Math.floor(frameIdxRef.current),
          (fr?.length ?? 1) - 1
        );
        const frame = fr?.[idx] ?? null;

        if (canvas && frame && td?.bounds) {
          const ctx = canvas.getContext("2d");
          const { w: width, h: height } = sizeRef.current;

          const dark = isDarkTheme();
          const palette = dark
            ? {
              bg: "rgba(0,0,0,0.92)",
              trackFill: "rgba(255,255,255,0.055)",
              trackStrokeOuter: "rgba(255,255,255,0.22)",
              trackStrokeInner: "rgba(255,255,255,0.18)",
              center: "rgba(255,255,255,0.13)",
              drsFill: "rgba(255,255,255,0.06)",
              drsStroke: "rgba(255,255,255,0.18)",

              markerText: "rgba(255,255,255,0.95)",
              markerShadow: "rgba(0,0,0,0.55)",
              labelFill: "rgba(0,0,0,0.62)",
              labelText: "rgba(255,255,255,0.92)",

              panelBg: "rgba(0,0,0,0.62)",
              panelText: "rgba(255,255,255,0.92)",
              panelMuted: "rgba(255,255,255,0.70)",
            }
            : {
              bg: "rgba(255,255,255,0.96)",
              trackFill: "rgba(0,0,0,0.045)",
              trackStrokeOuter: "rgba(0,0,0,0.20)",
              trackStrokeInner: "rgba(0,0,0,0.16)",
              center: "rgba(0,0,0,0.10)",
              drsFill: "rgba(0,0,0,0.035)",
              drsStroke: "rgba(0,0,0,0.14)",

              markerText: "rgba(255,255,255,0.98)",
              markerShadow: "rgba(0,0,0,0.12)",
              labelFill: "rgba(255,255,255,0.82)",
              labelText: "rgba(0,0,0,0.82)",

              panelBg: "rgba(255,255,255,0.82)",
              panelText: "rgba(0,0,0,0.86)",
              panelMuted: "rgba(0,0,0,0.55)",
            };

          ctx.fillStyle = palette.bg;
          ctx.fillRect(0, 0, width, height);

          const bounds = td.bounds;
          const trackW = Math.max(1e-6, bounds.maxX - bounds.minX);
          const trackH = Math.max(1e-6, bounds.maxY - bounds.minY);

          const rotationAngle = -(Number(rotDeg || 0) * Math.PI) / 180;
          const cos = Math.abs(Math.cos(rotationAngle));
          const sin = Math.abs(Math.sin(rotationAngle));
          const rotatedW = trackW * cos + trackH * sin;
          const rotatedH = trackW * sin + trackH * cos;

          const pad = fm ? 0.06 : 0.09;
          const usableW = width * (1 - pad * 2);
          const usableH = height * (1 - pad * 2);

          const baseScale = Math.min(usableW / rotatedW, usableH / rotatedH);
          scaleRef.current = baseScale;

          offsetRef.current = { x: width / 2, y: height / 2 };

          const centerX = (bounds.minX + bounds.maxX) / 2;
          const centerY = (bounds.minY + bounds.maxY) / 2;

          ctx.save();
          ctx.translate(offsetRef.current.x, offsetRef.current.y);
          ctx.rotate(rotationAngle);
          ctx.scale(scaleRef.current, scaleRef.current);
          ctx.translate(-centerX, -centerY);

          drawTrack(ctx, td, palette);

          if (sDRS && dz) {
            drawDRSZones(ctx, dz, palette);
          }

          drawDrivers(
            ctx,
            frame,
            sel,
            onSel,
            td,
            -rotationAngle,
            palette,
            scaleRef.current
          );

          ctx.restore();

          if (sel && frame.drivers?.[sel]) {
            drawDriverTelemetry(ctx, sel, frame.drivers[sel], width, height, palette);
          }
        }

        rafId = requestAnimationFrame(loop);
      };

      rafId = requestAnimationFrame(loop);
      return () => cancelAnimationFrame(rafId);
    }, []);

    const handleCanvasClick = (e) => {
      const canvas = canvasRef.current;
      const { frames: fr, trackData: td, rotation: rotDeg } = drawPropsRef.current;
      const frame = fr?.[Math.floor(frameIdxRef.current)] ?? null;
      if (!canvas || !frame || !td?.bounds) return;

      const rect = canvas.getBoundingClientRect();
      const canvasX = e.clientX - rect.left;
      const canvasY = e.clientY - rect.top;

      const { w: width, h: height } = sizeRef.current;

      const bounds = td.bounds;
      const centerX = (bounds.minX + bounds.maxX) / 2;
      const centerY = (bounds.minY + bounds.maxY) / 2;

      const rotationAngle = -(Number(rotDeg || 0) * Math.PI) / 180;

      const relX = canvasX - width / 2;
      const relY = canvasY - height / 2;

      const cos = Math.cos(-rotationAngle);
      const sin = Math.sin(-rotationAngle);
      const rx = relX * cos - relY * sin;
      const ry = relX * sin + relY * cos;

      const worldX = rx / scaleRef.current + centerX;
      const worldY = ry / scaleRef.current + centerY;

      const { onDriverSelect: onSel, selectedDriver: sel } = drawPropsRef.current;

      const hitWorld = getHitRadiusWorld(scaleRef.current) * 1.35;

      for (const [code, driver] of Object.entries(frame.drivers || {})) {
        const dx = (driver?.x ?? 0) - worldX;
        const dy = (driver?.y ?? 0) - worldY;
        if (Math.hypot(dx, dy) < hitWorld) {
          onSel?.(sel === code ? null : code);
          return;
        }
      }
    };

    return (
      <div
        ref={containerRef}
        className={[
          "relative h-full w-full overflow-hidden rounded-2xl",
          "bg-white dark:bg-neutral-950/40",
          "ring-1 ring-black/5 dark:ring-white/10",
        ].join(" ")}
      >
        <canvas
          ref={canvasRef}
          className="block h-full w-full"
          onClick={handleCanvasClick}
          style={{ cursor: "pointer" }}
        />
      </div>
    );
  }
);

TrackRenderer.displayName = "TrackRenderer";

function drawTrack(ctx, trackData, palette) {
  if (!trackData) return;

  const inner = trackData.innerBoundary || [];
  const outer = trackData.outerBoundary || [];

  if (inner.length > 2 && outer.length > 2) {
    ctx.fillStyle = palette.trackFill;
    ctx.beginPath();
    ctx.moveTo(outer[0].x, outer[0].y);
    for (let i = 1; i < outer.length; i++) ctx.lineTo(outer[i].x, outer[i].y);
    for (let i = inner.length - 1; i >= 0; i--) ctx.lineTo(inner[i].x, inner[i].y);
    ctx.closePath();
    ctx.fill();
  }

  if (outer.length > 1) {
    ctx.strokeStyle = palette.trackStrokeOuter;
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    ctx.moveTo(outer[0].x, outer[0].y);
    for (let i = 1; i < outer.length; i++) ctx.lineTo(outer[i].x, outer[i].y);
    ctx.stroke();
  }

  if (inner.length > 1) {
    ctx.strokeStyle = palette.trackStrokeInner;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(inner[0].x, inner[0].y);
    for (let i = 1; i < inner.length; i++) ctx.lineTo(inner[i].x, inner[i].y);
    ctx.stroke();
  }

  if (Array.isArray(trackData.centerline) && trackData.centerline.length > 1) {
    const c = trackData.centerline;
    ctx.strokeStyle = palette.center;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([10, 10]);
    ctx.beginPath();
    ctx.moveTo(c[0].x, c[0].y);
    for (let i = 1; i < c.length; i++) ctx.lineTo(c[i].x, c[i].y);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function drawDRSZones(ctx, drsZones, palette) {
  if (!Array.isArray(drsZones) || drsZones.length === 0) return;

  ctx.fillStyle = palette.drsFill;
  ctx.strokeStyle = palette.drsStroke;
  ctx.lineWidth = 1.3;

  for (const zone of drsZones) {
    const pts = zone?.points || [];
    if (pts.length < 2) continue;

    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
}

function getBaseRadiusWorld(scale) {
  const targetPx = 78;
  const r = targetPx / Math.max(1e-6, scale);
  return clamp(r, 54, 210);
}

function getHitRadiusWorld(scale) {
  const targetPx = 102;
  const r = targetPx / Math.max(1e-6, scale);
  return clamp(r, 66, 270);
}

function drawDrivers(
  ctx,
  currentFrame,
  selectedDriver,
  onDriverSelect,
  trackData,
  counterRotation,
  palette,
  scale
) {
  const entries = Object.entries(currentFrame?.drivers || {});
  if (entries.length === 0) return;

  const dark = isDarkTheme();
  const centerline = trackData?.centerline || [];

  const sorted = entries
    .map(([code, d]) => [code, d || {}])
    .sort((a, b) => (b[1].position ?? 999) - (a[1].position ?? 999));

  const positions = new Map();
  for (const [code, driver] of sorted) {
    let x, y;

    if (
      driver.x !== undefined &&
      driver.y !== undefined &&
      driver.x !== null &&
      driver.y !== null &&
      (driver.x !== 0 || driver.y !== 0)
    ) {
      x = driver.x;
      y = driver.y;
    } else if (centerline.length > 0 && driver.position != null) {
      const ratio = Math.max(0.05, Math.min(0.95, (Number(driver.position) - 1) / 18));
      const idx = Math.round(ratio * (centerline.length - 1));
      const pt = centerline[Math.max(0, Math.min(centerline.length - 1, idx))];
      if (pt) {
        x = pt.x;
        y = pt.y;
      }
    }

    if (x !== undefined && y !== undefined) positions.set(code, { x, y });
  }

  const rBase = getBaseRadiusWorld(scale);
  const rSelected = rBase * 1.12;

  for (const [code, driver] of sorted) {
    const p = positions.get(code);
    if (!p) continue;

    const x = p.x;
    const y = p.y;

    const isSelected = code === selectedDriver;
    const pos = Math.round(driver.position ?? 0);

    const teamName = getDriverTeamName(code, driver);
    const teamColor = getTeamColor(teamName) || "rgb(var(--accent))";

    const r = isSelected ? rSelected : rBase;

    const ring = dark ? "rgba(255,255,255,0.92)" : "rgba(0,0,0,0.16)";
    const ringSelected = dark ? "rgba(255,255,255,0.96)" : "rgba(0,0,0,0.22)";

    ctx.save();
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 2;
    ctx.shadowBlur = isSelected ? 16 : 12;
    ctx.shadowColor = palette.markerShadow;

    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = withAlpha(teamColor, dark ? 0.92 : 0.88);
    ctx.fill();

    ctx.shadowColor = "transparent";

    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.strokeStyle = isSelected ? ringSelected : ring;
    ctx.lineWidth = Math.max(1.6, 2.2 / Math.max(1e-6, scale));
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x, y, r - Math.max(1.2, 1.6 / Math.max(1e-6, scale)), 0, Math.PI * 2);
    ctx.strokeStyle = withAlpha("#000000", dark ? 0.10 : 0.06);
    ctx.lineWidth = Math.max(0.9, 1.2 / Math.max(1e-6, scale));
    ctx.stroke();

    ctx.restore();

    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(counterRotation);
    ctx.fillStyle = palette.markerText;

    const fontSize = clamp(r * 8.10, 108, 198); // 3x
    ctx.font = `900 ${fontSize}px system-ui, -apple-system, Segoe UI, Roboto, Arial`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(pos), 0, 0);
    ctx.restore();

    const showLabel = true;
    if (showLabel) {
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(counterRotation);

      const label = code;
      const labelFont = clamp(r * 3.96, 99, 126);
      ctx.font = `800 ${labelFont}px system-ui, -apple-system, Segoe UI, Roboto, Arial`;
      const w = ctx.measureText(label).width;
      const padX = clamp(r * 1.26, 30, 42);
      const boxW = w + padX * 2;
      const boxH = clamp(r * 2.85, 72, 90);
      const bx = -boxW / 2;
      const by = r + clamp(r * 1.35, 42, 66);
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 2;
      ctx.shadowColor = "rgba(0,0,0,0.16)";

      ctx.fillStyle = palette.labelText;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, 0, by + boxH / 2);
      ctx.restore();
    }
  }
}

function drawDriverTelemetry(ctx, code, driver, width, height, palette) {
  const t = normalizeDriver(driver);

  const pad = 14;
  const panelW = 300;
  const panelH = 212;

  const x = width - panelW - 16;
  const y = 16;

  const teamName = getDriverTeamName(code, driver);
  const teamColor = getTeamColor(teamName) || "rgb(var(--accent))";

  ctx.save();
  ctx.shadowOffsetX = 0;
  ctx.shadowOffsetY = 10;
  ctx.shadowColor = "rgba(0,0,0,0.18)";

  roundRect(ctx, x, y, panelW, panelH, 16);
  ctx.fillStyle = palette.panelBg;
  ctx.fill();

  ctx.shadowColor = "transparent";

  ctx.beginPath();
  ctx.moveTo(x + 16, y + 44);
  ctx.lineTo(x + panelW - 16, y + 44);
  ctx.strokeStyle = withAlpha(teamColor, 0.22);
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = withAlpha(teamColor, 0.95);
  ctx.font = "800 14px system-ui, -apple-system, Segoe UI, Roboto, Arial";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(code, x + pad, y + 22);

  const items = [
    ["Position", `P${t.position}`],
    ["Speed", `${t.speedKmh.toFixed(1)} km/h`],
    ["Throttle", `${t.throttlePct.toFixed(0)}%`],
    ["Brake", `${t.brakePct.toFixed(0)}%`],
    ["Tire", String(t.tireCompound)],
    ["Tire age", `${t.tireAge} laps`],
    ["DRS", t.drsOn ? "On" : "Off"],
    ["Gap", String(t.gap)],
  ];

  let lineY = y + 62;
  const lineH = 18;

  ctx.font = "600 12px system-ui, -apple-system, Segoe UI, Roboto, Arial";
  for (const [k, v] of items) {
    ctx.fillStyle = palette.panelMuted;
    ctx.textAlign = "left";
    ctx.fillText(`${k}:`, x + pad, lineY);

    ctx.fillStyle = palette.panelText;
    ctx.textAlign = "right";
    ctx.fillText(v, x + panelW - pad, lineY);

    lineY += lineH;
  }

  ctx.restore();
}

export default TrackRenderer;