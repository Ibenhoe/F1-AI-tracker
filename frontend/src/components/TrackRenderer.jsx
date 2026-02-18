import React, { useEffect, useRef, forwardRef } from 'react';
import './TrackRenderer.css';

/**
 * TrackRenderer Component
 * Canvas-based track visualization with real-time driver positions,
 * DRS zones, and telemetry overlays.
 */
const TrackRenderer = forwardRef(
  (
    {
      frames,
      frameIndex,
      currentFrame,
      trackData,
      drsZones,
      showDRS,
      selectedDriver,
      onDriverSelect,
      focusMode,
      rotation = 0,
    },
    ref
  ) => {
    const canvasRef = useRef(ref);
    const containerRef = useRef(null);
    const scaleRef = useRef(1);
    const offsetRef = useRef({ x: 0, y: 0 });

    // Initialize canvas and setup
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas || !containerRef.current) return;

      const ctx = canvas.getContext('2d');
      const container = containerRef.current;

      // Set canvas size to match container with padding to avoid cutoff by sidebar
      const resizeCanvas = () => {
        const rect = container.getBoundingClientRect();
        // Use 95% of container to ensure track doesn't get cut off
        const width = Math.max(600, rect.width * 0.95);
        const height = Math.max(400, rect.height * 0.95);
        
        canvas.width = width * window.devicePixelRatio;
        canvas.height = height * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        
        // Reset scale/offset refs on resize
        scaleRef.current = 1;
        offsetRef.current = { x: 0, y: 0 };
      };

      resizeCanvas();
      window.addEventListener('resize', resizeCanvas);

      return () => window.removeEventListener('resize', resizeCanvas);
    }, []);

    // Main draw loop
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas || !currentFrame || !trackData) return;

      const ctx = canvas.getContext('2d');
      const width = canvas.width / window.devicePixelRatio;
      const height = canvas.height / window.devicePixelRatio;

      // Clear canvas
      ctx.fillStyle = '#0a0e27';
      ctx.fillRect(0, 0, width, height);

      // Calculate initial scaling if needed
      if (scaleRef.current === 1 && trackData.bounds) {
        const bounds = trackData.bounds;
        const trackWidth = bounds.maxX - bounds.minX;
        const trackHeight = bounds.maxY - bounds.minY;

        let baseScale;
        
        if (focusMode) {
          // Focus mode: maximize track display (rotation=-90, no extra rotation)
          // Use full available space with small margins
          const scaleX = (width * 0.95) / trackWidth;
          const scaleY = (height * 0.95) / trackHeight;
          baseScale = Math.min(scaleX, scaleY);
        } else {
          // Normal mode: full track view with magnification
          const scaleX = (width * 0.85) / trackWidth;
          const scaleY = (height * 0.85) / trackHeight;
          baseScale = Math.min(scaleX, scaleY) * 1.8;
        }
        
        scaleRef.current = baseScale;

        offsetRef.current = {
          x: width / 2,
          y: height / 2,
        };
      }

      // Apply rotation transformation
      const rotationAngle = -(rotation * Math.PI / 180); // Convert degrees to radians
      const bounds = trackData.bounds;
      const trackCenterX = (bounds.minX + bounds.maxX) / 2;
      const trackCenterY = (bounds.minY + bounds.maxY) / 2;
      
      // In focus mode, show the full track centered
      let focusCenterX = trackCenterX;
      let focusCenterY = trackCenterY;
      
      // In focus mode, always center on full track
      if (focusMode) {
        focusCenterX = trackCenterX;
        focusCenterY = trackCenterY;
      }

      ctx.save();
      ctx.translate(offsetRef.current.x, offsetRef.current.y);
      ctx.rotate(rotationAngle);
      ctx.translate(-focusCenterX * scaleRef.current, -focusCenterY * scaleRef.current);

      // Draw track
      drawTrack(ctx, trackData, scaleRef.current, { x: 0, y: 0 });

      // Draw DRS zones
      if (showDRS && drsZones) {
        drawDRSZones(ctx, drsZones, scaleRef.current, { x: 0, y: 0 });
      }

      // Draw drivers
      drawDrivers(
        ctx,
        currentFrame,
        scaleRef.current,
        { x: 0, y: 0 },
        selectedDriver,
        onDriverSelect,
        trackData
      );

      ctx.restore();

      // Draw telemetry info for selected driver
      if (selectedDriver && currentFrame.drivers[selectedDriver]) {
        drawDriverTelemetry(
          ctx,
          selectedDriver,
          currentFrame.drivers[selectedDriver],
          width,
          height
        );
      }
    }, [currentFrame, trackData, drsZones, showDRS, selectedDriver, focusMode]);

    const handleCanvasClick = (e) => {
      const canvas = canvasRef.current;
      if (!canvas || !currentFrame || !trackData) return;

      const rect = canvas.getBoundingClientRect();
      const canvasX = e.clientX - rect.left;
      const canvasY = e.clientY - rect.top;

      // Account for rotation when calculating click position
      const rotationAngle = -Math.PI / 8; // Match the drawing rotation angle
      const bounds = trackData.bounds;
      const trackCenterX = (bounds.minX + bounds.maxX) / 2;
      const trackCenterY = (bounds.minY + bounds.maxY) / 2;

      // Reverse the transformations
      const relX = canvasX - offsetRef.current.x;
      const relY = canvasY - offsetRef.current.y;
      const rotCos = Math.cos(-rotationAngle);
      const rotSin = Math.sin(-rotationAngle);
      const rotatedX = relX * rotCos - relY * rotSin;
      const rotatedY = relX * rotSin + relY * rotCos;
      
      const x = rotatedX / scaleRef.current + trackCenterX;
      const y = rotatedY / scaleRef.current + trackCenterY;

      // Check if click is on any driver
      for (const [code, driver] of Object.entries(currentFrame.drivers)) {
        const dist = Math.hypot(driver.x - x, driver.y - y);
        if (dist < 15) {
          onDriverSelect(selectedDriver === code ? null : code);
          return;
        }
      }
    };

    return (
      <div ref={containerRef} className="track-renderer">
        <canvas
          ref={canvasRef}
          className="track-canvas"
          onClick={handleCanvasClick}
          style={{ cursor: 'pointer' }}
        />
      </div>
    );
  }
);

TrackRenderer.displayName = 'TrackRenderer';

/**
 * Draw the track (centerline, inner and outer boundaries)
 */
function drawTrack(ctx, trackData, scale, offset) {
  if (!trackData) return;

  // Draw track SURFACE (fill between inner and outer boundaries)
  if (trackData.innerBoundary && trackData.outerBoundary && 
      trackData.innerBoundary.length > 0 && trackData.outerBoundary.length > 0) {
    
    // Create path for track fill
    ctx.fillStyle = 'rgba(80, 80, 100, 0.4)';
    ctx.beginPath();
    
    // Outer boundary clockwise
    const outerFirst = trackData.outerBoundary[0];
    ctx.moveTo(outerFirst.x * scale + offset.x, outerFirst.y * scale + offset.y);
    for (let i = 1; i < trackData.outerBoundary.length; i++) {
      const point = trackData.outerBoundary[i];
      if (point && typeof point.x !== 'undefined' && typeof point.y !== 'undefined') {
        ctx.lineTo(point.x * scale + offset.x, point.y * scale + offset.y);
      }
    }
    
    // Inner boundary counter-clockwise (reverse)
    for (let i = trackData.innerBoundary.length - 1; i >= 0; i--) {
      const point = trackData.innerBoundary[i];
      if (point && typeof point.x !== 'undefined' && typeof point.y !== 'undefined') {
        ctx.lineTo(point.x * scale + offset.x, point.y * scale + offset.y);
      }
    }
    
    ctx.closePath();
    ctx.fill();
  }

  // Draw track boundaries - THICKER for better visibility
  if (trackData.innerBoundary && trackData.innerBoundary.length > 0) {
    ctx.strokeStyle = '#666666';
    ctx.lineWidth = 3;
    ctx.beginPath();

    const first = trackData.innerBoundary[0];
    if (first && typeof first.x !== 'undefined' && typeof first.y !== 'undefined') {
      ctx.moveTo(first.x * scale + offset.x, first.y * scale + offset.y);

      for (let i = 1; i < trackData.innerBoundary.length; i++) {
        const point = trackData.innerBoundary[i];
        if (point && typeof point.x !== 'undefined' && typeof point.y !== 'undefined') {
          ctx.lineTo(point.x * scale + offset.x, point.y * scale + offset.y);
        }
      }
      ctx.stroke();
    }
  }

  if (trackData.outerBoundary && trackData.outerBoundary.length > 0) {
    ctx.strokeStyle = '#666666';
    ctx.lineWidth = 3;
    ctx.beginPath();

    const first = trackData.outerBoundary[0];
    if (first && typeof first.x !== 'undefined' && typeof first.y !== 'undefined') {
      ctx.moveTo(first.x * scale + offset.x, first.y * scale + offset.y);

      for (let i = 1; i < trackData.outerBoundary.length; i++) {
        const point = trackData.outerBoundary[i];
        if (point && typeof point.x !== 'undefined' && typeof point.y !== 'undefined') {
          ctx.lineTo(point.x * scale + offset.x, point.y * scale + offset.y);
        }
      }
      ctx.stroke();
    }
    ctx.stroke();
  }

  // Draw centerline
  if (trackData.centerline) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();

    const first = trackData.centerline[0];
    ctx.moveTo(first.x * scale + offset.x, first.y * scale + offset.y);

    for (let i = 1; i < trackData.centerline.length; i++) {
      const point = trackData.centerline[i];
      ctx.lineTo(point.x * scale + offset.x, point.y * scale + offset.y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Draw finish line
  if (trackData.finishLine) {
    const start = trackData.finishLine.start;
    const end = trackData.finishLine.end;

    // Checkered pattern
    const checkSize = 10;
    const pattern = ctx.createPattern(
      createCheckerPattern(checkSize),
      'repeat'
    );
    ctx.fillStyle = pattern;

    const x1 = start.x * scale + offset.x;
    const y1 = start.y * scale + offset.y;
    const x2 = end.x * scale + offset.x;
    const y2 = end.y * scale + offset.y;

    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    const nx = -dy / len;
    const ny = dx / len;

    const width = 30;

    ctx.beginPath();
    ctx.moveTo(x1 + nx * width, y1 + ny * width);
    ctx.lineTo(x2 + nx * width, y2 + ny * width);
    ctx.lineTo(x2 - nx * width, y2 - ny * width);
    ctx.lineTo(x1 - nx * width, y1 - ny * width);
    ctx.closePath();
    ctx.fill();
  }
}

/**
 * Draw DRS zones on the track
 */
function drawDRSZones(ctx, drsZones, scale, offset) {
  if (!drsZones || drsZones.length === 0) return;

  ctx.fillStyle = 'rgba(76, 175, 80, 0.2)';
  ctx.strokeStyle = 'rgba(76, 175, 80, 0.8)';
  ctx.lineWidth = 2;

  drsZones.forEach((zone) => {
    if (!zone.points || zone.points.length < 2) return;

    ctx.beginPath();
    const first = zone.points[0];
    ctx.moveTo(first.x * scale + offset.x, first.y * scale + offset.y);

    for (let i = 1; i < zone.points.length; i++) {
      const point = zone.points[i];
      ctx.lineTo(point.x * scale + offset.x, point.y * scale + offset.y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  });
}

/**
 * Draw drivers on the track with colors and labels
 */
function drawDrivers(
  ctx,
  currentFrame,
  scale,
  offset,
  selectedDriver,
  onDriverSelect,
  trackData
) {
  const drivers = Object.entries(currentFrame.drivers || {});
  
  // Get track centerline for fallback positioning
  const centerline = trackData?.centerline || [];

  // DEBUG: Log driver count and track data on first render of this frame
  if (drivers.length > 0 && drivers.length <= 20) {
    const telemetryAvailable = drivers.filter(([_, d]) => d.x !== null && d.y !== null).length;
    console.log(`[DRAW] Frame: Lap ${currentFrame.lap}, Drivers: ${drivers.length}, Telemetry: ${telemetryAvailable}/${drivers.length}, Centerline: ${centerline.length}`);
  }

  // Sort drivers by position ASCENDING (P1 last, P18 first)
  // This ensures P1 is drawn last and appears on top of other drivers
  const sortedDrivers = drivers.sort((a, b) => {
    const posA = a[1].position || 999;
    const posB = b[1].position || 999;
    return posB - posA;  // Reverse: P18 first, P1 last (drawn on top)
  });
  
  // DEBUG: Log driver order to catch ranking bugs
  if (sortedDrivers.length > 0) {
    const topThree = sortedDrivers.slice(-3).reverse().map(([code, d]) => {
      const pos = Math.round(d.position);
      return `${code}(P${pos})`;
    }).join(' ← ');
    if (JSON.stringify(topThree) !== window._lastDriverOrder) {
      console.log(`[DRIVER ORDER] Lap ${currentFrame.lap}: ${topThree}`);
      window._lastDriverOrder = JSON.stringify(topThree);
    }
  }

  // Calculate grid spacing for drivers to avoid overlap
  // Distribute all 18 drivers evenly across the track
  const driverCount = sortedDrivers.length;
  const spacingRatio = 0.95 / driverCount;  // Each driver gets ~5.3% of track (95% / 18 drivers)

  // Draw each driver (sorted by position, so P1 is drawn LAST and appears ON TOP)
  // First pass: collect all drivers with valid positions
  const driverPositions = new Map();
  
  // Calculate positions
  sortedDrivers.forEach(([code, driver], driverIndex) => {
    let x, y;
    
    // PRIORITY 1: Use telemetry x,y coordinates from frame (REAL track position)
    if (driver.x !== undefined && driver.y !== undefined && driver.x !== null && driver.y !== null && (driver.x !== 0 || driver.y !== 0)) {
      x = driver.x * scale + offset.x;
      y = driver.y * scale + offset.y;
    }
    // PRIORITY 2: Fall back to centerline positioning (for missing telemetry)
    else if (centerline.length > 0 && driver.position !== undefined && driver.position !== null) {
      // Use position-based centerline positioning
      const trackPositionRatio = Math.max(0.05, Math.min(0.95, (driver.position - 1) / 18));
      const trackIndex = Math.round(trackPositionRatio * (centerline.length - 1));
      const trackPoint = centerline[Math.max(0, Math.min(centerline.length - 1, trackIndex))];
      
      if (trackPoint && trackPoint.x !== undefined && trackPoint.y !== undefined) {
        x = trackPoint.x * scale + offset.x;
        y = trackPoint.y * scale + offset.y;
      }
    }
    
    // Store position for rendering
    if (typeof x !== 'undefined' && typeof y !== 'undefined') {
      driverPositions.set(code, { x, y });
    }
  });

  // Create a map to track actual driver positions on the track
  // Calculate cumulative gap distances for proper spacing
  let cumulativeDistance = 0;
  const driverTrackPositions = new Map();
  const leaderCode = sortedDrivers.length > 0 ? sortedDrivers[0][0] : null;
  const leaderPos = leaderCode ? driverPositions.get(leaderCode) : null;
  
  sortedDrivers.forEach(([code, driver], index) => {
    if (index === 0) {
      // P1 gets the normal position from telemetry
      driverTrackPositions.set(code, { cumDistance: 0, isLeader: true });
    } else {
      // Other drivers are positioned behind with gap-based distance
      const gapStr = driver.gap || '+0.000';
      const gapSeconds = parseFloat(gapStr.replace('+', '')) || 0;
      cumulativeDistance += gapSeconds;
      driverTrackPositions.set(code, { cumDistance: cumulativeDistance, isLeader: false });
    }
  });

  // Render each driver (in sorted order so P1 appears on top)
  sortedDrivers.forEach(([code, driver]) => {
    const pos = driverPositions.get(code);
    if (!pos) return;  // Skip if no valid position calculated
    
    let x = pos.x;
    let y = pos.y;
    
    const isSelected = code === selectedDriver;
    const trackPos = driverTrackPositions.get(code);
    
    // Apply gap-based positioning along the centerline
    if (trackPos && !trackPos.isLeader && leaderPos && centerline.length > 0) {
      // Convert cumulative gap distance to pixels (1 second = 35 pixels)
      const gapPixels = trackPos.cumDistance * 35;
      
      // Find the centerline point closest to P1
      let closestIndex = 0;
      let minDist = Infinity;
      for (let i = 0; i < centerline.length; i++) {
        const point = centerline[i];
        const dist = Math.hypot(
          point.x * scale + offset.x - leaderPos.x,
          point.y * scale + offset.y - leaderPos.y
        );
        if (dist < minDist) {
          minDist = dist;
          closestIndex = i;
        }
      }
      
      // Move this driver forwards along centerline by the gap distance
      let currentDistance = 0;
      let targetIndex = closestIndex;
      
      for (let i = closestIndex; i < centerline.length && currentDistance < gapPixels; i++) {
        const p1 = centerline[i];
        const p2 = centerline[Math.min(centerline.length - 1, i + 1)];
        
        if (p1 && p2) {
          const segmentDist = Math.hypot(
            (p2.x - p1.x) * scale,
            (p2.y - p1.y) * scale
          );
          
          if (currentDistance + segmentDist >= gapPixels) {
            // Interpolate within this segment
            const ratio = (gapPixels - currentDistance) / segmentDist;
            const interpX = p1.x * scale + offset.x + (p2.x - p1.x) * scale * ratio;
            const interpY = p1.y * scale + offset.y + (p2.y - p1.y) * scale * ratio;
            x = interpX;
            y = interpY;
            break;
          }
          
          currentDistance += segmentDist;
          targetIndex = Math.min(centerline.length - 1, i + 1);
        }
      }
      
      // Fallback to segment endpoint if we didn't interpolate
      if (targetIndex >= 0 && targetIndex < centerline.length) {
        const point = centerline[targetIndex];
        if (point) {
          x = point.x * scale + offset.x;
          y = point.y * scale + offset.y;
        }
      }
    }

    // Driver circle - consistent size
    const radius = isSelected ? 20 : 16;
    const color = getTeamColor(code);

    // Shadow/glow effect - Selected drivers get blue glow
    if (isSelected) {
      ctx.shadowColor = 'rgba(0, 150, 255, 0.9)';
      ctx.shadowBlur = 25;
    } else {
      ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
      ctx.shadowBlur = 12;
    }
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;

    // Draw circle
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    // Draw border
    if (isSelected) {
      ctx.strokeStyle = '#00d4ff';
      ctx.lineWidth = 3;
      ctx.stroke();
    } else {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Draw position number
    ctx.shadowColor = 'transparent';
    ctx.fillStyle = '#ffffff';
    const fontSize = isSelected ? 14 : 12;
    ctx.font = `bold ${fontSize}px Arial`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const displayPosition = Math.round(driver.position);
    ctx.fillText(String(displayPosition), x, y);

    // Driver code label with background
    const labelY = y + 26;
    ctx.font = 'bold 11px Arial';
    const codeTextWidth = ctx.measureText(code).width;
    const labelPadding = 5;
    
    // Background for label
    if (isSelected) {
      ctx.fillStyle = 'rgba(0, 100, 200, 0.8)';
    } else {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    }
    ctx.fillRect(
      x - codeTextWidth / 2 - labelPadding,
      labelY - 8,
      codeTextWidth + labelPadding * 2,
      16
    );

    // Driver code text
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(code, x, labelY);
  });
}

/**
 * Draw detailed telemetry for selected driver
 */
function drawDriverTelemetry(
  ctx,
  code,
  driver,
  width,
  height
) {
  const panelWidth = 280;
  const panelHeight = 200;
  const padding = 15;
  const x = width - panelWidth - 20;
  const y = 20;

  // Background panel
  ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
  ctx.fillRect(x, y, panelWidth, panelHeight);

  // Border
  ctx.strokeStyle = getTeamColor(code);
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, panelWidth, panelHeight);

  // Header
  ctx.fillStyle = getTeamColor(code);
  ctx.fillRect(x, y, panelWidth, 30);

  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 14px Arial';
  ctx.textAlign = 'left';
  ctx.fillText(code, x + padding, y + 20);

  // Telemetry data
  let lineY = y + 50;
  const lineHeight = 20;

  const telemetryData = [
    ['Position:', `P${driver.position}`],
    ['Speed:', `${(driver.speed || 0).toFixed(1)} km/h`],
    ['Gear:', String(driver.gear || '-')],
    ['Throttle:', `${(driver.throttle || 0).toFixed(0)}%`],
    ['Brake:', `${(driver.brake || 0).toFixed(0)}%`],
    ['Tire:', driver.tire_compound || '-'],
    ['Tire Age:', `${driver.tire_age || 0} laps`],
    ['DRS:', driver.drs ? '✓ ACTIVE' : '✗ OFF'],
    ['Gap:', driver.gap || '-'],
  ];

  ctx.fillStyle = '#cccccc';
  ctx.font = '11px Courier New';
  ctx.textAlign = 'left';

  telemetryData.forEach(([label, value]) => {
    ctx.fillText(label, x + padding, lineY);
    ctx.fillStyle = '#ffff00';
    ctx.fillText(value, x + panelWidth - padding - 80, lineY);
    ctx.fillStyle = '#cccccc';
    lineY += lineHeight;
  });
}

/**
 * Get team color for driver code
 */
function getTeamColor(code) {
  const teamColors = {
    // Red Bull
    VER: '#0600ef',
    PER: '#0600ef',
    // Mercedes
    HAM: '#00d2be',
    RUS: '#00d2be',
    // Ferrari
    LEC: '#dc0000',
    SAI: '#dc0000',
    // McLaren
    NOR: '#ff8700',
    PIA: '#ff8700',
    // Alpine
    OCO: '#0082fa',
    GAS: '#0082fa',
    // Aston Martin
    ALO: '#006c3c',
    STR: '#006c3c',
    // Haas
    MAG: '#ffffff',
    HUL: '#ffffff',
    // Alfa Romeo / Sauber
    BOT: '#900000',
    ZHO: '#900000',
    // Williams
    ALB: '#005aff',
    SAR: '#005aff',
    // Racing Bulls
    RIC: '#4e4e4e',
    TSU: '#4e4e4e',
  };

  return teamColors[code] || '#999999';
}

/**
 * Create checkered pattern for finish line
 */
function createCheckerPattern(size) {
  const canvas = document.createElement('canvas');
  canvas.width = size * 2;
  canvas.height = size * 2;

  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, size * 2, size * 2);

  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, size, size);
  ctx.fillRect(size, size, size, size);

  return canvas;
}

export default TrackRenderer;
