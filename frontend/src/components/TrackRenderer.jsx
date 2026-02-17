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

  // Draw track boundaries - with safety checks
  if (trackData.innerBoundary && trackData.innerBoundary.length > 0) {
    ctx.strokeStyle = '#444444';
    ctx.lineWidth = 2;
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
    ctx.strokeStyle = '#444444';
    ctx.lineWidth = 2;
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
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.lineWidth = 1;
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

  // Calculate grid spacing for drivers to avoid overlap
  // Distribute all 18 drivers evenly across the track
  const driverCount = drivers.length;
  const spacingRatio = 0.95 / driverCount;  // Each driver gets ~5.3% of track (95% / 18 drivers)

  // Draw each driver
  drivers.forEach(([code, driver], driverIndex) => {
    // Calculate position on track
    let x, y;
    let usedTelemetry = false;
    
    // PRIORITY 1: Use telemetry x,y coordinates from frame (REAL track position!)
    // Telemetry coordinates from FastF1 are now enabled and cached
    if (driver.x !== undefined && driver.y !== undefined && driver.x !== null && driver.y !== null && (driver.x !== 0 || driver.y !== 0)) {
      x = driver.x * scale + offset.x;
      y = driver.y * scale + offset.y;
      usedTelemetry = true;
    }
    // PRIORITY 2: Fall back to centerline positioning based on race position
    else if (centerline.length > 0 && driver.position !== undefined && driver.position !== null) {
      // NEW: Use actual race position plus spacing to avoid driver overlap
      // Position 1-18 maps to a location along the entire track
      // Each position gets an even slice of the track to prevent clustering
      
      // Base position from race standings (1.0 to 18.0)
      const racePosition = driver.position;
      
      // ===== LAP 1 SPECIAL: Grid Formulation =====
      // On lap 1, position drivers in starting grid formation (2 columns)
      // Starting grid: P1 on left of grid, P2 on right, P3 below P1, P4 below P2, etc.
      if (currentFrame.lap <= 1 && centerline.length > 0) {
        // Use start/first part of track for grid
        const startPoint = centerline[0];
        if (startPoint) {
          // Create 2-wide grid layout (F1 standard)
          const gridRow = Math.floor((racePosition - 1) / 2);  // Which row (0,1,2,...)
          const gridCol = (racePosition - 1) % 2;  // Left (0) or right (1)
          
          const rowSpacing = 80;  // Distance between grid rows (in canvas units)
          const colSpacing = 60;  // Distance between left/right columns
          
          x = startPoint.x * scale + offset.x + (gridCol === 0 ? -colSpacing : colSpacing);
          y = startPoint.y * scale + offset.y + gridRow * rowSpacing;
        }
      } else {
        // Normal lap: distribute along track
        // Map to track location: position 1 -> start, position 18 -> 95% of track
        // This ensures drivers are spread across the entire track perimeter
        const trackPositionRatio = Math.max(0, Math.min(0.95, (racePosition - 1) / 18));
        const trackIndex = Math.round(trackPositionRatio * (centerline.length - 1));
        const trackPoint = centerline[Math.max(0, Math.min(centerline.length - 1, trackIndex))];
        
        if (trackPoint && trackPoint.x !== undefined && trackPoint.y !== undefined) {
          x = trackPoint.x * scale + offset.x;
          y = trackPoint.y * scale + offset.y;
        }
      }
      
      if (typeof x === 'undefined' || typeof y === 'undefined') {
        if (driverIndex < 2) {
          console.log(`  ${code}: FALLBACK to position ${racePosition.toFixed(1)} (no telemetry x,y)`);
        }
      }
    }
    
    // Skip driver if no valid coordinates
    if (typeof x === 'undefined' || typeof y === 'undefined') {
      return;
    }
    
    const isSelected = code === selectedDriver;

    // Driver circle - LARGER size for better visibility
    const radius = isSelected ? 15 : 11;
    const color = getTeamColor(code);

    // Shadow/glow effect
    ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
    ctx.shadowBlur = isSelected ? 20 : 10;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;

    // Draw circle
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();

    // Draw border for selected
    if (isSelected) {
      ctx.strokeStyle = '#ffff00';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Draw position number (rounded to nearest integer)
    ctx.shadowColor = 'transparent';
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 10px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const displayPosition = Math.round(driver.position);
    ctx.fillText(String(displayPosition), x, y);

    // Driver code label
    ctx.fillStyle = '#ffffff';
    ctx.font = '11px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(code, x, y + 22);
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
