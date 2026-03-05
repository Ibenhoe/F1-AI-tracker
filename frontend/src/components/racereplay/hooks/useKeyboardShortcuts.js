// src/components/racereplay/hooks/useKeyboardShortcuts.js
import { useEffect } from "react";

const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];
const SEEK_FRAMES = 10;

export default function useKeyboardShortcuts({
  raceData,
  currentFrame,
  isPlaying,
  setIsPlaying,
  setFrameIndex,
  showDRS,
  setShowDRS,
  showTelemetry,
  setShowTelemetry,
  focusMode,
  setFocusMode,
  realTimeMode,
  setRealTimeMode,
  setPlaybackSpeed,
  syncStartRef,
}) {
  useEffect(() => {
    const handleKey = (e) => {
      // Don't fire shortcuts when typing in an input / textarea
      const tag = document.activeElement?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") return;

      switch (e.code) {
        case "Space":
          e.preventDefault();
          if (!realTimeMode) setIsPlaying((p) => !p);
          break;

        case "ArrowLeft":
          e.preventDefault();
          setFrameIndex((i) => Math.max(0, i - SEEK_FRAMES));
          break;

        case "ArrowRight":
          e.preventDefault();
          setFrameIndex((i) => {
            const max = raceData ? raceData.frames.length - 1 : 0;
            return Math.min(max, i + SEEK_FRAMES);
          });
          break;

        case "Equal":   // '+' (unshifted = key)
        case "NumpadAdd":
          e.preventDefault();
          setPlaybackSpeed((s) => {
            const idx = SPEEDS.indexOf(s);
            return idx < SPEEDS.length - 1 ? SPEEDS[idx + 1] : s;
          });
          break;

        case "Minus":   // '-'
        case "NumpadSubtract":
          e.preventDefault();
          setPlaybackSpeed((s) => {
            const idx = SPEEDS.indexOf(s);
            return idx > 0 ? SPEEDS[idx - 1] : s;
          });
          break;

        case "KeyD":
          e.preventDefault();
          setShowDRS((v) => !v);
          break;

        case "KeyT":
          e.preventDefault();
          setShowTelemetry((v) => !v);
          break;

        case "KeyF":
          e.preventDefault();
          setFocusMode((v) => !v);
          break;

        case "KeyS":
          e.preventDefault();
          if (!realTimeMode && currentFrame) {
            syncStartRef.current = {
              wallMs: performance.now(),
              raceTime: currentFrame.raceTime || 0,
            };
            setIsPlaying(false);
          } else {
            syncStartRef.current = null;
          }
          setRealTimeMode((v) => !v);
          break;

        case "KeyR":
          e.preventDefault();
          setFrameIndex(0);
          setIsPlaying(false);
          if (realTimeMode) {
            setRealTimeMode(false);
            syncStartRef.current = null;
          }
          break;

        default:
          break;
      }
    };

    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [
    raceData,
    currentFrame,
    isPlaying,
    realTimeMode,
    setIsPlaying,
    setFrameIndex,
    setShowDRS,
    setShowTelemetry,
    setFocusMode,
    setRealTimeMode,
    setPlaybackSpeed,
    syncStartRef,
  ]);
}
