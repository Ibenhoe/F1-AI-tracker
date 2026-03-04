// src/components/racereplay/utils/formatTime.js
export default function formatTime(seconds) {
  const s = Number(seconds);
  if (!Number.isFinite(s) || s < 0) return "00:00:00.0";

  const hours = Math.floor(s / 3600);
  const minutes = Math.floor((s % 3600) / 60);
  const secs = Math.floor(s % 60);
  const tenths = Math.floor((s - Math.floor(s)) * 10);

  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${tenths}`;
}