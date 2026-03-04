// src/components/racereplay/utils/predictions.js

// ─── Win-probability model (position + gap + tire age) ───────────────────────
const POSITION_WEIGHTS = [
  100, 38, 18, 10, 6, 4, 2.5, 1.8, 1.2, 0.9,
  0.7, 0.55, 0.45, 0.35, 0.28, 0.22, 0.16, 0.12, 0.08, 0.05,
];

export function computePredictions(currentFrame) {
  if (!currentFrame?.drivers) return [];

  const entries = Object.entries(currentFrame.drivers).filter(([, d]) => d.status !== "OUT");
  if (entries.length === 0) return [];

  const weighted = entries.map(([code, d]) => {
    const pos = Math.max(1, Math.round(d.position || 20));
    const base = POSITION_WEIGHTS[Math.min(pos - 1, POSITION_WEIGHTS.length - 1)];
    const gap = parseFloat((d.gap || "+0").replace("+", "")) || 0;
    const gapF = gap < 5 ? 1.0 : gap < 20 ? 0.88 : gap < 45 ? 0.65 : 0.35;
    const tireAge = d.tire_age || 0;
    const tireF = tireAge < 10 ? 1.05 : tireAge < 25 ? 1.0 : tireAge < 40 ? 0.92 : 0.82;
    return { code, d, weight: base * gapF * tireF };
  });

  const total = weighted.reduce((s, w) => s + w.weight, 0) || 1;

  return weighted
    .map(({ code, d, weight }) => ({
      code,
      name: d.driver_name || code,
      team: d.team || "",
      position: Math.round(d.position || 20),
      probability: Math.round((weight / total) * 100),
      tireCompound: d.tire_compound || "MEDIUM",
      tireAge: d.tire_age || 0,
      gap: d.gap || "+0.000",
    }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 10);
}