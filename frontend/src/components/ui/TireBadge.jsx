function tireMeta(compound) {
  const c = String(compound ?? "").toUpperCase();

  if (c === "SOFT") return { label: "S", color: "#DC2626" };
  if (c === "MEDIUM") return { label: "M", color: "#D97706" };
  if (c === "HARD") return { label: "H", color: "#9CA3AF" };
  if (c === "INTERMEDIATE") return { label: "I", color: "#10B981" };
  if (c === "WET") return { label: "W", color: "#3B82F6" };

  return { label: "?", color: "#9CA3AF" };
}

export default function TireBadge({ compound, size = 26 }) {
  const { label, color } = tireMeta(compound);

  const s = Number(size) || 26;
  const ring = Math.max(2, Math.round(s * 0.1));
  const font = Math.max(10, Math.round(s * 0.42));

  return (
    <span
      className="inline-flex items-center justify-center rounded-full select-none"
      style={{
        width: s,
        height: s,
        border: `${ring}px solid ${color}`,
        color,
        fontSize: font,
        lineHeight: 1,
        fontWeight: 700,
        letterSpacing: "-0.02em",
        background:
          "radial-gradient(circle at 30% 30%, rgba(255,255,255,0.06), rgba(255,255,255,0.00) 60%)",
        boxShadow: "none",
      }}
      title={compound || "Unknown"}
      aria-label={compound || "Unknown tire"}
    >
      {label}
    </span>
  );
}