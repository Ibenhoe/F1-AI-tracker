export default function TireBadge({ compound }) {
  const map = {
    SOFT: {
      ring: "border-red-500",
      text: "text-red-400",
      bg: "bg-red-500/10",
    },
    MEDIUM: {
      ring: "border-yellow-400",
      text: "text-yellow-300",
      bg: "bg-yellow-400/10",
    },
    HARD: {
      ring: "border-neutral-300",
      text: "text-neutral-200",
      bg: "bg-neutral-300/10",
    },
    INTERMEDIATE: {
      ring: "border-green-400",
      text: "text-green-400",
      bg: "bg-green-400/10",
    },
    WET: {
      ring: "border-blue-400",
      text: "text-blue-400",
      bg: "bg-blue-400/10",
    },
  };

  const style = map[compound] ?? map.HARD;

  return (
    <div className="flex items-center gap-2">
      {/* Tire icon */}
      <div
        className={[
          "h-6 w-6 rounded-full border-2",
          "bg-neutral-900",
          style.ring,
          "shadow-inner",
        ].join(" ")}
      />

      {/* Label */}
      <span
        className={[
          "text-xs font-semibold tracking-wide",
          style.text,
        ].join(" ")}
      >
        {compound}
      </span>
    </div>
  );
}
