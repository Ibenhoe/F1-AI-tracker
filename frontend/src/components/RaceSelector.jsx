import { ChevronDown } from "lucide-react";

const RACES = {
  1: "Bahrain",
  2: "Saudi Arabia",
  3: "Australia",
  4: "Japan",
  5: "China",
  6: "Miami",
  7: "Monaco",
  8: "Canada",
  9: "Spain",
  10: "Austria",
  11: "United Kingdom",
  12: "Hungary",
  13: "Belgium",
  14: "Netherlands",
  15: "Italy",
  16: "Azerbaijan",
  17: "Singapore",
  18: "Austin",
  19: "Mexico",
  20: "Brazil",
  21: "Abu Dhabi",
};

export default function RaceSelector({
  selectedRace,
  onRaceChange,
  onSelectRace,
  disabled,
}) {
  const handleChange = (e) => {
    const value = parseInt(e.target.value, 10);
    if (onSelectRace) onSelectRace(value);
    else if (onRaceChange) onRaceChange(value);
  };

  const value = selectedRace || 21;

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="text-sm font-semibold tracking-tight">Race controls</h3>
        <span className="text-xs text-neutral-500">
          Locked while running
        </span>
      </div>

      <div className="relative">
        <select
          value={value}
          onChange={handleChange}
          disabled={disabled}
          className={[
            "w-full appearance-none rounded-lg border px-3 py-2 pr-9 text-sm",
            "border-neutral-800 bg-neutral-950/40 text-neutral-100",
            "focus:outline-none focus:ring-2 focus:ring-neutral-700",
            disabled ? "opacity-60" : "hover:border-neutral-700",
          ].join(" ")}
        >
          {Object.entries(RACES).map(([num, name]) => (
            <option key={num} value={num}>
              {num}. {name}
            </option>
          ))}
        </select>

        <ChevronDown
          size={16}
          className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-neutral-500"
        />
      </div>
    </div>
  );
}
