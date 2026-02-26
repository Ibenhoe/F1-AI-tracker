import { AlertTriangle, CheckCircle2, Info, XCircle } from "lucide-react";

function meta(type, colorCode) {
  const effectiveType = (colorCode || type || "").toString().toLowerCase();

  switch (effectiveType) {
    case "success":
    case "overtake":
      return {
        Icon: CheckCircle2,
        icon: "text-emerald-600 dark:text-emerald-400",
        rail: "#10B981", // emerald-500
      };

    case "warning":
    case "battle":
      return {
        Icon: AlertTriangle,
        icon: "text-amber-600 dark:text-amber-400",
        rail: "#F59E0B", // amber-500
      };

    case "danger":
    case "error":
      return {
        Icon: XCircle,
        icon: "text-red-600 dark:text-red-400",
        rail: "#EF4444", // red-500
      };

    default:
      return {
        Icon: Info,
        icon: "text-neutral-500 dark:text-neutral-400",
        rail: "rgba(0,0,0,0.14)", // subtle in light
        railDark: "rgba(255,255,255,0.14)",
      };
  }
}

export default function NotificationsPanel({ notifications }) {
  const list = Array.isArray(notifications) ? notifications : [];

  if (list.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-sm text-neutral-500 dark:text-neutral-400">
          No events yet
        </div>
      </div>
    );
  }

  return (
    <div className="h-full min-h-0 overflow-auto">
      <div className="divide-y divide-neutral-200/70 rounded-2xl bg-white/70 ring-1 ring-neutral-200/70 backdrop-blur-sm dark:divide-white/10 dark:bg-[rgb(var(--panel))] dark:ring-white/10 dark:backdrop-blur-none">
        {list.map((n) => {
          const m = meta(n.type, n.color_code);
          const Icon = m.Icon;

          // If backend sends a color_code that is a hex, you can optionally use it as rail color.
          // For now we stick to meta rails for consistency.
          const railStyle = {
            backgroundColor: m.rail,
          };

          return (
            <div
              key={n.id}
              className="relative px-4 py-3"
              style={{ minHeight: 74 }} // makes it feel like Battles rows (tweak to 72/76 if you want)
            >
              {/* Left rail (same concept as Battles) */}
              <div
                className="absolute left-0 top-0 h-full w-[3px] opacity-70"
                style={
                  m.rail === "rgba(0,0,0,0.14)"
                    ? { backgroundColor: railStyle.backgroundColor }
                    : railStyle
                }
                aria-hidden="true"
              />
              {/* Dark-mode neutral rail override */}
              {m.rail === "rgba(0,0,0,0.14)" ? (
                <div
                  className="absolute left-0 top-0 hidden h-full w-[3px] opacity-70 dark:block"
                  style={{ backgroundColor: m.railDark }}
                  aria-hidden="true"
                />
              ) : null}

              <div className="flex items-start gap-3">
                <Icon size={18} className={["mt-1 shrink-0", m.icon].join(" ")} />

                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium leading-snug text-neutral-900 dark:text-neutral-50">
                    {n.message}
                  </div>

                  <div className="mt-2 flex items-center gap-2 text-xs text-neutral-500 dark:text-neutral-400">
                    <span className="tabular-nums">{n.time}</span>
                    {typeof n.lapNumber === "number" ? (
                      <>
                        <span className="text-neutral-300 dark:text-white/20">•</span>
                        <span className="tabular-nums">Lap {n.lapNumber}</span>
                      </>
                    ) : null}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}