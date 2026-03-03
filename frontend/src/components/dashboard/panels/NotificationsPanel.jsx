// src/components/dashboard/panels/NotificationsPanel.jsx
import { AlertTriangle, CheckCircle2, Info, XCircle } from "lucide-react";
import Card from "../../ui/Card";

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
        rail: "rgba(0,0,0,0.14)",
        railDark: "rgba(255,255,255,0.14)",
      };
  }
}

export default function NotificationsPanel({ notifications }) {
  const list = Array.isArray(notifications) ? notifications : [];

  if (list.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
          No events yet
        </div>
      </div>
    );
  }

  return (
    <div className="h-full min-h-0 overflow-auto">
      <Card className="divide-y divide-black/5 dark:divide-white/10" clip bordered>
        {list.map((n, idx) => {
          const m = meta(n?.type, n?.color_code);
          const Icon = m.Icon;

          // Prefer stable id; fallback to content-derived key
          const key =
            n?.id ??
            `${n?.lapNumber ?? "x"}-${String(n?.message ?? "msg")}-${idx}`;

          const isDefaultRail = m.rail === "rgba(0,0,0,0.14)";
          const railStyle = isDefaultRail
            ? undefined
            : { backgroundColor: m.rail };

          return (
            <div
              key={key}
              className="relative px-4 py-3 transition-colors hover:bg-black/[0.02] dark:hover:bg-white/[0.03]"
              style={{ minHeight: 74 }}
            >
              {/* Left rail */}
              {isDefaultRail ? (
                <>
                  <div
                    className="absolute left-px top-0 h-full w-[3px] opacity-70 dark:hidden"
                    style={{ backgroundColor: m.rail }}
                    aria-hidden="true"
                  />
                  <div
                    className="absolute left-px top-0 h-full w-[3px] opacity-70 hidden dark:block"
                    style={{ backgroundColor: m.railDark }}
                    aria-hidden="true"
                  />
                </>
              ) : (
                <div
                  className="absolute left-px top-0 h-full w-[3px] opacity-70"
                  style={railStyle}
                  aria-hidden="true"
                />
              )}

              <div className="flex items-start gap-3">
                <Icon
                  size={18}
                  className={["mt-1 shrink-0", m.icon].join(" ")}
                  aria-hidden="true"
                />

                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium leading-snug text-neutral-900 dark:text-neutral-50">
                    {n?.message ?? "—"}
                  </div>

                  <div className="mt-2 flex items-center gap-2 text-xs text-neutral-500 dark:text-neutral-400">
                    <span className="tabular-nums">{n?.time ?? "—"}</span>

                    {typeof n?.lapNumber === "number" ? (
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
      </Card>
    </div>
  );
}