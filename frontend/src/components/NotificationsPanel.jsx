import { AlertTriangle, CheckCircle2, Info, XCircle } from "lucide-react";

function meta(type, colorCode) {
  const effectiveType = (colorCode || type || "").toString().toLowerCase();

  switch (effectiveType) {
    case "success":
    case "overtake":
      return {
        Icon: CheckCircle2,
        rail: "bg-emerald-400/80",
        icon: "text-emerald-600 dark:text-emerald-300",
        iconBg: "bg-emerald-500/10 dark:bg-white/5",
      };

    case "warning":
    case "battle":
      return {
        Icon: AlertTriangle,
        rail: "bg-amber-400/80",
        icon: "text-amber-600 dark:text-amber-300",
        iconBg: "bg-amber-500/10 dark:bg-white/5",
      };

    case "danger":
    case "error":
      return {
        Icon: XCircle,
        rail: "bg-red-400/80",
        icon: "text-red-600 dark:text-red-300",
        iconBg: "bg-red-500/10 dark:bg-white/5",
      };

    default:
      return {
        Icon: Info,
        rail: "bg-neutral-300/80 dark:bg-white/10",
        icon: "text-neutral-600 dark:text-neutral-400",
        iconBg: "bg-neutral-500/10 dark:bg-white/5",
      };
  }
}

export default function NotificationsPanel({ notifications }) {
  const list = Array.isArray(notifications) ? notifications : [];

  if (list.length === 0) {
    return (
      <div
        className={[
          "flex h-full items-center justify-center rounded-2xl",
          "bg-white ring-1 ring-neutral-200/70",
          "dark:bg-white/5 dark:ring-white/10",
        ].join(" ")}
      >
        <div className="text-sm text-neutral-600 dark:text-neutral-400">
          No events yet
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-auto pr-1 [scrollbar-width:thin]">
      <div className="space-y-2">
        {list.map((n) => {
          const m = meta(n.type, n.color_code);
          const Icon = m.Icon;

          return (
            <div
              key={n.id}
              className={[
                "relative overflow-hidden rounded-2xl",
                // light
                "bg-white ring-1 ring-neutral-200/70",
                "shadow-[0_1px_0_rgba(0,0,0,0.03),0_10px_28px_rgba(0,0,0,0.08)]",
                // dark
                "dark:bg-neutral-950/30 dark:ring-white/10",
                "dark:shadow-[0_1px_0_rgba(255,255,255,0.04),0_12px_34px_rgba(0,0,0,0.45)]",
                // consistent height
                "h-[86px]",
              ].join(" ")}
            >
              {/* left rail */}
              <div className={["absolute left-0 top-0 h-full w-1.5", m.rail].join(" ")} />

              <div className="flex h-full items-center gap-4 px-6">
                <div
                  className={[
                    "grid h-9 w-9 shrink-0 place-items-center rounded-xl",
                    "ring-1 ring-neutral-200/70",
                    "dark:ring-white/10",
                    m.iconBg,
                  ].join(" ")}
                >
                  <Icon size={16} className={m.icon} />
                </div>

                <div className="min-w-0 flex-1">
                  <div className="text-sm font-semibold leading-snug text-neutral-900 dark:text-neutral-50">
                    {n.message}
                  </div>
                  <div className="mt-1 text-xs tabular-nums text-neutral-600 dark:text-neutral-400">
                    {n.time}
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