import { AlertTriangle, CheckCircle2, Info, XCircle } from "lucide-react";
import Badge from "./ui/Badge.jsx";

function meta(type) {
  switch (type) {
    case "success":
      return {
        Icon: CheckCircle2,
        badge: "success",
        border: "border-emerald-200 dark:border-emerald-900/40",
        bg: "bg-emerald-50 dark:bg-emerald-950/30",
        icon: "text-emerald-700 dark:text-emerald-300",
      };
    case "warning":
      return {
        Icon: AlertTriangle,
        badge: "warning",
        border: "border-amber-200 dark:border-amber-900/40",
        bg: "bg-amber-50 dark:bg-amber-950/30",
        icon: "text-amber-800 dark:text-amber-300",
      };
    case "error":
      return {
        Icon: XCircle,
        badge: "danger",
        border: "border-red-200 dark:border-red-900/40",
        bg: "bg-red-50 dark:bg-red-950/30",
        icon: "text-red-700 dark:text-red-300",
      };
    default:
      return {
        Icon: Info,
        badge: "neutral",
        border: "border-neutral-200 dark:border-neutral-800",
        bg: "bg-white dark:bg-neutral-950/40",
        icon: "text-neutral-600 dark:text-neutral-400",
      };
  }
}

export default function NotificationsPanel({ notifications }) {
  const count = notifications?.length ?? 0;

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Notifications</h2>
          <p className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
            Race events and system messages
          </p>
        </div>
        <Badge variant="neutral">{count}</Badge>
      </div>

      {count === 0 ? (
        <div className="rounded-xl border border-neutral-200 bg-neutral-50 px-4 py-10 text-center dark:border-neutral-800 dark:bg-neutral-950/40">
          <div className="text-sm font-medium text-neutral-900 dark:text-neutral-200">
            No notifications yet
          </div>
          <div className="mt-1 text-xs text-neutral-600 dark:text-neutral-500">
            Events will appear here during the race.
          </div>
        </div>
      ) : (
        <div className="max-h-[420px] space-y-2 overflow-auto pr-1">
          {notifications.map((n) => {
            const m = meta(n.type);
            const Icon = m.Icon;

            return (
              <div key={n.id} className={["rounded-xl border px-3 py-3", m.border, m.bg].join(" ")}>
                <div className="flex items-start gap-3">
                  <div className="mt-0.5">
                    <Icon size={16} className={m.icon} />
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="text-sm text-neutral-900 dark:text-neutral-100">
                      {n.message}
                    </div>
                    <div className="mt-1 text-xs text-neutral-600 dark:text-neutral-500">
                      {n.time}
                    </div>
                  </div>

                  <div className="shrink-0">
                    <Badge variant={m.badge}>
                      {String(n.type ?? "info").toUpperCase()}
                    </Badge>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
