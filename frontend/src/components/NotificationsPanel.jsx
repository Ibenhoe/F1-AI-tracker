import { AlertTriangle, CheckCircle2, Info, XCircle } from "lucide-react";
import Badge from "./ui/Badge.jsx";

function meta(type, colorCode) {
  // Use color_code from backend if available, fallback to type
  const effectiveType = colorCode || type;
  
  // Default badge is "neutral"
  let badge = "neutral";
  
  switch (effectiveType) {
    case "success":
      badge = "success";
      return {
        Icon: CheckCircle2,
        badge: badge,
        border: "border-emerald-200 dark:border-emerald-900/40",
        bg: "bg-emerald-50 dark:bg-emerald-950/30",
        icon: "text-emerald-700 dark:text-emerald-300",
      };
    case "warning":
      badge = "warning";
      return {
        Icon: AlertTriangle,
        badge: badge,
        border: "border-amber-200 dark:border-amber-900/40",
        bg: "bg-amber-50 dark:bg-amber-950/30",
        icon: "text-amber-800 dark:text-amber-300",
      };
    case "danger":
      badge = "danger";
      return {
        Icon: XCircle,
        badge: badge,
        border: "border-red-200 dark:border-red-900/40",
        bg: "bg-red-50 dark:bg-red-950/30",
        icon: "text-red-700 dark:text-red-300",
      };
    case "error":
      badge = "danger";
      return {
        Icon: XCircle,
        badge: badge,
        border: "border-red-200 dark:border-red-900/40",
        bg: "bg-red-50 dark:bg-red-950/30",
        icon: "text-red-700 dark:text-red-300",
      };
    case "info":
      badge = "neutral";
      return {
        Icon: Info,
        badge: badge,
        border: "border-blue-200 dark:border-blue-900/40",
        bg: "bg-blue-50 dark:bg-blue-950/30",
        icon: "text-blue-600 dark:text-blue-400",
      };
    default:
      return {
        Icon: Info,
        badge: badge,
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
            const m = meta(n.type, n.color_code);
            const Icon = m.Icon;
            
            // Debug logging
            console.log("[NOTIFICATION] Event:", { type: n.type, color_code: n.color_code, message: n.message });
            
            // Determine badge text based on type and subtype
            let badgeText = "BATTLE";
            if (n.type === "overtake") {
              badgeText = "OVERTAKE";
            } else if (n.type === "battle") {
              badgeText = "BATTLE";
            }
            
            // Build color-based classes
            let cardClasses = "rounded-xl border px-3 py-3 transition-all";
            
            switch (n.color_code) {
              case "success":
                cardClasses += " border-green-300 dark:border-green-800 bg-green-50 dark:bg-green-950/40";
                break;
              case "danger":
                cardClasses += " border-red-300 dark:border-red-800 bg-red-50 dark:bg-red-950/40";
                break;
              case "warning":
                cardClasses += " border-amber-300 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/40";
                break;
              case "info":
                cardClasses += " border-blue-300 dark:border-blue-800 bg-blue-50 dark:bg-blue-950/40";
                break;
              default:
                cardClasses += " border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950/40";
            }

            return (
              <div key={n.id} className={cardClasses}>
                <div className="flex items-start gap-3">
                  <div className="mt-0.5">
                    <Icon size={16} className={m.icon} />
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                      {n.message}
                    </div>
                    <div className="mt-1 text-xs text-neutral-600 dark:text-neutral-500">
                      {n.time}
                    </div>
                  </div>

                  <div className="shrink-0">
                    <Badge variant={m.badge}>
                      {badgeText}
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
