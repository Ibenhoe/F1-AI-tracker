import { NavLink } from "react-router-dom";
import { BarChart3, LineChart, Play } from "lucide-react";
import { BarChart3, LineChart, BookOpen, FileText } from "lucide-react";

function NavItem({ to, icon: Icon, label }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      className={({ isActive }) =>
        [
          "group relative flex items-center gap-2 rounded-lg px-3 py-2 text-sm",
          "transition-colors",

          // LIGHT base
          "text-neutral-700 hover:bg-neutral-100",
          // DARK base
          "dark:text-neutral-300 dark:hover:bg-white/5",

          // ACTIVE: no fill, just stronger text
          isActive ? "text-neutral-900 dark:text-neutral-50" : "",
        ].join(" ")
      }
    >
      {/* Left accent rail when active */}
      <span
        className={[
          "absolute left-0 top-1/2 h-5 w-[2px] -translate-y-1/2 rounded-full",
          "opacity-0 transition-opacity",
          "group-aria-[current=page]:opacity-100",
        ].join(" ")}
        style={{ backgroundColor: "rgb(var(--accent))" }}
        aria-hidden="true"
      />

      {Icon ? (
        <Icon
          size={16}
          className={[
            "shrink-0 transition-colors",

            // idle icon
            "text-neutral-500 group-hover:text-neutral-700",
            "dark:text-neutral-400 dark:group-hover:text-neutral-200",

            // active icon slightly brighter (no accent)
            "group-aria-[current=page]:text-neutral-800 dark:group-aria-[current=page]:text-neutral-100",
          ].join(" ")}
        />
      ) : null}

      <span className="truncate">{label}</span>
    </NavLink>
  );
}

export default function Sidebar() {
  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="px-4 py-5">
        <div className="text-sm font-semibold tracking-tight text-neutral-900 dark:text-neutral-100">
          F1 AI Tracker
        </div>
        <div className="mt-1 text-xs text-neutral-600 dark:text-neutral-400">
          Analytics & prediction dashboard
        </div>
      </div>

      {/* Nav */}
      <div className="px-3">
        <div className="space-y-5">
          <div className="space-y-2">
            <div className="px-2 text-[11px] font-medium uppercase tracking-wider text-neutral-500">
              General
            </div>
            <div className="space-y-1">
              <NavItem to="/" icon={BarChart3} label="Dashboard" />
            </div>
          </div>

          <div className="space-y-2">
            <div className="px-2 text-[11px] font-medium uppercase tracking-wider text-neutral-500">
              Analysis
            </div>
            <div className="space-y-1">
              <NavItem to="/pre-race" icon={LineChart} label="Pre-race analysis" />
              <NavItem to="/race-replay" icon={Play} label="Race replay" />
              <NavItem to="/wiki" icon={BookOpen} label="Wiki" />
              <NavItem to="/docs" icon={FileText} label="Docs" />
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-auto px-4 py-4 text-xs text-neutral-500">
        <div className="flex items-center justify-between">
          <span>v1.0</span>
          <span className="text-neutral-400 dark:text-neutral-600">Local</span>
        </div>
      </div>
    </div>
  );
}
