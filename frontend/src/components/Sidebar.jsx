import { NavLink } from "react-router-dom";
import { BarChart3, LineChart, Play, BookOpen, FileText } from "lucide-react";

function NavItem({ to, icon: Icon, label }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      className={({ isActive }) =>
        [
          "group relative flex items-center gap-2 rounded-xl px-3 py-2.5 text-sm",
          "transition-colors duration-150",

          // LIGHT base
          "text-neutral-700 hover:bg-neutral-100/70",
          // DARK base
          "dark:text-neutral-300 dark:hover:bg-white/5",

          // Focus ring (keyboard)
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2",
          "focus-visible:ring-neutral-300 focus-visible:ring-offset-white",
          "dark:focus-visible:ring-white/20 dark:focus-visible:ring-offset-neutral-950",

          // ACTIVE: subtle surface highlight + stronger text
          isActive
            ? "bg-neutral-100 text-neutral-900 dark:bg-white/5 dark:text-neutral-50"
            : "",
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
            "dark:text-neutral-400 dark:group-hover:text-neutral-300",

            // active icon slightly brighter (no accent)
            "group-aria-[current=page]:text-neutral-800 dark:group-aria-[current=page]:text-neutral-100",
          ].join(" ")}
        />
      ) : null}

<span className="truncate group-aria-[current=page]:font-medium">{label}</span>
    </NavLink>
  );
}

export default function Sidebar() {
  return (
    <div className="flex h-full flex-col bg-white/70 dark:bg-neutral-950/40 backdrop-blur supports-[backdrop-filter]:bg-white/60 dark:supports-[backdrop-filter]:bg-neutral-950/30 border-r border-neutral-200/70 dark:border-white/10">
      {/* Header */}
      <div className="px-5 py-6">
        <div className="text-base font-semibold tracking-tight text-neutral-900 dark:text-neutral-100">
          F1 AI Tracker
        </div>
        <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
          Analytics & prediction
        </div>
      </div>

      {/* Nav */}
      <div className="px-3">
        <div className="space-y-6">
          <div className="space-y-3">
            <div className="px-3 text-[10px] font-semibold uppercase tracking-widest text-neutral-400 dark:text-neutral-600">
              General
            </div>
            <div className="space-y-1">
              <NavItem to="/" icon={BarChart3} label="Dashboard" />
            </div>
          </div>

          <div className="space-y-3">
            <div className="px-3 text-[10px] font-semibold uppercase tracking-widest text-neutral-400 dark:text-neutral-600">
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
      <div className="mt-auto px-5 py-5 text-[11px] text-neutral-400 dark:text-neutral-600">        <div className="flex items-center justify-between">
        <span>v1.0</span>
        <span className="text-neutral-400 dark:text-neutral-600">Local</span>
      </div>
      </div>
    </div>
  );
}
