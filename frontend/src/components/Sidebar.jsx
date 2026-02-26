import { NavLink } from "react-router-dom";
import { BarChart3, LineChart, Play, BookOpen, FileText } from "lucide-react";

function NavItem({ to, icon: Icon, label }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      className={({ isActive }) =>
        [
          "group flex items-center gap-3 rounded-2xl px-3 py-2 text-sm",
          "transition-colors",

          // base
          "text-neutral-700 hover:bg-black/5",
          "dark:text-neutral-300 dark:hover:bg-white/[0.04]",

          // focus
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))]",

          // active
          isActive
            ? "bg-black/5 text-neutral-900 dark:bg-white/[0.06] dark:text-neutral-50"
            : "",
        ].join(" ")
      }
    >
      {/* Minimal active indicator */}
      <span
        className={[
          "h-2 w-2 rounded-full",
          "opacity-0 group-aria-[current=page]:opacity-100",
        ].join(" ")}
        style={{ backgroundColor: "rgb(var(--accent))" }}
        aria-hidden="true"
      />

      {Icon ? (
        <Icon
          size={16}
          className={[
            "shrink-0",
            "text-neutral-500 group-hover:text-neutral-700",
            "dark:text-neutral-400 dark:group-hover:text-neutral-300",
            "group-aria-[current=page]:text-neutral-800 dark:group-aria-[current=page]:text-neutral-100",
          ].join(" ")}
        />
      ) : null}

      <span className="truncate group-aria-[current=page]:font-medium">{label}</span>
    </NavLink>
  );
}

function Section({ title, children }) {
  return (
    <div className="space-y-2">
      <div className="px-3 text-[10px] font-semibold uppercase tracking-widest text-neutral-400 dark:text-neutral-500">
        {title}
      </div>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

export default function Sidebar() {
  return (
    <div className="flex h-full flex-col px-3 py-5">
      {/* Header */}
      <div className="px-2 py-2">
        <div className="text-base font-semibold tracking-tight text-neutral-900 dark:text-neutral-100">
          F1 AI Tracker
        </div>
        <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
          Analytics & prediction
        </div>
      </div>

      {/* Nav */}
      <div className="mt-6 space-y-6">
        <Section title="General">
          <NavItem to="/" icon={BarChart3} label="Dashboard" />
        </Section>

        <Section title="Analysis">
          <NavItem to="/pre-race" icon={LineChart} label="Pre-race analysis" />
          <NavItem to="/race-replay" icon={Play} label="Race replay" />
          <NavItem to="/wiki" icon={BookOpen} label="Wiki" />
          <NavItem to="/docs" icon={FileText} label="Docs" />
        </Section>
      </div>

      {/* Footer */}
      <div className="mt-auto px-2 pt-6 text-[11px] text-neutral-400 dark:text-neutral-500">
        <div className="flex items-center justify-between">
          <span>v1.0</span>
          <span>Local</span>
        </div>
      </div>
    </div>
  );
}