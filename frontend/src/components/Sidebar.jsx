import { NavLink } from "react-router-dom";
import { BarChart3, LineChart } from "lucide-react";

function NavItem({ to, icon: Icon, label }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      className={({ isActive }) =>
        [
          "flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors",
          "hover:bg-neutral-900 hover:text-neutral-50",
          isActive
            ? "bg-neutral-900 text-neutral-50"
            : "text-neutral-300",
        ].join(" ")
      }
    >
      {Icon ? <Icon size={16} className="text-neutral-400" /> : null}
      <span className="truncate">{label}</span>
    </NavLink>
  );
}

export default function Sidebar() {
  return (
    <div className="flex h-full flex-col">
      <div className="px-4 py-5">
        <div className="text-sm font-semibold tracking-tight text-neutral-50">
          F1 AI Tracker
        </div>
        <div className="mt-1 text-xs text-neutral-400">
          Analytics & prediction dashboard
        </div>
      </div>

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
            </div>
          </div>
        </div>
      </div>

      <div className="mt-auto px-4 py-4 text-xs text-neutral-500">
        <div className="flex items-center justify-between">
          <span>v1.0</span>
          <span className="text-neutral-600">Local</span>
        </div>
      </div>
    </div>
  );
}
