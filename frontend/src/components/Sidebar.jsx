// src/components/Sidebar.jsx
import { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
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

      <span className="truncate group-aria-[current=page]:font-medium">
        {label}
      </span>
    </NavLink>
  );
}

function Section({ title, children }) {
  return (
    <div className="space-y-1">
      <div className="px-3 text-[10px] font-semibold uppercase tracking-widest text-neutral-400 dark:text-neutral-500">
        {title}
      </div>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

const DOCS_SECTIONS = [
  { id: "overview", title: "Project Overview" },
  { id: "architecture", title: "System Architecture" },
  { id: "data-layer", title: "Data Layer" },
  { id: "ml-model", title: "Machine Learning Model" },
  { id: "confidence", title: "Confidence Scoring" },
  { id: "race-sim", title: "Race Simulation" },
  { id: "api", title: "Backend API" },
  { id: "socketio", title: "Real-time Communication" },
  { id: "frontend", title: "Frontend Architecture" },
  { id: "tire-strategy", title: "Tire Strategy Model" },
  { id: "battle", title: "Battle Detector" },
  { id: "wiki", title: "Historical Wiki" },
  { id: "setup", title: "Getting Started" },
];

function scrollToId(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

export default function Sidebar() {
  const { pathname } = useLocation();
  const onDocs = pathname === "/docs";

  // Scroll-spy active subsection (driven by Docs page via window event)
  const [activeDocsId, setActiveDocsId] = useState("overview");

  useEffect(() => {
    if (!onDocs) return;

    const handler = (e) => {
      const id = e?.detail?.id;
      if (typeof id === "string" && id.length) setActiveDocsId(id);
    };

    window.addEventListener("docs:active-section", handler);
    return () => window.removeEventListener("docs:active-section", handler);
  }, [onDocs]);

  function scrollToIdAndSetActive(id) {
    scrollToId(id);
    setActiveDocsId(id); // immediate feedback on click
  }

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
      <div
        className={[
          "mt-4 space-y-3",
          "flex-1 min-h-0 overflow-y-auto",
          "[scrollbar-width:none]", // Firefox
          "[-ms-overflow-style:none]", // IE/Edge legacy
          "[&::-webkit-scrollbar]:hidden", // Chrome/Safari
        ].join(" ")}
      >
        <Section title="General">
          <NavItem to="/" icon={BarChart3} label="Dashboard" />
        </Section>

        <Section title="Race">
          <NavItem to="/pre-race" icon={LineChart} label="Pre-race analysis" />
          <NavItem to="/race-replay" icon={Play} label="Race replay" />
        </Section>

        <Section title="Explore">
          <NavItem to="/wiki" icon={BookOpen} label="Wiki" />
        </Section>

        <Section title="Docs">
          <NavItem to="/docs" icon={FileText} label="Docs" />

          {onDocs && (
            <div className="mt-2">
              <div className="mt-2 ml-3 pl-3 border-l border-neutral-200 dark:border-neutral-800 space-y-1">
                {DOCS_SECTIONS.map((s) => {
                  const isActive = activeDocsId === s.id;

                  return (
                    <button
                      key={s.id}
                      onClick={() => scrollToIdAndSetActive(s.id)}
                      className={[
                        "w-full text-left rounded-xl px-3 py-1.5",
                        "text-[13px] transition-colors",
                        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))]",

                        // base
                        "text-neutral-700 hover:bg-black/5",
                        "dark:text-neutral-300 dark:hover:bg-white/[0.04]",

                        // active subsection highlight
                        isActive
                          ? [
                            "bg-black/5 text-neutral-900",
                            "dark:bg-white/[0.06] dark:text-neutral-50",
                            "ring-1 ring-inset ring-black/5 dark:ring-white/10",
                          ].join(" ")
                          : "",
                      ].join(" ")}
                      aria-current={isActive ? "true" : undefined}
                      title={s.title}
                    >
                      <span className="truncate">{s.title}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}
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