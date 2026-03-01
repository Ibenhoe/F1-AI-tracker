// src/components/Sidebar.jsx
import { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { BarChart3, LineChart, Play, BookOpen, FileText } from "lucide-react";

function NavItem({ to, icon: Icon, label, collapsed = false }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      title={label}
      aria-label={label}
      className={({ isActive }) =>
        [
          "group flex items-center rounded-2xl px-3 py-2 text-sm",
          collapsed ? "justify-center gap-0" : "gap-3",
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
      {/* Active dot only when expanded (keeps collapsed mode clean) */}
      {!collapsed ? (
        <span
          className={[
            "h-2 w-2 rounded-full",
            "opacity-0 group-aria-[current=page]:opacity-100",
          ].join(" ")}
          style={{ backgroundColor: "rgb(var(--accent))" }}
          aria-hidden="true"
        />
      ) : null}

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

      {!collapsed ? (
        <span className="truncate group-aria-[current=page]:font-medium">
          {label}
        </span>
      ) : null}
    </NavLink>
  );
}

function Section({ title, children, collapsed = false }) {
  return (
    <div className="space-y-1">
      {!collapsed ? (
        <div className="px-3 text-[10px] font-semibold uppercase tracking-widest text-neutral-400 dark:text-neutral-500">
          {title}
        </div>
      ) : null}
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

export default function Sidebar({ collapsed = false }) {
  const { pathname } = useLocation();
  const onDocs = pathname === "/docs";

  // Docs scroll-spy active subsection (driven by Docs page via window event)
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

  return (
    <div
      className={[
        "relative flex h-full flex-col py-5",
        collapsed ? "px-2" : "px-3",
      ].join(" ")}
    >
      {/* Collapsed accent rail/glow (subtle, clean) */}
      {collapsed ? (
        <>
          <div
            className="pointer-events-none absolute inset-y-0 left-0 w-[2px] opacity-80"
            style={{
              background:
                "linear-gradient(to bottom, rgba(var(--accent),0) 0%, rgba(var(--accent),0.9) 18%, rgba(var(--accent),0.9) 82%, rgba(var(--accent),0) 100%)",
            }}
            aria-hidden="true"
          />
          <div
            className="pointer-events-none absolute top-6 left-0 h-10 w-10 -translate-x-1/2 rounded-full blur-2xl opacity-30"
            style={{ backgroundColor: "rgb(var(--accent))" }}
            aria-hidden="true"
          />
        </>
      ) : null}

      {/* Header */}
      <div className={["py-2", collapsed ? "px-1" : "px-2"].join(" ")}>
        {!collapsed ? (
          <>
            <div className="text-base font-semibold tracking-tight text-neutral-900 dark:text-neutral-100">
              F1 AI Tracker
            </div>
            <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
              Analytics & prediction
            </div>
          </>
        ) : (
          // Replace the ugly grey placeholder with a clean accent monogram badge
          <div className="flex items-center justify-center">
            <div
              className={[
                "h-10 w-10 rounded-2xl",
                "ring-1 ring-inset ring-black/5 dark:ring-white/10",
                "bg-white/60 dark:bg-white/10",
                "flex items-center justify-center",
              ].join(" ")}
              style={{
                boxShadow: "0 0 0 1px rgba(0,0,0,0.02), 0 10px 30px rgba(var(--accent),0.12)",
              }}
              title="F1 AI Tracker"
              aria-label="F1 AI Tracker"
            >
              <span
                className="text-[12px] font-semibold tracking-tight"
                style={{ color: "rgb(var(--accent))" }}
              >
                F1
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Nav */}
      <div
        className={[
          "mt-4 space-y-3",
          "flex-1 min-h-0 overflow-y-auto",
          "[scrollbar-width:none]",
          "[-ms-overflow-style:none]",
          "[&::-webkit-scrollbar]:hidden",
        ].join(" ")}
      >
        <Section title="General" collapsed={collapsed}>
          <NavItem to="/" icon={BarChart3} label="Dashboard" collapsed={collapsed} />
        </Section>

        <Section title="Race" collapsed={collapsed}>
          <NavItem to="/pre-race" icon={LineChart} label="Pre-race analysis" collapsed={collapsed} />
          <NavItem to="/race-replay" icon={Play} label="Race replay" collapsed={collapsed} />
        </Section>

        <Section title="Explore" collapsed={collapsed}>
          <NavItem to="/wiki" icon={BookOpen} label="Wiki" collapsed={collapsed} />
        </Section>

        <Section title="Docs" collapsed={collapsed}>
          <NavItem to="/docs" icon={FileText} label="Docs" collapsed={collapsed} />

          {/* Only show subsection list when expanded AND on docs */}
          {!collapsed && onDocs ? (
            <div className="mt-2">
              <div className="mt-2 ml-3 pl-3 border-l border-neutral-200 dark:border-neutral-800 space-y-1">
                {DOCS_SECTIONS.map((s) => {
                  const isActive = activeDocsId === s.id;

                  return (
                    <button
                      key={s.id}
                      onClick={() => {
                        scrollToId(s.id);
                        setActiveDocsId(s.id);
                      }}
                      className={[
                        "w-full text-left rounded-xl px-3 py-1.5",
                        "text-[13px] transition-colors",
                        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))]",

                        // base
                        "text-neutral-700 hover:bg-black/5",
                        "dark:text-neutral-300 dark:hover:bg-white/[0.04]",

                        // active subsection highlight (no dot)
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
          ) : null}
        </Section>
      </div>

      {/* Footer */}
      {/* Expanded: keep footer as-is. Collapsed: completely hidden as requested. */}
      {!collapsed ? (
        <div className="mt-auto px-2 pt-6 text-[11px] text-neutral-400 dark:text-neutral-500">
          <div className="flex items-center justify-between">
            <span>v1.0</span>
            <span>Local</span>
          </div>
        </div>
      ) : null}
    </div>
  );
}