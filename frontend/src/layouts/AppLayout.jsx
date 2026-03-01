import { useEffect, useState } from "react";
import { Outlet } from "react-router-dom";
import { Moon, Sun, PanelLeftClose, PanelLeftOpen } from "lucide-react";

import Sidebar from "../components/Sidebar.jsx";
import Button from "../components/ui/Button.jsx";
import AccentSelect from "../components/ui/AccentSelect.jsx";
import { useTheme } from "../providers/ThemeProvider.jsx";

const STORAGE_KEY = "ui:sidebar-collapsed";

export default function AppLayout() {
  const { theme, toggle, accent, setAccent, accents } = useTheme();

  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw === "1") setCollapsed(true);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, collapsed ? "1" : "0");
    } catch {
      // ignore
    }
  }, [collapsed]);

  return (
    // Split scrolling: layout is fixed-height, only main content scrolls
    <div className="h-screen overflow-hidden bg-[rgb(var(--bg))] text-[rgb(var(--fg))]">
      <div className="flex h-full">
        {/* Sidebar (fixed) */}
        <aside
          className={[
            "h-full shrink-0",
            collapsed ? "w-20" : "w-72",
            "transition-[width] duration-200 ease-out",

            // surface
            "bg-white/70 backdrop-blur-sm dark:backdrop-blur-none",
            "dark:bg-[rgb(var(--panel))]",
            "ring-1 ring-inset ring-neutral-200/70 dark:ring-white/10",

            // no sidebar scrollbar; internal areas handle overflow
            "overflow-hidden flex flex-col",
          ].join(" ")}
        >
          <Sidebar collapsed={collapsed} />
        </aside>

        {/* Main column */}
        <div className="flex min-w-0 flex-1 min-h-0 flex-col">
          {/* Top bar */}
          <header className="sticky top-0 z-10">
            <div className="bg-white/60 backdrop-blur-md dark:bg-[#141416]/95 dark:backdrop-blur-md">
              <div className="px-6 py-4">
                <div className="grid grid-cols-[auto,1fr,auto] items-center gap-3">
                  {/* Left: sidebar toggle (fixed position) */}
                  <div className="justify-self-start">
                    <Button
                      variant="ghost"
                      onClick={() => setCollapsed((v) => !v)}
                      aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                      title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                      className="px-3"
                    >
                      {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
                      <span className="text-sm">{collapsed ? "Expand" : "Collapse"}</span>
                    </Button>
                  </div>

                  {/* Middle: empty (reserved space keeps left fixed) */}
                  <div />

                  {/* Right: theme + accent (stays right aligned) */}
                  <div className="justify-self-end flex items-center gap-3">
                    <Button
                      variant="ghost"
                      onClick={toggle}
                      aria-label="Toggle theme"
                      title={`Theme: ${theme}`}
                      className="px-3"
                    >
                      <Sun size={16} className="dark:hidden" />
                      <Moon size={16} className="hidden dark:block" />
                      <span className="text-sm">Theme</span>
                    </Button>

                    <AccentSelect value={accent} options={accents} onChange={setAccent} />
                  </div>
                </div>
              </div>

              {/* Hairline separator */}
              <div className="h-px w-full bg-neutral-200/60 dark:bg-white/10" />
            </div>
          </header>

          {/* Main scroll area */}
          <main className="flex-1 min-h-0 overflow-y-auto">
            {/* IMPORTANT: max-w stays; you can remove max-w-7xl if you truly want full-width content */}
            <div className="mx-auto max-w-7xl px-6 py-8">
              <Outlet />
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}