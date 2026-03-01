import { Outlet } from "react-router-dom";
import { Moon, Sun } from "lucide-react";

import Sidebar from "../components/Sidebar.jsx";
import Button from "../components/ui/Button.jsx";
import AccentSelect from "../components/ui/AccentSelect.jsx";
import { useTheme } from "../providers/ThemeProvider.jsx";

export default function AppLayout() {
  const { theme, toggle, accent, setAccent, accents } = useTheme();

  return (
    // Split scrolling: layout is fixed-height, only main content scrolls
    <div className="h-screen overflow-hidden bg-[rgb(var(--bg))] text-[rgb(var(--fg))]">
      <div className="flex h-full">
        {/* Sidebar (fixed) */}
        <aside
          className={[
            "w-72 shrink-0 h-full",
            // soft surface instead of hard border slab
            "bg-white/70 backdrop-blur-sm dark:backdrop-blur-none",
            "dark:bg-[rgb(var(--panel))]",
            // subtle divider (less harsh than border-r)
            "ring-1 ring-inset ring-neutral-200/70 dark:ring-white/10",
            // optional: if sidebar ever becomes taller than viewport, it can scroll internally
            "overflow-hidden flex flex-col",
          ].join(" ")}
        >
          <Sidebar />
        </aside>

        {/* Main column */}
        <div className="flex min-w-0 flex-1 min-h-0 flex-col">
          {/* Top bar */}
          <header className="sticky top-0 z-10">
            <div className="bg-white/60 backdrop-blur-md dark:bg-[#141416]/95 dark:backdrop-blur-md">
              <div className="mx-auto flex max-w-7xl items-center justify-end gap-3 px-6 py-4">
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

              {/* Hairline separator */}
              <div className="h-px w-full bg-neutral-200/60 dark:bg-white/10" />
            </div>
          </header>

          {/* Main scroll area */}
          <main className="flex-1 min-h-0 overflow-y-auto">
            <div className="mx-auto max-w-7xl px-6 py-8">
              <Outlet />
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}