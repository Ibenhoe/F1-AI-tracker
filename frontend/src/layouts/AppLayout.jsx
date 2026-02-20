import { Outlet } from "react-router-dom";
import { Moon, Sun } from "lucide-react";

import Sidebar from "../components/Sidebar.jsx";
import Button from "../components/ui/Button.jsx";
import AccentSelect from "../components/ui/AccentSelect.jsx";
import { useTheme } from "../providers/ThemeProvider.jsx";

export default function AppLayout() {
  const { theme, toggle, accent, setAccent, accents } = useTheme();

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100">
      <div className="flex min-h-screen">
        {/* Sidebar column surface */}
        <aside
          className={[
            "w-72",
            "border-r border-neutral-200/80 bg-white",
            "dark:border-white/10 dark:bg-neutral-950/70",
          ].join(" ")}
        >
          <Sidebar />
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          {/* Top bar */}
          <header className="sticky top-0 z-10 border-b border-neutral-200/80 bg-white/70 backdrop-blur dark:border-white/10 dark:bg-neutral-950/60">
            <div className="flex items-center justify-end gap-3 px-4 py-3 sm:px-6 lg:px-8">
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
          </header>

          <main className="flex-1 py-8">
            <div className="px-4 sm:px-6 lg:px-8">
              <Outlet />
            </div>
          </main>
        </div>
      </div>
    </div>
  );
}
