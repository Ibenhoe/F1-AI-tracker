import { Outlet } from "react-router-dom";
import { Moon, Sun } from "lucide-react";

import Sidebar from "../components/Sidebar.jsx";
import Button from "../components/ui/Button.jsx";
import { useTheme } from "../providers/ThemeProvider.jsx";

export default function AppLayout() {
  const { theme, toggle } = useTheme();

  return (
    <div className="min-h-screen bg-white text-neutral-900 dark:bg-neutral-950 dark:text-neutral-100">
      <div className="flex min-h-screen">
        <aside className="w-72 border-r border-neutral-200 dark:border-neutral-900">
          <Sidebar />
        </aside>

        <div className="flex min-w-0 flex-1 flex-col">
          {/* Top bar */}
          <header className="sticky top-0 z-10 border-b border-neutral-200 bg-white/70 backdrop-blur dark:border-neutral-900 dark:bg-neutral-950/60">
            <div className="container flex items-center justify-end py-3">
              <Button
                variant="ghost"
                onClick={toggle}
                className="border border-neutral-200 bg-white/40 text-neutral-900 hover:bg-white dark:border-neutral-800 dark:bg-neutral-950/40 dark:text-neutral-100 dark:hover:bg-neutral-900"
                aria-label="Toggle theme"
                title={`Theme: ${theme}`}
              >
                <Sun size={16} className="dark:hidden" />
                <Moon size={16} className="hidden dark:block" />
                <span className="text-sm">Theme</span>
              </Button>
            </div>
          </header>

          <main className="container flex-1 py-8">
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  );
}
