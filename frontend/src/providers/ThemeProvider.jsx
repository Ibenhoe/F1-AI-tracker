import { createContext, useContext, useEffect, useMemo, useState } from "react";

const ThemeContext = createContext(null);

/**
 * theme:
 * - "system" (default)
 * - "dark"
 * - "light"
 */
const STORAGE_KEY = "f1ai.theme";

function getSystemPrefersDark() {
  return window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function applyThemeClass(theme) {
  const root = document.documentElement;

  const isDark =
    theme === "dark" || (theme === "system" && getSystemPrefersDark());

  root.classList.toggle("dark", isDark);
}

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "light" || stored === "dark" || stored === "system") {
      return stored;
    }
    return "system";
  });

  useEffect(() => {
    applyThemeClass(theme);
    localStorage.setItem(STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    if (!window.matchMedia) return;

    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => {
      applyThemeClass(theme);
    };

    if (media.addEventListener) media.addEventListener("change", handler);
    else media.addListener(handler);

    return () => {
      if (media.removeEventListener) media.removeEventListener("change", handler);
      else media.removeListener(handler);
    };
  }, [theme]);

  const value = useMemo(
    () => ({
      theme,
      setTheme,
      toggle: () =>
        setTheme((t) => (t === "dark" ? "light" : t === "light" ? "dark" : "dark")),
    }),
    [theme]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error("useTheme must be used within ThemeProvider");
  }
  return ctx;
}
