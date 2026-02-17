import { createContext, useContext, useEffect, useMemo, useState } from "react";

const ThemeContext = createContext(null);
const STORAGE_KEY = "f1ai.theme";
const ACCENT_KEY = "f1ai.accent";

export const ACCENTS = [
  { id: "white", label: "White", hex: "#FFFFFF" },
  { id: "blue", label: "Blue", hex: "#3B82F6" },
  { id: "green", label: "Green", hex: "#22C55E" },
  { id: "orange", label: "Orange", hex: "#F97316" },
  { id: "purple", label: "Purple", hex: "#8B5CF6" },
  { id: "pink", label: "Pink", hex: "#EC4899" },
];

function getSystemPrefersDark() {
  return (
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  );
}

function applyThemeClass(theme) {
  const root = document.documentElement;
  const isDark = theme === "dark" || (theme === "system" && getSystemPrefersDark());
  root.classList.toggle("dark", isDark);
  return isDark;
}

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  const n = parseInt(h, 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

/**
 * Sets:
 * --accent: "r g b"
 * --accent-weak: same rgb, used with low alpha
 * --accent-strong: same rgb, used with higher alpha
 * --accent-fg: readable text on top of accent bg
 */
function applyAccentVars(accentId, hex, isDark) {
  const root = document.documentElement;
  const { r, g, b } = hexToRgb(hex);

  root.style.setProperty("--accent", `${r} ${g} ${b}`);

  // Use the same accent rgb, but provide semantic helpers for consistent UI usage
  root.style.setProperty("--accent-weak", `${r} ${g} ${b}`);
  root.style.setProperty("--accent-strong", `${r} ${g} ${b}`);

  // Foreground color when something uses accent as background.
  // If accent is white -> use dark text, else:
  // - in dark mode: use near-black text (better on bright accents)
  // - in light mode: use white text
  if (accentId === "white") {
    root.style.setProperty("--accent-fg", "17 24 39"); // neutral-900
  } else {
    root.style.setProperty("--accent-fg", isDark ? "10 10 10" : "255 255 255");
  }
}

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "light" || stored === "dark" || stored === "system") return stored;
    return "system";
  });

  const [accent, setAccent] = useState(() => {
    const stored = localStorage.getItem(ACCENT_KEY);
    const exists = ACCENTS.find((a) => a.id === stored);
    return exists?.id ?? "white";
  });

  useEffect(() => {
    const isDark = applyThemeClass(theme);
    localStorage.setItem(STORAGE_KEY, theme);

    const hex = ACCENTS.find((a) => a.id === accent)?.hex ?? "#FFFFFF"; // default WHITE
    applyAccentVars(accent, hex, isDark);
    localStorage.setItem(ACCENT_KEY, accent);
  }, [theme, accent]);

  useEffect(() => {
    if (!window.matchMedia) return;

    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => {
      // re-apply theme + accent when system theme flips
      const isDark = applyThemeClass(theme);
      const hex = ACCENTS.find((a) => a.id === accent)?.hex ?? "#FFFFFF";
      applyAccentVars(accent, hex, isDark);
    };

    if (media.addEventListener) media.addEventListener("change", handler);
    else media.addListener(handler);

    return () => {
      if (media.removeEventListener) media.removeEventListener("change", handler);
      else media.removeListener(handler);
    };
  }, [theme, accent]);

  const value = useMemo(
    () => ({
      theme,
      setTheme,
      toggle: () =>
        setTheme((t) => (t === "dark" ? "light" : t === "light" ? "dark" : "dark")),
      accent,
      setAccent,
      accents: ACCENTS,
    }),
    [theme, accent]
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}
