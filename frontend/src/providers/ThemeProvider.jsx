import { createContext, useContext, useEffect, useMemo, useState } from "react";

const ThemeContext = createContext(null);
const STORAGE_KEY = "f1ai.theme";
const ACCENT_KEY = "f1ai.accent";

export const ACCENTS = [
  {
    id: "mercedes",
    label: "Mercedes",
    primary: "#00D7B6",
    secondary: "#C0C0C0",
  },
  {
    id: "redbull",
    label: "Red Bull Racing",
    primary: "#4781D7",
    secondary: "#DC1E35",
  },
  {
    id: "ferrari",
    label: "Ferrari",
    primary: "#ED1131",
    secondary: "#FFD200",
  },
  {
    id: "mclaren",
    label: "McLaren",
    primary: "#F47600",
    secondary: "#00A3E0",
  },
  {
    id: "alpine",
    label: "Alpine",
    primary: "#00A1E8",
    secondary: "#FF4DA6",
  },
  {
    id: "racingbulls",
    label: "Racing Bulls",
    primary: "#6C98FF",
    secondary: "#1E2A78",
  },
  {
    id: "astonmartin",
    label: "Aston Martin",
    primary: "#229971",
    secondary: "#CEDC00",
  },
  {
    id: "williams",
    label: "Williams",
    primary: "#1868DB",
    secondary: "#00C3FF",
  },
  {
    id: "kicksauber",
    label: "Kick Sauber",
    primary: "#01C00E",
    secondary: "#000000",
  },
  {
    id: "haas",
    label: "Haas",
    primary: "#9C9FA2",
    secondary: "#E10600",
  },
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

function applyAccentVars(accentId, primaryHex, secondaryHex, isDark) {
  const root = document.documentElement;

  const p = hexToRgb(primaryHex);
  const s = hexToRgb(secondaryHex ?? primaryHex);

  root.style.setProperty("--accent", `${p.r} ${p.g} ${p.b}`);
  root.style.setProperty("--accent-weak", `${p.r} ${p.g} ${p.b}`);
  root.style.setProperty("--accent-strong", `${p.r} ${p.g} ${p.b}`);

  root.style.setProperty("--accent-secondary", `${s.r} ${s.g} ${s.b}`);

  root.style.setProperty("--accent-fg", isDark ? "10 10 10" : "255 255 255");

  root.style.setProperty("--accent-secondary-fg", isDark ? "10 10 10" : "255 255 255");

  if (primaryHex.toUpperCase() === "#FFFFFF") {
    root.style.setProperty("--accent-fg", "17 24 39");
  }
  if ((secondaryHex ?? primaryHex).toUpperCase() === "#FFFFFF") {
    root.style.setProperty("--accent-secondary-fg", "17 24 39");
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
    return exists?.id ?? "mercedes";
  });

  useEffect(() => {
    const isDark = applyThemeClass(theme);
    localStorage.setItem(STORAGE_KEY, theme);

    const a = ACCENTS.find((x) => x.id === accent) ?? ACCENTS[0];
    applyAccentVars(a.id, a.primary, a.secondary, isDark);
    localStorage.setItem(ACCENT_KEY, accent);
  }, [theme, accent]);

  useEffect(() => {
    if (!window.matchMedia) return;

    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => {
      // re-apply theme + accent when system theme flips
      const isDark = applyThemeClass(theme);
      const a = ACCENTS.find((x) => x.id === accent) ?? ACCENTS[0];
      applyAccentVars(a.id, a.primary, a.secondary, isDark);
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
