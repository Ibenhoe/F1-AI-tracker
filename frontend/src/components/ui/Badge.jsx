import { cn } from "../../lib/cn.js";

const base =
  "inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium transition-colors";

const variants = {
  neutral: cn(
    "bg-black/[0.04] text-neutral-700 ring-1 ring-black/5",
    "dark:bg-white/[0.06] dark:text-neutral-200 dark:ring-white/10"
  ),

  accent: cn(
    "bg-[rgb(var(--accent)_/_0.10)] text-[rgb(var(--accent))] ring-1 ring-black/5",
    "dark:bg-[rgb(var(--accent)_/_0.18)] dark:text-[rgb(var(--accent))] dark:ring-white/10"
  ),

  success: cn(
    "bg-black/[0.04] text-emerald-700 ring-1 ring-black/5",
    "dark:bg-white/[0.06] dark:text-emerald-200 dark:ring-white/10"
  ),

  warning: cn(
    "bg-black/[0.04] text-amber-800 ring-1 ring-black/5",
    "dark:bg-white/[0.06] dark:text-amber-200 dark:ring-white/10"
  ),

  danger: cn(
    "bg-black/[0.04] text-red-700 ring-1 ring-black/5",
    "dark:bg-white/[0.06] dark:text-red-200 dark:ring-white/10"
  ),
};

export default function Badge({ className, variant = "neutral", children }) {
  return (
    <span className={cn(base, variants[variant] ?? variants.neutral, className)}>
      {children}
    </span>
  );
}