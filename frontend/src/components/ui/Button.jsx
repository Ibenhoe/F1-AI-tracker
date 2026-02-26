import { cn } from "../../lib/cn.js";

const base =
  "inline-flex items-center justify-center gap-2 rounded-2xl px-3 py-2 text-sm font-medium " +
  "transition-colors duration-150 " +
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))] " +
  "disabled:pointer-events-none disabled:opacity-50";

const variants = {
  primary: cn(
    "bg-[rgb(var(--accent))] text-[rgb(var(--accent-fg))]",
    "hover:brightness-95 active:brightness-90"
  ),

  secondary: cn(
    // iOS-like neutral button
    "bg-neutral-100 text-neutral-900",
    "hover:bg-neutral-200/70 active:bg-neutral-200",
    "dark:bg-white/10 dark:text-neutral-100 dark:hover:bg-white/15 dark:active:bg-white/20"
  ),

  ghost: cn(
    // toolbar-like pill
    "bg-transparent text-neutral-700",
    "hover:bg-black/5 active:bg-black/10",
    "dark:text-neutral-200 dark:hover:bg-white/10 dark:active:bg-white/15"
  ),

  danger: cn(
    "bg-red-600 text-white",
    "hover:brightness-95 active:brightness-90"
  ),
};

export default function Button({
  className,
  variant = "secondary",
  type = "button",
  children,
  ...props
}) {
  return (
    <button
      type={type}
      className={cn(base, variants[variant] ?? variants.secondary, className)}
      {...props}
    >
      {children}
    </button>
  );
}