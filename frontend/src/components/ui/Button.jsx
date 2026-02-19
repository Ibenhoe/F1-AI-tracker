import { cn } from "../../lib/cn.js";

const base =
  "inline-flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium " +
  "transition-all duration-150 ease-out " +
  "disabled:pointer-events-none disabled:opacity-50";

const variants = {
  primary: cn(
    "bg-[rgb(var(--accent))] text-[rgb(var(--accent-fg))]",
    "hover:brightness-95 active:brightness-90"
  ),

  secondary: cn(
    "bg-white text-neutral-900 border border-neutral-200",
    "hover:bg-neutral-50",
    "dark:bg-neutral-950 dark:text-neutral-100 dark:border-white/10 dark:hover:bg-neutral-900"
  ),

  ghost: cn(
    "bg-transparent text-neutral-700",
    "hover:bg-neutral-100",
    "dark:text-neutral-200 dark:hover:bg-neutral-800/60"
  ),

  danger: cn(
    "bg-red-600 text-white",
    "hover:bg-red-500 active:bg-red-600/90"
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
