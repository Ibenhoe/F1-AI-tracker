import { cn } from "../../lib/cn.js";

const base =
  "inline-flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium " +
  "transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 " +
  "disabled:pointer-events-none disabled:opacity-50 " +
  // Focus rings: clean parity
  "focus:ring-neutral-400 focus:ring-offset-white " +
  "dark:focus:ring-neutral-600 dark:focus:ring-offset-neutral-950";

const variants = {
  primary: cn(
    // LIGHT
    "bg-neutral-900 text-white hover:bg-neutral-800",
    // DARK (inverted primary for contrast)
    "dark:bg-neutral-50 dark:text-neutral-950 dark:hover:bg-neutral-200"
  ),

  secondary: cn(
    // LIGHT
    "bg-white text-neutral-900 border border-neutral-200 hover:bg-neutral-50",
    // DARK
    "dark:bg-neutral-900 dark:text-neutral-100 dark:border-neutral-800 dark:hover:bg-neutral-800"
  ),

  ghost: cn(
    // LIGHT
    "bg-transparent text-neutral-700 hover:bg-neutral-100",
    // DARK (avoid gray-on-gray slab)
    "dark:text-neutral-200 dark:hover:bg-neutral-800/60"
  ),

  danger: cn(
    "bg-red-600 text-white hover:bg-red-500 dark:bg-red-600 dark:hover:bg-red-500"
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
