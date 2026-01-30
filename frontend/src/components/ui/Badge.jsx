import { cn } from "../../lib/cn.js";

const variants = {
  neutral: "border-neutral-800 text-neutral-300 bg-neutral-900/60",
  success: "border-emerald-900/60 text-emerald-200 bg-emerald-950/40",
  warning: "border-amber-900/60 text-amber-200 bg-amber-950/40",
  danger: "border-red-900/60 text-red-200 bg-red-950/40",
};

export default function Badge({ className, variant = "neutral", children }) {
    return (
        <span
        className={cn(
            "inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium",
            variants[variant] ?? variants.neutral,
            className
        )}
        >
            {children}
        </span>
    );
}