import { cn } from "../../lib/cn.js";

const base =
    "inline-flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors" +
    "focus:outline-none focus:ring-2 focus:ring-neutral-700 focus:ring-offset-2 focus:ring-offset-neutral-950" +
    "disabled:pointer-events-none disabled:opacity:50";

const variants = {
    primary: "bg-neutral-100 text-neutral-950 hover:bg-neutral-200",
    secondary: "bg-neutral-900 text-neutral-100 border border-neutral-800 hover:border-neutral-700",
    ghost: "bg-transparent text-neutral-200 hover:bg-neutral-900",
    danger: "bg-red-600 text-white hover:bg-red-500",
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