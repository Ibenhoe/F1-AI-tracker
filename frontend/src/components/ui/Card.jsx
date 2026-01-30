import { cn } from "../../lib/cn.js"

export default function Card({ className, children }) {
    return (
        <div
            className={cn(
                "rounded-xl border border-neutral-800 bg-neutral-900/60 p-6",
                className
            )}
        >
            {children}
        </div>
    );
}