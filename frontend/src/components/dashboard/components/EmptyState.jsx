// src/components/dashboard/components/EmptyState.jsx
export default function EmptyState({
  title,
  subtitle,
  minHeight = 220,
  icon: Icon = null,
}) {
  if (!title) return null;

  return (
    <div
      className="flex h-full items-center justify-center"
      style={{ minHeight }}
      role="status"
      aria-live="polite"
    >
      <div className="text-center">
        {Icon ? (
          <div className="mb-3 flex justify-center text-neutral-400 dark:text-neutral-500">
            <Icon size={22} />
          </div>
        ) : null}

        <p className="text-sm font-medium text-neutral-900 dark:text-neutral-50">
          {title}
        </p>

        {subtitle ? (
          <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
            {subtitle}
          </p>
        ) : null}
      </div>
    </div>
  );
}