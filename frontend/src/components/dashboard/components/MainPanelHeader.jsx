// src/components/dashboard/components/MainPanelHeader.jsx
import SegmentedControl from "./SegmentedControl";

export default function MainPanelHeader({
  value,
  onChange,
  items = [],
  title = "",
  subtitle = "",
  ariaLabel = "Main panel",
  kicker = "Live",
}) {
  return (
    <div className="mb-4 flex flex-col gap-3 sm:grid sm:grid-cols-3 sm:items-end">
      {/* Title block takes 2 columns on >= sm */}
      <div className="space-y-1 sm:col-span-2">
        <p className="text-[11px] font-semibold uppercase tracking-widest text-neutral-400 dark:text-neutral-600">
          {kicker}
        </p>

        <h2 className="text-base font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
          {title}
        </h2>

        {subtitle ? (
          <p className="text-xs text-neutral-500 dark:text-neutral-400">
            {subtitle}
          </p>
        ) : null}
      </div>

      {/* Control block sits right / centered */}
      <div className="sm:col-span-1 sm:flex sm:justify-end">
        <div className="w-full sm:w-auto">
          <SegmentedControl
            value={value}
            onChange={onChange}
            ariaLabel={ariaLabel}
            items={items}
          />
        </div>
      </div>
    </div>
  );
}