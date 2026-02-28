// src/components/wiki/components/DriverModal.jsx
import { X } from "lucide-react";
import { getTeamColor } from "../../../utils/teamColors";
import Card from "../../ui/Card";

function Stat({ label, value, mono = false }) {
  return (
    <Card className="p-3" clip bordered>
      <p className="text-[11px] font-semibold uppercase tracking-widest text-neutral-400 dark:text-neutral-600">
        {label}
      </p>
      <p
        className={[
          "mt-1 text-base font-semibold",
          mono ? "font-mono tabular-nums" : "tabular-nums",
          "text-neutral-900 dark:text-neutral-100",
        ].join(" ")}
      >
        {value}
      </p>
    </Card>
  );
}

export default function DriverModal({
  selectedDriver,
  onClose,
  driverWiki,
  driverWikiLoading,
}) {
  if (!selectedDriver) return null;

  const team = selectedDriver.team || "—";
  const teamColor = getTeamColor(team) || "rgb(var(--accent))";

  const lastInitial =
    selectedDriver.fullName?.split(" ").slice(-1)[0]?.charAt(0) || "D";

  const pos =
    selectedDriver.row?.position != null ? `P${selectedDriver.row.position}` : null;

  const ptsRaw = Number(selectedDriver.row?.points ?? 0);
  const pts = ptsRaw > 0 ? `+${ptsRaw}` : ptsRaw === 0 ? "0" : null;

  const time = selectedDriver.row?.time ? String(selectedDriver.row.time) : null;

  const wikiUrl = driverWiki?.content_urls?.desktop?.page || null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Driver details"
    >
      <div
        className={[
          "relative w-full max-w-2xl overflow-hidden",
          "rounded-2xl",
          "bg-white dark:bg-neutral-950",
          "ring-1 ring-black/10 dark:ring-white/10",
        ].join(" ")}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top accent */}
        <div
          className="h-1 w-full opacity-90"
          style={{ backgroundColor: teamColor }}
          aria-hidden="true"
        />

        {/* Close */}
        <button
          type="button"
          onClick={onClose}
          className={[
            "absolute right-4 top-4",
            "inline-flex h-10 w-10 items-center justify-center rounded-2xl",
            "ring-1 ring-black/5 dark:ring-white/10",
            "bg-transparent hover:bg-black/[0.03] dark:hover:bg-white/[0.05]",
            "text-neutral-700 dark:text-neutral-200 transition-colors",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[rgb(var(--accent))]",
          ].join(" ")}
          aria-label="Close"
        >
          <X size={18} />
        </button>

        <div className="p-6 sm:p-7">
          {/* Header + stats as one aligned grid */}
          <div className="grid grid-cols-1 gap-5 sm:grid-cols-[210px_1fr] sm:items-stretch">
            {/* Photo (spans header + stats height) */}
            <div className="sm:row-span-2 self-stretch">
              <div className="h-full w-full rounded-2xl overflow-hidden bg-black/[0.04] dark:bg-white/[0.06] ring-1 ring-black/5 dark:ring-white/10">
                {driverWikiLoading ? (
                  <div className="flex h-full items-center justify-center">
                    <span className="text-xs text-neutral-500 dark:text-neutral-400">
                      Loading…
                    </span>
                  </div>
                ) : driverWiki?.thumbnail?.source ? (
                  <img
                    src={driverWiki.thumbnail.source}
                    alt={selectedDriver.fullName}
                    className="h-full w-full object-cover object-top"
                  />
                ) : (
                  <div className="flex h-full items-center justify-center">
                    <span className="text-6xl font-semibold text-neutral-700 dark:text-neutral-200">
                      {lastInitial}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Info */}
            <div className="min-w-0 pr-12">
              <p className="text-[11px] font-semibold uppercase tracking-widest text-neutral-400 dark:text-neutral-600">
                Driver
              </p>

              <h3 className="mt-1 text-2xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
                {selectedDriver.fullName}
              </h3>

              <p className="mt-1 text-sm font-medium" style={{ color: teamColor }}>
                {team}
              </p>

              <div className="mt-4 h-px w-full bg-black/5 dark:bg-white/10" />
            </div>

            {/* Session stats */}
            <div className="min-w-0 grid grid-cols-2 gap-3">
              {pos ? <Stat label="Position" value={pos} /> : null}
              {pts != null ? <Stat label="Points" value={pts} /> : null}
              {time ? (
                <div className="col-span-2">
                  <Stat label="Time / Status" value={time} mono />
                </div>
              ) : null}
            </div>
          </div>

          {/* Bio */}
          {driverWiki?.extract ? (
            <div className="mt-6">
              <p className="text-[11px] font-semibold uppercase tracking-widest text-neutral-400 dark:text-neutral-600">
                Biography
              </p>

              <p className="mt-2 text-sm text-neutral-700 dark:text-neutral-300 leading-relaxed line-clamp-8">
                {driverWiki.extract}
              </p>

              {wikiUrl ? (
                <a
                  href={wikiUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-3 inline-flex text-xs font-semibold text-[rgb(var(--accent))] hover:underline"
                >
                  Read more
                </a>
              ) : null}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}