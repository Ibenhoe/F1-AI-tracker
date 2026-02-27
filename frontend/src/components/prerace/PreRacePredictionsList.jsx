import { ArrowDown, ArrowUp, Minus } from "lucide-react";
import Card from "../ui/Card";
import { getTeamColor } from "../../utils/teamColors";

function clamp(n, a, b) {
    const x = Number(n);
    if (!Number.isFinite(x)) return a;
    return Math.max(a, Math.min(b, x));
}

function PosDelta({ value }) {
    const v = Number(value ?? 0);

    if (v > 0) {
        return (
            <span className="inline-flex items-center justify-end gap-1 text-xs font-semibold text-emerald-600 dark:text-emerald-400 tabular-nums">
                <ArrowUp size={14} />+{v}
            </span>
        );
    }

    if (v < 0) {
        return (
            <span className="inline-flex items-center justify-end gap-1 text-xs font-semibold text-red-600 dark:text-red-400 tabular-nums">
                <ArrowDown size={14} />
                {v}
            </span>
        );
    }

    return (
        <span className="inline-flex items-center justify-end gap-1 text-xs font-semibold text-neutral-500 dark:text-neutral-400 tabular-nums">
            <Minus size={14} />0
        </span>
    );
}

function Confidence({ value }) {
    const v = clamp(value, 0, 100);

    return (
        <div className="flex items-center justify-end gap-2">
            <div className="h-2 w-20 overflow-hidden rounded-full bg-black/[0.06] dark:bg-white/[0.08]">
                <div
                    className="h-2 rounded-full bg-[rgb(var(--accent))]"
                    style={{ width: `${v}%` }}
                    aria-hidden="true"
                />
            </div>
            <span className="w-10 text-right text-xs font-semibold tabular-nums text-neutral-700 dark:text-neutral-300">
                {Math.round(v)}%
            </span>
        </div>
    );
}

/**
 * predictions: array of objects from backend
 * expected keys (best-effort):
 *  - driver_name | driver
 *  - team
 *  - grid_position
 *  - confidence
 */
export default function PreRacePredictionsList({ predictions }) {
    const list = Array.isArray(predictions) ? predictions.slice(0, 10) : [];

    return (
        <Card className="overflow-hidden" clip bordered>
            {/* Header row (DriversList style) */}
            <div
                className={[
                    "grid min-w-0",
                    "grid-cols-[44px_1fr_76px_64px_120px] md:grid-cols-[44px_1fr_92px_76px_64px_140px]",
                    "items-center gap-3 px-4 py-2.5",
                    "text-[11px] font-semibold uppercase tracking-widest",
                    "text-neutral-500 dark:text-neutral-400",
                    "border-b border-black/5 dark:border-white/10",
                    "bg-transparent",
                ].join(" ")}
            >
                <div className="text-center">Pos</div>
                <div className="min-w-0">Driver</div>
                <div className="hidden text-right md:block">Grid</div>
                <div className="text-right">Grid</div>
                <div className="text-right">Δ</div>
                <div className="hidden text-right md:block">Confidence</div>
                <div className="text-right md:hidden">Conf</div>
            </div>

            {/* Rows */}
            <div className="divide-y divide-black/5 dark:divide-white/10">
                {list.length > 0 ? (
                    list.map((pred, idx) => {
                        const driver = pred.driver_name || pred.driver || "Unknown";
                        const team = pred.team || "—";
                        const teamColor = getTeamColor(team);

                        const predictedPos = idx + 1;

                        const grid =
                            Number.isFinite(Number(pred.grid_position)) && Number(pred.grid_position) > 0
                                ? Number(pred.grid_position)
                                : null;

                        // delta = grid - predicted (positive means predicted improved vs grid)
                        const delta = grid == null ? 0 : grid - predictedPos;

                        const conf = clamp(Number(pred.confidence ?? 0), 0, 100);

                        return (
                            <div
                                key={`${idx}-${driver}-${team}`}
                                className={[
                                    "relative",
                                    "grid min-w-0",
                                    "grid-cols-[44px_1fr_76px_64px_120px] md:grid-cols-[44px_1fr_92px_76px_64px_140px]",
                                    "items-center gap-3 px-4 py-3",
                                    "bg-transparent hover:bg-black/[0.02] dark:hover:bg-white/[0.03]",
                                    "transition-colors",
                                ].join(" ")}
                            >
                                {/* Team hairline accent */}
                                <div
                                    className="absolute left-0 top-0 h-full w-[2px]"
                                    style={{
                                        backgroundColor: teamColor ? teamColor : "rgba(0,0,0,0.08)",
                                        opacity: teamColor ? 0.75 : 1,
                                    }}
                                    aria-hidden="true"
                                />

                                {/* Predicted position */}
                                <div className="text-center text-sm font-semibold tabular-nums text-neutral-900 dark:text-neutral-100">
                                    {predictedPos}
                                </div>

                                {/* Driver + team */}
                                <div className="min-w-0">
                                    <div className="truncate text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                                        {driver}
                                    </div>
                                    <div className="truncate text-xs text-neutral-500 dark:text-neutral-400">
                                        {team}
                                    </div>

                                    {/* Mobile: show confidence under name */}
                                    <div className="mt-1 flex items-center justify-between gap-2 md:hidden">
                                        <span className="text-[11px] font-medium text-neutral-400 dark:text-neutral-600">
                                            Confidence
                                        </span>
                                        <span className="text-xs font-semibold tabular-nums text-neutral-700 dark:text-neutral-300">
                                            {Math.round(conf)}%
                                        </span>
                                    </div>
                                </div>

                                {/* Grid (desktop label column) */}
                                <div className="hidden text-right md:block">
                                    <div className="text-sm font-semibold tabular-nums text-neutral-900 dark:text-neutral-100">
                                        {grid != null ? `P${grid}` : "—"}
                                    </div>
                                </div>

                                {/* Grid (always visible) */}
                                <div className="text-right text-sm font-semibold tabular-nums text-neutral-900 dark:text-neutral-100">
                                    {grid != null ? `P${grid}` : "—"}
                                </div>

                                {/* Δ */}
                                <div className="text-right">
                                    {grid == null ? (
                                        <span className="text-xs font-semibold text-neutral-400 dark:text-neutral-600 tabular-nums">
                                            —
                                        </span>
                                    ) : (
                                        <PosDelta value={delta} />
                                    )}
                                </div>

                                {/* Confidence (desktop) */}
                                <div className="hidden md:block">
                                    <Confidence value={conf} />
                                </div>

                                {/* Confidence (mobile compact) */}
                                <div className="text-right md:hidden">
                                    <span className="text-xs font-semibold tabular-nums text-neutral-700 dark:text-neutral-300">
                                        {Math.round(conf)}%
                                    </span>
                                </div>
                            </div>
                        );
                    })
                ) : (
                    <div className="px-6 py-12 text-center">
                        <div className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
                            Waiting for race data…
                        </div>
                        <div className="mt-1 text-xs text-neutral-500 dark:text-neutral-500">
                            Standings will appear once telemetry is available.
                        </div>
                    </div>
                )}
            </div>
        </Card>
    );
}