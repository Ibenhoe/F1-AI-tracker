export default function Table({ headers, rows }) {
  return (
    <div
      className={[
        "my-4 overflow-x-auto",
        "rounded-2xl",
        "bg-white dark:bg-neutral-950/40",
        "ring-1 ring-black/5 dark:ring-white/10",
      ].join(" ")}
    >
      <table className="w-full text-sm text-left">
        <thead>
          <tr
            className={[
              "border-b border-black/5 dark:border-white/10",
              "text-[11px] font-semibold uppercase tracking-widest",
              "text-neutral-500 dark:text-neutral-400",
            ].join(" ")}
          >
            {headers.map((h) => (
              <th key={h} className="px-4 py-3 font-semibold">
                {h}
              </th>
            ))}
          </tr>
        </thead>

        <tbody className="divide-y divide-black/5 dark:divide-white/10">
          {rows.map((row, i) => (
            <tr
              key={i}
              className="transition-colors hover:bg-black/[0.015] dark:hover:bg-white/[0.02]"
            >
              {row.map((cell, j) => (
                <td
                  key={j}
                  className={[
                    "px-4 py-3 align-top",
                    j === 0
                      ? "font-mono font-semibold tabular-nums text-[rgb(var(--accent))]"
                      : "text-neutral-700 dark:text-neutral-300",
                  ].join(" ")}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}