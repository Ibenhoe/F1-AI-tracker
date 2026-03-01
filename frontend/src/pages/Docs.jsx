// src/pages/Docs.jsx
import React from "react";
import { DocsContent } from "../components/docs";

import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";

export default function Docs() {
  return (
    <div className="space-y-5">
      {/* HEADER (Dashboard/Wiki style) */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
            Docs
          </h1>
          <p className="text-sm text-neutral-500 dark:text-neutral-400">
            Technical documentation for the real-time Formula 1 race prediction system.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div
            className={[
              "flex flex-wrap items-center gap-2 rounded-2xl px-3 py-2",
              "bg-white dark:bg-neutral-950/40",
              "ring-1 ring-black/5 dark:ring-white/10",
            ].join(" ")}
          >
            <Badge variant="accent">F1 AI Tracker</Badge>
            <Badge variant="neutral">Docs</Badge>
            <Badge variant="neutral">v1</Badge>
          </div>
        </div>
      </div>

      {/* CONTENT */}
      <Card className="p-5" clip>
        <div className="w-full">
          <DocsContent />
        </div>
      </Card>
    </div>
  );
}