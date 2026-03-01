// src/pages/Docs.jsx
import React, { useEffect, useRef } from "react";
import { DocsContent } from "../components/docs";

import Card from "../components/ui/Card";
import Badge from "../components/ui/Badge";

function publishActiveDocsSection(id) {
  window.dispatchEvent(new CustomEvent("docs:active-section", { detail: { id } }));
}

export default function Docs() {
  const lastIdRef = useRef(null);

  useEffect(() => {
    const sections = Array.from(document.querySelectorAll("section[id]"));
    if (!sections.length) return;

    const setActive = (id) => {
      if (!id || lastIdRef.current === id) return;
      lastIdRef.current = id;
      publishActiveDocsSection(id);
    };

    // initialize (top of page)
    setActive(sections[0].id);

    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);

        const top = visible[0]?.target?.id;
        if (top) setActive(top);
      },
      {
        // tuned for your scroll-mt-24 / sticky header vibe
        root: null,
        rootMargin: "-20% 0px -70% 0px",
        threshold: [0.05, 0.15, 0.3, 0.5, 0.75],
      }
    );

    sections.forEach((s) => io.observe(s));
    return () => io.disconnect();
  }, []);

  return (
    <div className="space-y-5">
      {/* HEADER */}
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