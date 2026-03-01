export default function CodeBlock({ children }) {
  return (
    <div
      className={[
        "my-5 overflow-x-auto",
        "rounded-2xl",
        "bg-neutral-950 dark:bg-black",
        "ring-1 ring-white/10",
      ].join(" ")}
    >
      <pre className="p-5 text-[13px] font-mono leading-relaxed text-neutral-200">
        <code>{children}</code>
      </pre>
    </div>
  );
}