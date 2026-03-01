export default function InlineCode({ children }) {
  return (
    <span className="font-mono font-semibold tabular-nums text-[rgb(var(--accent))]">
      {children}
    </span>
  );
}