export function H3({ children }) {
  return (
    <h3 className="mt-8 mb-3 text-base font-semibold tracking-tight text-neutral-900 dark:text-neutral-50">
      {children}
    </h3>
  );
}

export function P({ children }) {
  return (
    <p className="text-sm leading-relaxed text-neutral-700 dark:text-neutral-300">
      {children}
    </p>
  );
}