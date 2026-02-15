import { useEffect, useState } from 'react';

export default function Toast({ message, type = 'success', duration = 3000, onClose }) {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    if (!isVisible) return;

    const timer = setTimeout(() => {
      setIsVisible(false);
      onClose?.();
    }, duration);

    return () => clearTimeout(timer);
  }, [isVisible, duration, onClose]);

  if (!isVisible) return null;

  // Determine styles based on type
  const getStyles = () => {
    switch (type) {
      case 'success':
        return {
          bg: 'bg-emerald-50 dark:bg-emerald-950/30',
          border: 'border-emerald-200 dark:border-emerald-900/40',
          text: 'text-emerald-900 dark:text-emerald-100',
        };
      case 'error':
        return {
          bg: 'bg-red-50 dark:bg-red-950/30',
          border: 'border-red-200 dark:border-red-900/40',
          text: 'text-red-900 dark:text-red-100',
        };
      case 'warning':
        return {
          bg: 'bg-amber-50 dark:bg-amber-950/30',
          border: 'border-amber-200 dark:border-amber-900/40',
          text: 'text-amber-900 dark:text-amber-100',
        };
      default:
        return {
          bg: 'bg-white dark:bg-neutral-950/40',
          border: 'border-neutral-200 dark:border-neutral-800',
          text: 'text-neutral-900 dark:text-neutral-100',
        };
    }
  };

  const styles = getStyles();

  return (
    <div
      className={`
        fixed top-6 right-6 z-50
        animate-in fade-in slide-in-from-top-2
        rounded-xl border px-4 py-3 shadow-lg
        ${styles.bg} ${styles.border} ${styles.text}
        max-w-sm
      `}
    >
      <span className="text-sm font-medium">{message}</span>
    </div>
  );
}
