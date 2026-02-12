import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <div className="space-y-3">
      <h1 className="text-2xl font-semibold tracking-tight">Page not found</h1>
      <p className="text-neutral-300">
        The page you’re looking for doesn’t exist.
      </p>
      <Link
        to="/"
        className="inline-flex items-center rounded-lg border border-neutral-800 px-3 py-2 text-sm hover:border-neutral-700"
      >
        Back to dashboard
      </Link>
    </div>
  );
}
