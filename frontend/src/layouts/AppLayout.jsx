import { Outlet } from "react-router-dom";
import Sidebar from "../components/Sidebar.jsx";

export default function AppLayout() {
  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <div className="flex min-h-screen">
        <aside className="w-72 border-r border-neutral-900">
          <Sidebar />
        </aside>

        <div className="flex-1">
          <main className="container py-8">
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  );
}
