import { Navigate, Route, Routes } from "react-router-dom";

import AppLayout from "./layouts/AppLayout.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import PreRaceAnalysis from "./pages/PreRaceAnalysis.jsx";
import NotFound from "./pages/NotFound.jsx";


export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Dashboard />} />
        <Route path="/pre-race" element={<PreRaceAnalysis />} />

        <Route path="/dashboard" element={<Navigate to="/" replace />} />

        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
