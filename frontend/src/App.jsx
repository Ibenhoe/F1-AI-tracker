import { Navigate, Route, Routes } from "react-router-dom";

import AppLayout from "./layouts/AppLayout.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import PreRaceAnalysis from "./pages/PreRaceAnalysis.jsx";
import RaceReplay from "./pages/RaceReplay.jsx";
import NotFound from "./pages/NotFound.jsx";
import Wiki from "./pages/Wiki.jsx";
import Docs from "./pages/Docs.jsx";


export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Dashboard />} />
        <Route path="/pre-race" element={<PreRaceAnalysis />} />
        <Route path="/race-replay" element={<RaceReplay />} />
        <Route path="/wiki" element={<Wiki />} />
        <Route path="/docs" element={<Docs />} />

        <Route path="/dashboard" element={<Navigate to="/" replace />} />

        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
