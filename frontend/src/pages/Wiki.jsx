import { useState, useEffect } from "react";

export default function Wiki() {
  // State for filters
  const [selectedYear, setSelectedYear] = useState(2024);
  const [races, setRaces] = useState([]);
  const [selectedRaceId, setSelectedRaceId] = useState("");
  const [viewType, setViewType] = useState("race"); // 'race', 'grid', 'qualifying'
  
  // State for data
  const [tableData, setTableData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Generate years list (e.g., 1950 to current)
  const currentYear = new Date().getFullYear();
  const years = Array.from({ length: currentYear - 1950 + 1 }, (_, i) => currentYear - i);

  // Mapping for driver references to full names
  const driverMapping = {
    "hamilton": "Lewis Hamilton",
    "heidfeld": "Nick Heidfeld",
    "rosberg": "Nico Rosberg",
    "alonso": "Fernando Alonso",
    "kovalainen": "Heikki Kovalainen",
    "nakajima": "Kazuki Nakajima",
    "bourdais": "Sébastien Bourdais",
    "raikkonen": "Kimi Räikkönen",
    "kubica": "Robert Kubica",
    "glock": "Timo Glock",
    "sato": "Takuma Sato",
    "piquet_jr": "Nelson Piquet Jr.",
    "massa": "Felipe Massa",
    "coulthard": "David Coulthard",
    "trulli": "Jarno Trulli",
    "sutil": "Adrian Sutil",
    "webber": "Mark Webber",
    "button": "Jenson Button",
    "davidson": "Anthony Davidson",
    "vettel": "Sebastian Vettel",
    "fisichella": "Giancarlo Fisichella",
    "barrichello": "Rubens Barrichello",
    "ralf_schumacher": "Ralf Schumacher",
    "michael_schumacher": "Michael Schumacher",
    "wurz": "Alexander Wurz",
    "speed": "Scott Speed",
    "albers": "Christijan Albers",
    "liuzzi": "Vitantonio Liuzzi",
    "montoya": "Juan Pablo Montoya",
    "klien": "Christian Klien",
    "villeneuve": "Jacques Villeneuve",
    "panis": "Olivier Panis",
    "matta": "Cristiano da Matta",
    "frentzen": "Heinz-Harald Frentzen",
    "yoong": "Alex Yoong",
    "irvine": "Eddie Irvine",
    "rosa": "Pedro de la Rosa",
    "bernoldi": "Enrique Bernoldi",
    "verstappen": "Jos Verstappen",
    "gene": "Marc Gené",
    "mazzacane": "Gastón Mazzacane",
    "burti": "Luciano Burti",
    "marques": "Tarso Marques",
    "badoer": "Luca Badoer",
    "zanardi": "Alessandro Zanardi",
    "takagi": "Toranosuke Takagi",
    "brundle": "Martin Brundle",
    "montermini": "Andrea Montermini",
    "lavaggi": "Giovanni Lavaggi",
    "sospiri": "Vincenzo Sospiri",
    "morbidelli": "Gianni Morbidelli",
    "fontana": "Norberto Fontana",
    "lamy": "Pedro Lamy",
    "katayama": "Ukyo Katayama",
    "damon_hill": "Damon Hill",
    "magnussen": "Jan Magnussen",
    "karthikeyan": "Narain Karthikeyan",
    "monteiro": "Tiago Monteiro",
    "friesacher": "Patrick Friesacher",
    "doornbos": "Robert Doornbos",
    "ide": "Yuji Ide",
    "montagny": "Franck Montagny",
    "yamamoto": "Sakon Yamamoto",
    "max_verstappen": "Max Verstappen",
    "perez": "Sergio Pérez",
    "leclerc": "Charles Leclerc",
    "sainz": "Carlos Sainz",
    "russell": "George Russell",
    "norris": "Lando Norris"
  };

  // Helper to format names (e.g. "max_verstappen" -> "Max Verstappen")
  const formatName = (name) => {
    if (!name) return "";
    const lowerName = name.toString().toLowerCase();
    if (driverMapping[lowerName]) {
        return driverMapping[lowerName];
    }
    return name
      .toString()
      .replace(/_/g, " ")
      .split(" ")
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(" ");
  };

  // 1. Fetch Races when Year changes
  useEffect(() => {
    async function fetchRaces() {
      try {
        // Fetch from backend (which reads from your CSV/DB)
        const response = await fetch(`http://localhost:5000/api/races/${selectedYear}`);
        if (!response.ok) {
            throw new Error(`Server antwoordde met status: ${response.status}`);
        }
        const data = await response.json();
        setRaces(data);
        setError(null); // Clear error on success
        
        // Auto-select first race if available
        if (data.length > 0) {
          setSelectedRaceId(data[0].raceId);
        } else {
          setRaces([]);
          setSelectedRaceId("");
        }
      } catch (err) {
        console.error("Failed to fetch races", err);
        setError(`Fout bij laden races: ${err.message}. Controleer of de backend draait en de routes heeft.`);
        setRaces([]);
      }
    }
    fetchRaces();
  }, [selectedYear]);

  // 2. Fetch Session Data when Race or ViewType changes
  useEffect(() => {
    if (!selectedRaceId) return;

    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(`http://localhost:5000/api/wiki/${selectedRaceId}/${viewType}`);
        if (!response.ok) throw new Error("Failed to load data");
        const data = await response.json();
        setTableData(data);
      } catch (err) {
        setError("Could not load data.");
        setTableData([]);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [selectedRaceId, viewType]);

  return (
    <div className="min-h-screen bg-white dark:bg-neutral-950 text-neutral-900 dark:text-white p-6">
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <div className="mb-8 border-b border-neutral-200 dark:border-neutral-800 pb-4">
          <h1 className="text-3xl font-bold text-[rgb(var(--accent))]">F1 Historical Wiki</h1>
          <p className="text-neutral-600 dark:text-neutral-400 mt-2">Explore race results, qualifying sessions, and starting grids.</p>
        </div>

        {error && (
          <div className="mb-6 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 px-4 py-3 rounded-lg">
            {error}
          </div>
        )}

        {/* Controls Section */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 p-6 rounded-lg shadow-sm">
          
          {/* Year Selector */}
          <div>
            <label className="block text-sm font-medium text-neutral-600 dark:text-neutral-400 mb-2">Season</label>
            <select 
              value={selectedYear}
              onChange={(e) => setSelectedYear(Number(e.target.value))}
              className="w-full bg-white dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 text-neutral-900 dark:text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-red-500"
            >
              {years.map(y => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </div>

          {/* Race Selector */}
          <div>
            <label className="block text-sm font-medium text-neutral-600 dark:text-neutral-400 mb-2">Grand Prix</label>
            <select 
              value={selectedRaceId}
              onChange={(e) => setSelectedRaceId(e.target.value)}
              disabled={races.length === 0}
              className="w-full bg-white dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 text-neutral-900 dark:text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-red-500 disabled:opacity-50"
            >
              {races.map(race => (
                <option key={race.raceId} value={race.raceId}>
                  Round {race.round}: {race.name}
                </option>
              ))}
            </select>
          </div>

          {/* Session Type Dropdown (The requested feature) */}
          <div>
            <label className="block text-sm font-medium text-neutral-600 dark:text-neutral-400 mb-2">Session View</label>
            <select 
              value={viewType}
              onChange={(e) => setViewType(e.target.value)}
              className="w-full bg-white dark:bg-neutral-800 border border-neutral-200 dark:border-neutral-700 text-neutral-900 dark:text-white rounded-md px-4 py-2 focus:outline-none focus:ring-2 focus:ring-red-500"
            >
              <option value="race">🏁 Race Result</option>
              <option value="grid">🚦 Starting Grid</option>
              <option value="qualifying">⏱️ Qualifying</option>
            </select>
          </div>
        </div>

        {/* Data Display */}
        <div className="bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-lg shadow-sm overflow-hidden">
          <div className="p-4 border-b border-neutral-200 dark:border-neutral-800 flex justify-between items-center">
            <h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">
              {viewType === 'race' && "Race Classification"}
              {viewType === 'grid' && "Starting Grid"}
              {viewType === 'qualifying' && "Qualifying Results"}
            </h2>
            {loading && <span className="text-sm text-yellow-500 animate-pulse">Loading data...</span>}
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead className="bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-300 uppercase text-xs">
                <tr>
                  <th className="px-6 py-3">Pos</th>
                  <th className="px-6 py-3">Driver</th>
                  <th className="px-6 py-3">Team</th>
                  
                  {/* Dynamic Columns based on View Type */}
                  {viewType === 'race' && (
                    <>
                      <th className="px-6 py-3">Time/Status</th>
                      <th className="px-6 py-3">Points</th>
                    </>
                  )}
                  {viewType === 'grid' && (
                    <th className="px-6 py-3">Quali Time</th>
                  )}
                  {viewType === 'qualifying' && (
                    <>
                      <th className="px-6 py-3">Q1</th>
                      <th className="px-6 py-3">Q2</th>
                      <th className="px-6 py-3">Q3</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-200 dark:divide-neutral-800">
                {loading ? (
                  <tr>
                    <td colSpan="6" className="px-6 py-8 text-center text-neutral-500 dark:text-neutral-400">
                      Fetching data...
                    </td>
                  </tr>
                ) : tableData.length === 0 ? (
                  <tr>
                    <td colSpan="6" className="px-6 py-8 text-center text-neutral-500 dark:text-neutral-400">
                      No data available for this session.
                    </td>
                  </tr>
                ) : (
                  tableData.map((row, index) => (
                    <tr key={index} className="hover:bg-neutral-50 dark:hover:bg-neutral-800/50 transition-colors">
                      <td className="px-6 py-4 font-medium text-neutral-700 dark:text-neutral-300">{row.position}</td>
                      <td className="px-6 py-4 font-bold text-neutral-900 dark:text-neutral-100">{formatName(row.driver)}</td>
                      <td className="px-6 py-4 text-neutral-500 dark:text-neutral-400">{row.team}</td>
                      
                      {viewType === 'race' && (
                        <>
                          <td className="px-6 py-4 font-mono text-sm text-neutral-600 dark:text-neutral-300">{row.time}</td>
                          <td className="px-6 py-4 text-green-400 font-bold">{row.points > 0 ? `+${row.points}` : ''}</td>
                        </>
                      )}
                      {viewType === 'grid' && (
                        <td className="px-6 py-4 font-mono text-sm text-neutral-600 dark:text-neutral-300">{row.time}</td>
                      )}
                      {viewType === 'qualifying' && (
                        <>
                          <td className="px-6 py-4 font-mono text-sm text-neutral-500 dark:text-neutral-400">{row.q1}</td>
                          <td className="px-6 py-4 font-mono text-sm text-neutral-500 dark:text-neutral-400">{row.q2}</td>
                          <td className="px-6 py-4 font-mono text-sm text-neutral-900 dark:text-neutral-100 font-bold">{row.q3}</td>
                        </>
                      )}
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  );
}