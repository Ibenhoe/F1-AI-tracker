import { useState, useEffect, useRef } from "react";

const SECTIONS = [
  { id: "overview",       title: "Project Overview" },
  { id: "architecture",   title: "System Architecture" },
  { id: "data-layer",     title: "Data Layer" },
  { id: "ml-model",       title: "Machine Learning Model" },
  { id: "confidence",     title: "Confidence Scoring" },
  { id: "race-sim",       title: "Race Simulation" },
  { id: "api",            title: "Backend API" },
  { id: "socketio",       title: "Real-time Communication" },
  { id: "frontend",       title: "Frontend Architecture" },
  { id: "tire-strategy",  title: "Tire Strategy Model" },
  { id: "battle",         title: "Battle Detector" },
  { id: "wiki",           title: "Historical Wiki" },
  { id: "setup",          title: "Getting Started" },
];

function Section({ id, title, children }) {
  return (
    <section id={id} className="mb-16 scroll-mt-8">
      <h2 className="text-2xl font-bold text-neutral-900 dark:text-neutral-50 mb-6 pb-3 border-b border-neutral-200 dark:border-neutral-800">
        {title}
      </h2>
      <div className="space-y-4 text-neutral-700 dark:text-neutral-300 leading-relaxed">
        {children}
      </div>
    </section>
  );
}

function H3({ children }) {
  return <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mt-6 mb-2">{children}</h3>;
}

function P({ children }) {
  return <p className="text-[15px] leading-7">{children}</p>;
}

function Table({ headers, rows }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-neutral-200 dark:border-neutral-800 my-4">
      <table className="w-full text-sm text-left">
        <thead className="bg-neutral-100 dark:bg-neutral-800 text-xs uppercase tracking-wide text-neutral-500 dark:text-neutral-400">
          <tr>
            {headers.map(h => (
              <th key={h} className="px-4 py-3 font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-neutral-200 dark:divide-neutral-800">
          {rows.map((row, i) => (
            <tr key={i} className="hover:bg-neutral-50 dark:hover:bg-neutral-800/40 transition-colors">
              {row.map((cell, j) => (
                <td key={j} className={`px-4 py-3 ${j === 0 ? "font-mono text-[rgb(var(--accent))] font-medium" : "text-neutral-700 dark:text-neutral-300"}`}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CodeBlock({ children }) {
  return (
    <pre className="bg-neutral-100 dark:bg-neutral-800/80 rounded-lg p-4 overflow-x-auto text-sm font-mono text-neutral-800 dark:text-neutral-200 my-4 border border-neutral-200 dark:border-neutral-700">
      <code>{children}</code>
    </pre>
  );
}

function Callout({ label, children }) {
  return (
    <div className="border-l-4 border-[rgb(var(--accent))] pl-4 py-2 bg-neutral-50 dark:bg-neutral-800/40 rounded-r-lg my-4">
      {label && <p className="text-xs uppercase tracking-widest text-[rgb(var(--accent))] font-semibold mb-1">{label}</p>}
      <p className="text-sm text-neutral-700 dark:text-neutral-300 leading-relaxed">{children}</p>
    </div>
  );
}

export default function Docs() {
  const [active, setActive] = useState("overview");
  const observerRef = useRef(null);

  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      entries => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActive(entry.target.id);
            break;
          }
        }
      },
      { rootMargin: "-20% 0px -70% 0px", threshold: 0 }
    );
    SECTIONS.forEach(s => {
      const el = document.getElementById(s.id);
      if (el) observerRef.current.observe(el);
    });
    return () => observerRef.current?.disconnect();
  }, []);

  const scrollTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="min-h-screen bg-white dark:bg-neutral-950 text-neutral-900 dark:text-white">
      <div className="max-w-7xl mx-auto flex gap-10 p-6 pt-8">

        {/* Sidebar TOC */}
        <aside className="hidden lg:block w-56 xl:w-64 shrink-0">
          <div className="sticky top-8">
            <p className="text-xs uppercase tracking-widest text-neutral-500 dark:text-neutral-400 font-semibold mb-4">
              Contents
            </p>
            <nav className="space-y-0.5">
              {SECTIONS.map(s => (
                <button
                  key={s.id}
                  onClick={() => scrollTo(s.id)}
                  className={[
                    "w-full text-left px-3 py-2 rounded-md text-sm transition-colors",
                    active === s.id
                      ? "bg-neutral-100 dark:bg-neutral-800 text-[rgb(var(--accent))] font-medium"
                      : "text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-neutral-100 hover:bg-neutral-50 dark:hover:bg-neutral-800/50",
                  ].join(" ")}
                >
                  {s.title}
                </button>
              ))}
            </nav>
          </div>
        </aside>

        {/* Main content */}
        <main className="flex-1 min-w-0 max-w-3xl">

          {/* Page header */}
          <div className="mb-12 pb-6 border-b border-neutral-200 dark:border-neutral-800">
            <h1 className="text-4xl font-bold text-[rgb(var(--accent))] mb-3">F1 AI Tracker</h1>
            <p className="text-lg text-neutral-600 dark:text-neutral-400">
              Technical documentation for the real-time Formula 1 race prediction system.
            </p>
          </div>

          {/* ── OVERVIEW ─────────────────────────────────────── */}
          <Section id="overview" title="Project Overview">
            <P>
              F1 AI Tracker is a full-stack application that combines live Formula 1 race data with
              incremental machine learning to generate and continuously refine race winner predictions
              on a lap-by-lap basis. The system is designed around the principle that predictions should
              improve as a race unfolds — adapting to retirements, pit-stop strategies, and pace changes
              in real time rather than relying solely on pre-race analysis.
            </P>
            <P>
              The project is split into three logical layers: a Python backend responsible for data
              fetching, model training and race orchestration; a React frontend that visualises the live
              state of the race and AI predictions; and a real-time bridge between the two built on
              Flask-SocketIO.
            </P>
            <Callout label="Design philosophy">
              Predictions are intentionally capped at 85% confidence. No real-world race outcome is
              deterministic — mechanical failures, weather changes and racing incidents mean even the
              strongest favourite can retire. The system reflects this uncertainty at all times.
            </Callout>
          </Section>

          {/* ── ARCHITECTURE ─────────────────────────────────── */}
          <Section id="architecture" title="System Architecture">
            <P>
              The application follows a hub-and-spoke architecture where the Flask backend acts as the
              central hub. The frontend connects via WebSocket and receives pushed updates; it never
              polls the backend. All heavy computation — data fetching, model training, simulation — runs
              on the backend in background threads.
            </P>
            <Table
              headers={["Component", "Technology", "Responsibility"]}
              rows={[
                ["Backend",         "Python / Flask",          "REST API, race state management, model orchestration"],
                ["Real-time layer", "Flask-SocketIO",          "Bi-directional push of lap updates to all clients"],
                ["ML core",         "scikit-learn / XGBoost",  "Incremental SGD training + ensemble predictions"],
                ["Data layer",      "FastF1",                  "Fetches live and historical session data from the F1 API"],
                ["Frontend",        "React 19 + Vite",         "Live race dashboard with WebSocket client"],
                ["Styling",         "Tailwind CSS",            "Utility-first styling with dark-mode and team themes"],
              ]}
            />
          </Section>

          {/* ── DATA LAYER ───────────────────────────────────── */}
          <Section id="data-layer" title="Data Layer">
            <P>
              All raw F1 data is sourced through the <span className="font-mono text-[rgb(var(--accent))]">FastF1</span> Python
              library, which provides a high-level interface to the official F1 timing API. On first
              access, session data is downloaded and cached locally inside the system temp directory so
              that subsequent calls are instantaneous.
            </P>
            <H3>Historical training data</H3>
            <P>
              The model is pre-trained on five years of historical race data stored in{" "}
              <span className="font-mono text-[rgb(var(--accent))]">f1_historical_5years.csv</span>. This
              baseline gives the incremental learner a sensible starting point before any live data
              is available. A lighter processed version (<span className="font-mono text-[rgb(var(--accent))]">processed_f1_training_data.csv</span>)
              is used as a fallback if the main file is missing.
            </P>
            <H3>Live session data fields</H3>
            <P>
              For each driver on each lap, the following fields are extracted and fed to the model:
            </P>
            <Table
              headers={["Field", "Description"]}
              rows={[
                ["grid_position",      "Starting position in the race"],
                ["driver_age",         "Driver's age at the time of the race"],
                ["constructor_points", "Constructor Championship points — proxy for team strength"],
                ["circuit_id",         "Numeric ID of the circuit"],
                ["tire_compound",      "SOFT / MEDIUM / HARD / INTERMEDIATE / WET"],
                ["tire_age",           "Number of laps on the current set of tyres"],
                ["lap_time_ms",        "Current lap time in milliseconds"],
                ["position",           "Current race position"],
                ["pit_stops",          "Total pit stops completed"],
              ]}
            />
            <Callout label="Caching">
              FastF1 caches sessions in <span className="font-mono">{"{TEMP_DIR}/fastf1_cache/"}</span>. The path
              is resolved with Python's <span className="font-mono">tempfile.gettempdir()</span> so it works across
              Windows, macOS and Linux without configuration.
            </Callout>
          </Section>

          {/* ── ML MODEL ─────────────────────────────────────── */}
          <Section id="ml-model" title="Machine Learning Model">
            <P>
              The prediction system uses an ensemble of three model types trained on the same feature
              set. By combining their outputs rather than relying on a single model, the system is more
              robust to noise in any individual prediction.
            </P>
            <Table
              headers={["Model", "Type", "Role"]}
              rows={[
                ["SGDRegressor",      "Online linear model",  "Primary model — updated every lap via partial_fit()"],
                ["MLPRegressor",      "Neural network",       "Captures non-linear interactions between features"],
                ["GradientBoosting",  "Ensemble tree model",  "Provides stable baseline prediction from historical data"],
              ]}
            />
            <H3>Incremental learning</H3>
            <P>
              The key differentiator of this system is that the model is updated <em>during</em> the
              race, not just before it. After each lap, new lap-time and position data is fed to the
              SGDRegressor via <span className="font-mono text-[rgb(var(--accent))]">partial_fit()</span>.
              This means the model can react if a driver suddenly finds pace, suffers degradation, or
              pits unexpectedly. The MLP and GradientBoosting models are retrained periodically (not
              every lap) because they do not support online learning natively.
            </P>
            <H3>Feature engineering</H3>
            <P>
              Raw fields are preprocessed before training. Tire compound is one-hot encoded.
              Lap times are normalised relative to the session fastest lap. Constructor points are
              log-scaled to reduce the influence of outliers when a constructor dominates the season.
            </P>
            <H3>Pre-race model</H3>
            <P>
              A separate <span className="font-mono text-[rgb(var(--accent))]">prerace_model.py</span> uses
              qualifying pace and historical track performance to generate predictions before the
              formation lap. This gives the live model a head-start and provides the pre-race analysis
              view in the dashboard. It uses XGBoost trained on the five-year historical dataset.
            </P>
          </Section>

          {/* ── CONFIDENCE ───────────────────────────────────── */}
          <Section id="confidence" title="Confidence Scoring">
            <P>
              Confidence scores represent the model's certainty that a given driver will finish in the
              top 5. They are calculated from four components and hard-capped at 85%.
            </P>
            <CodeBlock>{`Confidence = base_score
           + pace_spread_bonus     (max +10%)
           + model_maturity_bonus  (max +3%)
           - volatility_penalty    (-3% to -9%)
           → clamped to [0%, 85%]

base_score           = 72% (starting point for any prediction)
pace_spread_bonus    = proportional to gap between P1 pace and field average
model_maturity_bonus = grows as more laps of data are seen (saturates ~lap 20)
volatility_penalty   = increases when positions change rapidly lap-over-lap`}</CodeBlock>
            <Callout label="Why 85%?">
              Even from pole position with dominant pace, a driver can retire from a mechanical failure,
              a collision, or a safety-car anomaly. Capping at 85% prevents overconfident predictions
              that ignore this fundamental racing uncertainty.
            </Callout>
          </Section>

          {/* ── RACE SIMULATION ──────────────────────────────── */}
          <Section id="race-sim" title="Race Simulation">
            <P>
              The race simulator (<span className="font-mono text-[rgb(var(--accent))]">race_simulator.py</span>)
              acts as the orchestration layer. It consumes lap-by-lap data from FastF1 and replays it
              through the system at a configurable speed multiplier. The simulation can be paused,
              resumed and scrubbed forward.
            </P>
            <H3>Lap cycle</H3>
            <Table
              headers={["Step", "Action"]}
              rows={[
                ["1. Fetch lap data",      "Pull the next lap's telemetry for all active drivers from the cached FastF1 session"],
                ["2. Update driver state", "Recalculate position, gap to leader, tire age, and pit stop count"],
                ["3. Detect events",       "Battle detector checks for on-track fights; event generator creates narrative entries"],
                ["4. Train model",         "Feed new lap data to the SGDRegressor via partial_fit()"],
                ["5. Generate predictions","Run ensemble inference and compute confidence scores"],
                ["6. Broadcast",           "Emit lap/update via SocketIO to all connected clients"],
                ["7. Rate limit",          "Wait according to simulation_speed before processing the next lap"],
              ]}
            />
            <H3>Simulation speed</H3>
            <P>
              Speed is controlled by a float multiplier. At <span className="font-mono">1.0</span> each
              synthetic lap takes approximately the same wall-clock time as a real F1 lap (~90 s).
              At <span className="font-mono">10.0</span> the same lap completes in ~9 s. The frontend
              sends speed changes via the{" "}
              <span className="font-mono text-[rgb(var(--accent))]">race/speed</span> SocketIO event.
            </P>
          </Section>

          {/* ── API ──────────────────────────────────────────── */}
          <Section id="api" title="Backend API">
            <P>
              The Flask backend exposes a REST API for race initialisation, pre-race analysis and
              historical data access, alongside the SocketIO real-time layer.
            </P>
            <H3>REST endpoints</H3>
            <Table
              headers={["Method", "Endpoint", "Description"]}
              rows={[
                ["GET",  "/api/health",                   "Service health check — returns status ok"],
                ["GET",  "/api/races",                    "List of available 2024 race rounds"],
                ["GET",  "/api/races/{year}",             "All races for a given season year"],
                ["POST", "/api/race/init",                "Initialise a race session — fetches FastF1 data"],
                ["POST", "/api/race/prerace-analysis",    "Run XGBoost pre-race prediction for a given race number"],
                ["GET",  "/api/wiki/{race_id}/{view}",    "Race / grid / qualifying results from training CSV"],
                ["GET",  "/api/docs",                     "Documentation section content (legacy)"],
              ]}
            />
            <H3>CORS policy</H3>
            <P>
              In development the API accepts requests from <span className="font-mono">localhost:5173</span> and{" "}
              <span className="font-mono">localhost:3000</span>. In production, the{" "}
              <span className="font-mono">CORS_ALLOWED_ORIGINS</span> environment variable must be set to a
              comma-separated list of trusted domain origins before deployment.
            </P>
          </Section>

          {/* ── SOCKETIO ─────────────────────────────────────── */}
          <Section id="socketio" title="Real-time Communication">
            <P>
              All live race state is delivered over a persistent WebSocket connection managed by
              Flask-SocketIO on the server and <span className="font-mono">socket.io-client</span> on the
              frontend. The backend uses <span className="font-mono">async_mode='threading'</span> which
              runs each client in its own thread without requiring an async event loop.
            </P>
            <H3>Client-to-server events</H3>
            <Table
              headers={["Event", "Payload", "Effect"]}
              rows={[
                ["race/start",  "{ speed: 1.0 }",  "Starts the lap-by-lap simulation loop"],
                ["race/pause",  "—",               "Pauses after the current lap completes"],
                ["race/resume", "—",               "Resumes from the paused lap"],
                ["race/speed",  "{ speed: 2.0 }",  "Changes simulation speed multiplier immediately"],
              ]}
            />
            <H3>Server-to-client events</H3>
            <Table
              headers={["Event", "Payload", "Description"]}
              rows={[
                ["connect_response", "{ status }",                                     "Sent on initial connection to confirm the socket is live"],
                ["lap/update",       "{ lap_number, drivers, predictions, events }",   "Full race state snapshot emitted after each lap"],
                ["race/finished",    "{ final_standings }",                            "Emitted once when the last lap completes"],
                ["init/progress",    "{ progress, status }",                           "FastF1 fetch progress during race initialisation"],
              ]}
            />
            <H3>Driver object schema</H3>
            <P>Each driver in the <span className="font-mono">drivers[]</span> array of a lap/update event carries:</P>
            <CodeBlock>{`{
  position:       1,               // Current race position
  driver_code:    "VER",           // 3-letter FIA code
  driver_name:    "Max Verstappen",
  team:           "Red Bull Racing",
  lap_time:       "1:28.473",      // Current lap time formatted
  tire_compound:  "SOFT",          // SOFT | MEDIUM | HARD | INTER | WET
  tire_age:       5,               // Laps completed on current tyre set
  pit_stops:      1,               // Total stops completed
  gap:            "+0.000",        // Gap to P1 in seconds (P1 shows "+0.000")
  laps_completed: 58
}`}</CodeBlock>
          </Section>

          {/* ── FRONTEND ─────────────────────────────────────── */}
          <Section id="frontend" title="Frontend Architecture">
            <P>
              The React frontend is a single-page application built with Vite. It uses React Router for
              client-side navigation between the main views. Global race state received from the
              WebSocket is managed through React context rather than a third-party state library to
              minimise dependencies.
            </P>
            <H3>Page structure</H3>
            <Table
              headers={["Route", "Page", "Purpose"]}
              rows={[
                ["/",       "Dashboard", "Live race simulation — driver list, predictions, model metrics"],
                ["/wiki",   "Wiki",      "Historical race results, qualifying and grids with circuit info"],
                ["/docs",   "Docs",      "This documentation page"],
                ["/replay", "Replay",    "Post-race replay of a completed session"],
              ]}
            />
            <H3>Key components</H3>
            <Table
              headers={["Component", "Responsibility"]}
              rows={[
                ["RaceSelector",       "Race and year selection — triggers /api/race/init"],
                ["DriversList",        "Scrolling live leaderboard with team colours and gaps"],
                ["PredictionsPanel",   "Top-5 AI predictions with animated confidence bars"],
                ["ModelMetricsPanel",  "Running accuracy, MAE and training sample count"],
                ["TrackRenderer",      "Canvas-based simplified track map with driver dots"],
                ["WeatherWidget",      "Temperature, humidity and track condition display"],
                ["NotificationsPanel", "Live event feed — overtakes, pit stops, fastest laps"],
                ["RaceControls",       "Play / Pause / Speed controls that emit SocketIO events"],
              ]}
            />
            <H3>Team theming</H3>
            <P>
              The UI supports 10 team colour themes selectable from the settings panel. Each theme
              sets CSS custom properties <span className="font-mono">--accent</span> and{" "}
              <span className="font-mono">--accent-secondary</span> as RGB triplets on the document root,
              so all styled components automatically reflect the selected team colour without re-renders.
            </P>
          </Section>

          {/* ── TIRE STRATEGY ────────────────────────────────── */}
          <Section id="tire-strategy" title="Tire Strategy Model">
            <P>
              A dedicated tire strategy model (<span className="font-mono text-[rgb(var(--accent))]">tire_strategy_model.py</span>)
              predicts optimal pit windows and compound choices based on current race conditions. It is
              pre-loaded in a background thread at server startup to avoid latency when the first race
              is initialised.
            </P>
            <Table
              headers={["Factor", "Effect on strategy"]}
              rows={[
                ["Tire compound life",  "SOFT: ~20 laps optimal. MEDIUM: ~30. HARD: ~40+"],
                ["Pit loss time",       "Average 20–24 seconds stationary + in/out lap delta"],
                ["Track temperature",   "Above 45°C accelerates degradation by up to 15%"],
              ]}
            />
            <P>
              Trained models are serialised to JSON in the <span className="font-mono">models/</span> directory
              (<span className="font-mono">compound_model.json</span>,{" "}
              <span className="font-mono">pit_stop_model.json</span>,{" "}
              <span className="font-mono">stops_model.json</span>) so they load instantly without retraining.
            </P>
          </Section>

          {/* ── BATTLE DETECTOR ──────────────────────────────── */}
          <Section id="battle" title="Battle Detector">
            <P>
              The battle detector (<span className="font-mono text-[rgb(var(--accent))]">battle_detector.py</span>)
              monitors the gap between every consecutive pair of drivers each lap. When the gap drops
              below the DRS detection threshold of 1.0 second, a battle is flagged.
            </P>
            <Table
              headers={["Metric", "Calculation"]}
              rows={[
                ["Detection gap",    "< 1.0 second between two consecutive drivers"],
                ["Battle intensity", "Rate of gap change lap-over-lap + consecutive laps within threshold"],
                ["Overtake likely",  "Gap < 0.3 s and closing at > 0.1 s/lap"],
              ]}
            />
            <P>
              Detected battles are passed to the event generator which converts them into narrative
              strings displayed in the notifications panel, for example:{" "}
              <em>"Norris is hunting Russell — gap 0.7 s with 8 laps remaining"</em>.
            </P>
          </Section>

          {/* ── WIKI ─────────────────────────────────────────── */}
          <Section id="wiki" title="Historical Wiki">
            <P>
              The Wiki page queries historical race data from{" "}
              <span className="font-mono text-[rgb(var(--accent))]">unprocessed_f1_training_data.csv</span> through
              the <span className="font-mono">/api/wiki/{"{race_id}/{view}"}</span> endpoint. Three view modes
              are available:
            </P>
            <Table
              headers={["View", "Data shown"]}
              rows={[
                ["Race Result",   "Final classification sorted by finishing position, with race time/status and points"],
                ["Starting Grid", "Grid order with best qualifying time"],
                ["Qualifying",    "Q1, Q2 and Q3 times for all classified drivers"],
              ]}
            />
            <P>
              Circuit information (length, lap record, history) and a Wikipedia thumbnail are fetched
              client-side from the Wikipedia REST API when a race is selected. Driver photos and
              biographical summaries are similarly fetched from Wikipedia on demand when a driver name
              is clicked in the results table.
            </P>
          </Section>

          {/* ── SETUP ────────────────────────────────────────── */}
          <Section id="setup" title="Getting Started">
            <H3>Prerequisites</H3>
            <P>
              Python 3.9+ with conda or a virtual environment, and Node.js 18+ for the frontend. All
              Python dependencies are listed in <span className="font-mono">requirements.txt</span>.
            </P>
            <H3>Install and run</H3>
            <CodeBlock>{`# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start the Flask backend (port 5000)
python app.py

# 3. In a second terminal, install and start the frontend
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173`}</CodeBlock>
            <H3>First race session</H3>
            <P>
              Select a year and Grand Prix in the top selector, then click Initialise. The backend will
              fetch the FastF1 session data — this may take 15–60 seconds on first load; subsequent
              runs use the local cache. Once initialised, press Play to start the simulation.
            </P>
            <H3>Environment variables</H3>
            <Table
              headers={["Variable", "Default", "Description"]}
              rows={[
                ["FLASK_ENV",            "development", "Set to production to enable strict CORS"],
                ["CORS_ALLOWED_ORIGINS", "—",           "Comma-separated list of trusted origins for production"],
              ]}
            />
            <Callout label="Data files">
              The CSV files <span className="font-mono">f1_historical_5years.csv</span> and{" "}
              <span className="font-mono">unprocessed_f1_training_data.csv</span> must be present in the
              project root for the ML model and Wiki page respectively. If the historical file is
              missing, the model falls back to{" "}
              <span className="font-mono">processed_f1_training_data.csv</span>.
            </Callout>
          </Section>

        </main>
      </div>
    </div>
  );
}