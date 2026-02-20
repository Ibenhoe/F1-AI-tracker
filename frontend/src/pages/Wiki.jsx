import { useState, useEffect } from "react";
import { ACCENTS as TEAM_THEMES } from "../providers/ThemeProvider";

// Circuit info keyed by country name (as returned by /api/races/:year)
const CIRCUIT_INFO = {
  "Belgium": {
    name: "Circuit de Spa-Francorchamps",
    location: "Stavelot, Belgium",
    length: "7.004 km",
    laps: 44,
    distance: "308.052 km",
    firstGP: 1950,
    lapRecord: "1:46.286 – Valtteri Bottas (2018)",
    flag: "🇧🇪",
    wikiPage: "Circuit de Spa-Francorchamps",
    history: "One of the longest and most challenging circuits in the world, Spa-Francorchamps winds through the Ardennes Forest in eastern Belgium. Originally a public road circuit first used in 1921, it is famous for the iconic Raidillon/Eau Rouge complex, the ultra-fast Blanchimont corner, and the Bus Stop chicane. Its high altitude and microclimate mean rain can fall at one end of the circuit while the other remains dry — making strategy and tyre calls exceptionally difficult. Michael Schumacher (6×) and Ayrton Senna (5×) are among its legendary winners.",
  },
  "Italy": {
    name: "Autodromo Nazionale Monza",
    location: "Monza, Lombardy, Italy",
    length: "5.793 km",
    laps: 53,
    distance: "306.720 km",
    firstGP: 1950,
    lapRecord: "1:21.046 – Rubens Barrichello (2004)",
    flag: "🇮🇹",
    wikiPage: "Autodromo Nazionale Monza",
    history: "Known as the Temple of Speed, Monza is the fastest circuit on the F1 calendar. Built in 1922 inside a royal park north of Milan, it features long straights (reaching over 340 km/h) and tight chicanes. The passionate Tifosi crowd makes Monza unforgettable. Ferrari's home race has produced dramatic late-race battles; Sebastian Vettel won here as a 21-year-old in 2008 in a wet masterclass.",
  },
  "United Kingdom": {
    name: "Silverstone Circuit",
    location: "Silverstone, Northamptonshire, UK",
    length: "5.891 km",
    laps: 52,
    distance: "306.198 km",
    firstGP: 1950,
    lapRecord: "1:27.097 – Max Verstappen (2020)",
    flag: "🇬🇧",
    wikiPage: "Silverstone Circuit",
    history: "Host of the very first World Championship Grand Prix on 13 May 1950, Silverstone occupies a former WWII bomber base in the English Midlands. Famous high-speed corners like Copse and Maggots-Becketts-Chapel showcase a driver's bravery. It is the spiritual home of British motorsport and has seen legends from Stirling Moss to Lewis Hamilton — who has won here a record eight times.",
  },
  "Monaco": {
    name: "Circuit de Monaco",
    location: "Monte Carlo, Monaco",
    length: "3.337 km",
    laps: 78,
    distance: "260.286 km",
    firstGP: 1950,
    lapRecord: "1:10.166 – Lewis Hamilton (2021)",
    flag: "🇲🇨",
    wikiPage: "Circuit de Monaco",
    history: "The most glamorous race in motorsport, Monaco threads through the streets of the tiny principality. Hairpins at Loews, the tunnel section, the Chicane over the harbour — there is no margin for error as the barriers are millimetres from the car. Overtaking is nearly impossible, making qualifying paramount. Ayrton Senna is the circuit's greatest master with six victories.",
  },
  "Bahrain": {
    name: "Bahrain International Circuit",
    location: "Sakhir, Bahrain",
    length: "5.412 km",
    laps: 57,
    distance: "308.238 km",
    firstGP: 2004,
    lapRecord: "1:31.447 – Pedro de la Rosa (2005)",
    flag: "🇧🇭",
    wikiPage: "Bahrain International Circuit",
    history: "Built on the edge of the desert, Bahrain was the first Middle-Eastern country to host a Formula 1 Grand Prix in 2004. The night race format introduced in 2014 creates stunning visuals under floodlights. Its mix of slow corners and long straights typically generates strategic races with hard tyre battles.",
  },
  "Saudi Arabia": {
    name: "Jeddah Corniche Circuit",
    location: "Jeddah, Saudi Arabia",
    length: "6.174 km",
    laps: 50,
    distance: "308.450 km",
    firstGP: 2021,
    lapRecord: "1:30.734 – Lewis Hamilton (2021)",
    flag: "🇸🇦",
    wikiPage: "Jeddah Corniche Circuit",
    history: "A street circuit hugging the Red Sea coastline, Jeddah debuted in 2021 and is the second-fastest circuit on the calendar. Its walls-lined layout with sweeping blind corners at 300+ km/h makes it one of the most daring tracks in F1. First winner: Lewis Hamilton in a dramatic title-duel race against Max Verstappen.",
  },
  "Australia": {
    name: "Albert Park Circuit",
    location: "Melbourne, Victoria, Australia",
    length: "5.278 km",
    laps: 58,
    distance: "306.124 km",
    firstGP: 1996,
    lapRecord: "1:20.235 – Charles Leclerc (2022)",
    flag: "🇦🇺",
    wikiPage: "Albert Park Circuit",
    history: "The season-opener since 1996, Albert Park winds around a scenic lake in Melbourne. Low-grip tarmac initially makes for unpredictable results. Michael Schumacher and Damon Hill famously collided here in 1994 when it was still in the championship in Adelaide.",
  },
  "Japan": {
    name: "Suzuka International Racing Course",
    location: "Suzuka, Mie Prefecture, Japan",
    length: "5.807 km",
    laps: 53,
    distance: "307.471 km",
    firstGP: 1987,
    lapRecord: "1:30.983 – Lewis Hamilton (2019)",
    flag: "🇯🇵",
    wikiPage: "Suzuka International Racing Course",
    history: "Suzuka is one of the most technically demanding circuits in the world, featuring a unique figure-of-eight layout with the legendary 130R corner and Spoon Curve. It has been the venue for several dramatic title deciders including the Senna-Prost collision in 1989 and Schumacher's championship wins.",
  },
  "China": {
    name: "Shanghai International Circuit",
    location: "Shanghai, China",
    length: "5.451 km",
    laps: 56,
    distance: "305.066 km",
    firstGP: 2004,
    lapRecord: "1:32.238 – Michael Schumacher (2004)",
    flag: "🇨🇳",
    wikiPage: "Shanghai International Circuit",
    history: "Designed by Hermann Tilke, the Shanghai circuit is shaped like the Chinese character shang (上) for Shanghai. Its long back straight and heavy braking zones favour tyre degradation races. The circuit returned to the calendar in 2024 after a COVID-19 absence.",
  },
  "Spain": {
    name: "Circuit de Barcelona-Catalunya",
    location: "Montmeló, Catalonia, Spain",
    length: "4.675 km",
    laps: 66,
    distance: "308.424 km",
    firstGP: 1991,
    lapRecord: "1:16.330 – Max Verstappen (2023)",
    flag: "🇪🇸",
    wikiPage: "Circuit de Barcelona-Catalunya",
    history: "A permanent test venue for most F1 teams, Barcelona is one of the most familiar circuits on the calendar. Teams know the track inside-out which makes the race largely about strategy and car setup. The opening corner Elf/Turn 1 into the tight T2 is a classic overtaking opportunity.",
  },
  "Austria": {
    name: "Red Bull Ring",
    location: "Spielberg, Styria, Austria",
    length: "4.318 km",
    laps: 71,
    distance: "306.452 km",
    firstGP: 1970,
    lapRecord: "1:05.619 – Carlos Sainz (2020)",
    flag: "🇦🇹",
    wikiPage: "Red Bull Ring",
    history: "Nestled in the Styrian mountains, the Red Bull Ring is a compact circuit with long uphill straights and simple-looking but tricky corners. Originally the Österreichring, it was extensively renovated by Red Bull in 2011. Its altitude creates unique engine and cooling scenarios.",
  },
  "Hungary": {
    name: "Hungaroring",
    location: "Mogyoród, Budapest, Hungary",
    length: "4.381 km",
    laps: 70,
    distance: "306.630 km",
    firstGP: 1986,
    lapRecord: "1:16.627 – Lewis Hamilton (2020)",
    flag: "🇭🇺",
    wikiPage: "Hungaroring",
    history: "Eastern Europe's first F1 race behind the Iron Curtain, the Hungaroring is nicknamed Monaco without the walls due to its narrow, twisty layout with almost no real straights. Overtaking is difficult and strategy is everything — tyre management and undercuts often decide the race.",
  },
  "Netherlands": {
    name: "Circuit Zandvoort",
    location: "Zandvoort, North Holland, Netherlands",
    length: "4.259 km",
    laps: 72,
    distance: "306.552 km",
    firstGP: 1952,
    lapRecord: "1:11.097 – Lewis Hamilton (2021)",
    flag: "🇳🇱",
    wikiPage: "Circuit Zandvoort",
    history: "Built among the sand dunes on the Dutch coast, Zandvoort returned to the F1 calendar in 2021 as Max Verstappen's home race. The renovated circuit features heavily banked corners (Hugenholtz Curve at 18 degrees!) that produce exciting wheel-to-wheel racing. The passionate Orange Army of Dutch fans creates an electric atmosphere.",
  },
  "Azerbaijan": {
    name: "Baku City Circuit",
    location: "Baku, Azerbaijan",
    length: "6.003 km",
    laps: 51,
    distance: "306.049 km",
    firstGP: 2017,
    lapRecord: "1:43.009 – Charles Leclerc (2019)",
    flag: "🇦🇿",
    wikiPage: "Baku City Circuit",
    history: "A street circuit weaving through the historic walled city of Baku, featuring a long 2.2 km straight from Turn 16 to Turn 1 — one of the longest in F1. Cars reach over 360 km/h here. The narrow castle section (15th century walls) is only 7.6 m wide. Baku is famous for safety-car interruptions and wild last-lap drama.",
  },
  "Singapore": {
    name: "Marina Bay Street Circuit",
    location: "Marina Bay, Singapore",
    length: "4.940 km",
    laps: 62,
    distance: "306.143 km",
    firstGP: 2008,
    lapRecord: "1:35.867 – Kevin Magnussen (2023)",
    flag: "🇸🇬",
    wikiPage: "Marina Bay Street Circuit",
    history: "F1's first permanent night race held under floodlights, Singapore winds past landmarks including the Marina Bay Sands hotel and the colonial Supreme Court building. Extreme heat and humidity (35°C+, 70% humidity) test driver fitness over 2 hours. The treacherous Turn 3 and the Singapore Sling chicane demand absolute precision.",
  },
  "Canada": {
    name: "Circuit Gilles Villeneuve",
    location: "Île Notre-Dame, Montreal, Canada",
    length: "4.361 km",
    laps: 70,
    distance: "305.270 km",
    firstGP: 1978,
    lapRecord: "1:13.078 – Valtteri Bottas (2019)",
    flag: "🇨🇦",
    wikiPage: "Circuit Gilles Villeneuve",
    history: "Named after the beloved Canadian Ferrari driver Gilles Villeneuve who died in 1982, the circuit sits on an island in the St. Lawrence River. The infamous Wall of Champions at the final chicane has claimed Damon Hill, Michael Schumacher and Jacques Villeneuve. Long straights and heavy braking creates classic overtaking and brake failure drama.",
  },
  "United States": {
    name: "Circuit of the Americas",
    location: "Austin, Texas, USA",
    length: "5.513 km",
    laps: 56,
    distance: "308.405 km",
    firstGP: 2012,
    lapRecord: "1:36.169 – Charles Leclerc (2019)",
    flag: "🇺🇸",
    wikiPage: "Circuit of the Americas",
    history: "The first purpose-built F1 circuit in the USA, COTA was designed with a 133-foot elevation change. Turn 1 is a spectacular blind uphill braking zone. The track features sections inspired by iconic corners from Silverstone, Maggots and the Esses of Suzuka, making it a driver favourite.",
  },
  "Austin": {
    name: "Circuit of the Americas",
    location: "Austin, Texas, USA",
    length: "5.513 km",
    laps: 56,
    distance: "308.405 km",
    firstGP: 2012,
    lapRecord: "1:36.169 – Charles Leclerc (2019)",
    flag: "🇺🇸",
    wikiPage: "Circuit of the Americas",
    history: "The first purpose-built F1 circuit in the USA, COTA was designed with a 133-foot elevation change. Turn 1 is a spectacular blind uphill braking zone. The track features sections inspired by iconic corners from Silverstone and Suzuka, making it a driver favourite.",
  },
  "Mexico": {
    name: "Autodromo Hermanos Rodriguez",
    location: "Mexico City, Mexico",
    length: "4.304 km",
    laps: 71,
    distance: "305.354 km",
    firstGP: 1963,
    lapRecord: "1:17.774 – Valtteri Bottas (2021)",
    flag: "🇲🇽",
    wikiPage: "Autodromo Hermanos Rodriguez",
    history: "Located at 2,240 m above sea level, the thin air of Mexico City significantly reduces downforce and aerodynamic drag — producing top speeds of 370+ km/h on the back straight. The Foro Sol baseball stadium is converted into a grandstand for the final infield section, creating one of the most unique atmospheres in sport.",
  },
  "Brazil": {
    name: "Autodromo Jose Carlos Pace (Interlagos)",
    location: "São Paulo, Brazil",
    length: "4.309 km",
    laps: 71,
    distance: "305.879 km",
    firstGP: 1973,
    lapRecord: "1:10.540 – Valtteri Bottas (2018)",
    flag: "🇧🇷",
    wikiPage: "Autódromo José Carlos Pace",
    history: "Interlagos — between the lakes — is known for its anti-clockwise layout, dramatic elevation changes, and unpredictable weather. It has hosted some of F1's most emotional moments: Ayrton Senna winning in front of his home crowd, and the 2008 title-decider where Lewis Hamilton took the championship on the final corner of the final lap.",
  },
  "Abu Dhabi": {
    name: "Yas Marina Circuit",
    location: "Yas Island, Abu Dhabi, UAE",
    length: "5.281 km",
    laps: 58,
    distance: "306.183 km",
    firstGP: 2009,
    lapRecord: "1:26.103 – Max Verstappen (2021)",
    flag: "🇦🇪",
    wikiPage: "Yas Marina Circuit",
    history: "The season finale venue since 2009, Abu Dhabi hosts the last race under the iconic glowing yas hotel that straddles a section of track. The 2021 finale was one of the most controversial in history, with Max Verstappen overtaking Lewis Hamilton on the last lap to claim his first championship in disputed circumstances.",
  },
  "Miami": {
    name: "Miami International Autodrome",
    location: "Miami Gardens, Florida, USA",
    length: "5.412 km",
    laps: 57,
    distance: "308.326 km",
    firstGP: 2022,
    lapRecord: "1:29.708 – Max Verstappen (2023)",
    flag: "🇺🇸",
    wikiPage: "Miami International Autodrome",
    history: "Built around the Hard Rock Stadium in Miami Gardens, this street-hybrid circuit debuted in 2022 as part of F1's USA expansion. A fake marina, palm trees and the Hard Rock brand give it an American spectacle vibe. In high heat and humidity it produces genuine tyre and strategy challenges.",
  },
  "Las Vegas": {
    name: "Las Vegas Strip Circuit",
    location: "Las Vegas, Nevada, USA",
    length: "6.201 km",
    laps: 50,
    distance: "309.958 km",
    firstGP: 2023,
    lapRecord: "1:35.490 – Oscar Piastri (2023)",
    flag: "🇺🇸",
    wikiPage: "Las Vegas Strip Circuit",
    history: "Racing down the iconic Las Vegas Strip past casinos and hotels, this night race debuted in 2023 as a spectacle event. Cars reach 342 km/h down Las Vegas Boulevard. The cold desert night temperatures create unusual tyre warm-up challenges, and Max Verstappen won the inaugural race.",
  },
  "Qatar": {
    name: "Losail International Circuit",
    location: "Al Daayen, Qatar",
    length: "5.380 km",
    laps: 57,
    distance: "306.811 km",
    firstGP: 2021,
    lapRecord: "1:23.196 – Max Verstappen (2023)",
    flag: "🇶🇦",
    wikiPage: "Lusail International Circuit",
    history: "Originally a MotoGP venue, Losail circuit held its first F1 race in 2021. The flowing high-speed layout and extreme heat (45°C+) cause devastating tyre degradation. In 2023, lap 1 chaos saw multiple penalties and five drivers receive grid penalties for yellow flag ignoring.",
  },
};

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
  const [circuitInfo, setCircuitInfo] = useState(null);
  const [circuitImage, setCircuitImage] = useState(null);
  const [selectedDriver, setSelectedDriver] = useState(null); // { driverRef, fullName, team, row }
  const [driverWiki, setDriverWiki] = useState(null);         // Wikipedia API data
  const [driverWikiLoading, setDriverWikiLoading] = useState(false);

  // Wikipedia titles for known drivers (used to fetch photo + bio)
  const DRIVER_WIKI = {
    "hamilton": "Lewis Hamilton",
    "max_verstappen": "Max Verstappen",
    "verstappen": "Jos Verstappen",
    "leclerc": "Charles Leclerc",
    "norris": "Lando Norris",
    "piastri": "Oscar Piastri",
    "sainz": "Carlos Sainz Jr.",
    "russell": "George Russell (racing driver)",
    "perez": "Sergio Pérez",
    "alonso": "Fernando Alonso",
    "stroll": "Lance Stroll",
    "ocon": "Esteban Ocon",
    "gasly": "Pierre Gasly",
    "tsunoda": "Yuki Tsunoda",
    "bottas": "Valtteri Bottas",
    "zhou": "Zhou Guanyu",
    "hulkenberg": "Nico Hülkenberg",
    "albon": "Alexander Albon",
    "kevin_magnussen": "Kevin Magnussen",
    "ricciardo": "Daniel Ricciardo",
    "vettel": "Sebastian Vettel",
    "schumacher": "Mick Schumacher",
    "michael_schumacher": "Michael Schumacher",
    "raikkonen": "Kimi Räikkönen",
    "button": "Jenson Button",
    "webber": "Mark Webber",
    "massa": "Felipe Massa",
    "barrichello": "Rubens Barrichello",
    "coulthard": "David Coulthard",
    "rosberg": "Nico Rosberg",
    "kubica": "Robert Kubica",
    "grosjean": "Romain Grosjean",
    "kvyat": "Daniil Kvyat",
    "giovinazzi": "Antonio Giovinazzi",
    "latifi": "Nicholas Latifi",
    "lawson": "Liam Lawson",
    "colapinto": "Franco Colapinto",
    "bearman": "Oliver Bearman",
    "antonelli": "Andrea Kimi Antonelli",
    "doohan": "Jack Doohan",
    "bortoleto": "Gabriel Bortoleto",
    "hadjar": "Isack Hadjar",
    "villeneuve": "Jacques Villeneuve",
    "montoya": "Juan Pablo Montoya",
    "heidfeld": "Nick Heidfeld",
  };

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
    "norris": "Lando Norris",
    "piastri": "Oscar Piastri",
    "stroll": "Lance Stroll",
    "ocon": "Esteban Ocon",
    "gasly": "Pierre Gasly",
    "tsunoda": "Yuki Tsunoda",
    "bottas": "Valtteri Bottas",
    "zhou": "Guanyu Zhou",
    "hulkenberg": "Nico Hülkenberg",
    "hülkenberg": "Nico Hülkenberg",
    "albon": "Alexander Albon",
    "sargeant": "Logan Sargeant",
    "de_vries": "Nyck de Vries",
    "lawson": "Liam Lawson",
    "kevin_magnussen": "Kevin Magnussen",
    "ricciardo": "Daniel Ricciardo",
    "drugovich": "Felipe Drugovich",
    "wehrlein": "Pascal Wehrlein",
    "giovinazzi": "Antonio Giovinazzi",
    "schumacher": "Mick Schumacher",
    "mazepin": "Nikita Mazepin",
    "latifi": "Nicholas Latifi",
    "grosjean": "Romain Grosjean",
    "kvyat": "Daniil Kvyat",
    "leclerc_arthur": "Arthur Leclerc",
    "bearman": "Oliver Bearman",
    "colapinto": "Franco Colapinto",
    "antonelli": "Kimi Antonelli",
    "doohan": "Jack Doohan",
    "bortoleto": "Gabriel Bortoleto",
    "hadjar": "Isack Hadjar"
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

  // Normalize historical/variant team names to a canonical label
  const teamNameAliases = {
    "rb f1 team": "Racing Bulls",
    "alphatauri": "Racing Bulls",
    "alpha tauri": "Racing Bulls",
    "scuderia alphatauri": "Racing Bulls",
    "scuderia toro rosso": "Racing Bulls",
    "toro rosso": "Racing Bulls",
    "alfa romeo": "Kick Sauber",
    "alfa romeo racing": "Kick Sauber",
    "sauber": "Kick Sauber",
    "stake f1 team": "Kick Sauber",
    "renault": "Alpine",
    "force india": "Aston Martin",
    "racing point": "Aston Martin",
    "bwt racing point": "Aston Martin",
    "jordan": "Aston Martin",
    "lotus f1 team": "Mercedes",
    "brawn": "Mercedes",
    "red bull": "Red Bull Racing",
    "red bull racing": "Red Bull Racing",
    "mclaren f1 team": "McLaren",
    "scuderia ferrari": "Ferrari",
    "haas f1 team": "Haas",
    "williams racing": "Williams",
  };

  // Helper to get team color
  const getTeamColor = (teamName) => {
    if (!teamName) return "#808080";
    const normalized = teamNameAliases[teamName.toLowerCase()];
    const searchName = normalized || teamName;
    const theme = TEAM_THEMES.find(t =>
      searchName.toLowerCase().includes(t.label.toLowerCase()) ||
      t.label.toLowerCase().includes(searchName.toLowerCase())
    );
    return theme ? theme.primary : "#808080";
  };

  // Fetch Wikipedia image for the circuit
  useEffect(() => {
    if (!circuitInfo?.wikiPage) { setCircuitImage(null); return; }
    setCircuitImage(null);
    fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(circuitInfo.wikiPage)}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => setCircuitImage(data?.thumbnail?.source || null))
      .catch(() => setCircuitImage(null));
  }, [circuitInfo]);

  // Fetch Wikipedia data when a driver is selected
  useEffect(() => {
    if (!selectedDriver) { setDriverWiki(null); return; }
    const wikiTitle = DRIVER_WIKI[selectedDriver.driverRef] || selectedDriver.fullName;
    setDriverWikiLoading(true);
    setDriverWiki(null);
    fetch(`https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(wikiTitle)}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { setDriverWiki(data); setDriverWikiLoading(false); })
      .catch(() => setDriverWikiLoading(false));
  }, [selectedDriver]);

  // Derive circuit info whenever the selected race changes
  useEffect(() => {
    if (!selectedRaceId || races.length === 0) { setCircuitInfo(null); return; }
    const race = races.find(r => String(r.raceId) === String(selectedRaceId));
    if (!race) { setCircuitInfo(null); return; }
    // Try direct country name match, then partial match
    const name = race.name || "";
    const info = CIRCUIT_INFO[name] ||
      Object.entries(CIRCUIT_INFO).find(([k]) =>
        name.toLowerCase().includes(k.toLowerCase()) ||
        k.toLowerCase().includes(name.toLowerCase())
      )?.[1] ||
      null;
    setCircuitInfo(info);
  }, [selectedRaceId, races]);

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
    <>
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

        {/* Circuit Info Card */}
        {circuitInfo && (
          <div className="mb-8 bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-800 rounded-lg shadow-sm overflow-hidden">
            {/* Header */}
            <div className="p-4 border-b border-neutral-200 dark:border-neutral-800">
              <h2 className="text-xl font-bold text-[rgb(var(--accent))]">
                {circuitInfo.flag} {circuitInfo.name}
              </h2>
              <p className="text-sm text-neutral-500 dark:text-neutral-400">{circuitInfo.location}</p>
            </div>

            {/* Body: two-column on md+ */}
            <div className="flex flex-col md:flex-row">
              {/* Left: stats + history */}
              <div className="flex-1 p-5 flex flex-col gap-4">
                {/* Stats row */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {[
                    { label: "Track Length", value: circuitInfo.length },
                    { label: "Race Laps",    value: circuitInfo.laps },
                    { label: "Distance",     value: circuitInfo.distance },
                    { label: "First GP",     value: circuitInfo.firstGP },
                  ].map(({ label, value }) => (
                    <div key={label} className="bg-neutral-50 dark:bg-neutral-800 rounded-lg p-3 text-center">
                      <p className="text-xs uppercase tracking-wide text-neutral-500 dark:text-neutral-400 mb-1">{label}</p>
                      <p className="text-lg font-bold text-neutral-900 dark:text-white">{value}</p>
                    </div>
                  ))}
                </div>
                {/* Lap record */}
                <div className="bg-neutral-50 dark:bg-neutral-800 rounded-lg px-4 py-3">
                  <p className="text-xs uppercase tracking-wide text-neutral-500 dark:text-neutral-400 mb-1">⏱ Lap Record</p>
                  <p className="font-mono text-sm font-semibold text-[rgb(var(--accent))]">{circuitInfo.lapRecord}</p>
                </div>
                {/* History */}
                <div>
                  <p className="text-xs uppercase tracking-wide text-neutral-500 dark:text-neutral-400 mb-2">Circuit History</p>
                  <p className="text-sm text-neutral-700 dark:text-neutral-300 leading-relaxed">{circuitInfo.history}</p>
                </div>
              </div>

              {/* Right: track map image from Wikipedia */}
              <div className="md:w-72 lg:w-80 flex items-center justify-center p-5 border-t md:border-t-0 md:border-l border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-800/50 min-h-40">
                {circuitImage ? (
                  <img
                    src={circuitImage}
                    alt={`${circuitInfo.name} layout`}
                    className="max-w-full max-h-64 object-contain drop-shadow-md rounded"
                  />
                ) : (
                  <span className="text-neutral-400 dark:text-neutral-600 text-sm">Loading map…</span>
                )}
              </div>
            </div>
          </div>
        )}

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
                      <td className="relative px-6 py-4 font-medium text-neutral-700 dark:text-neutral-300">
                        <div 
                          className="absolute left-0 top-0 h-full w-1 rounded-l" 
                          style={{ backgroundColor: getTeamColor(row.team) }}
                        />
                        {row.position}
                      </td>
                      <td
                        className="px-6 py-4 font-bold text-neutral-900 dark:text-neutral-100 cursor-pointer hover:text-[rgb(var(--accent))] transition-colors select-none"
                        onClick={() => setSelectedDriver({ driverRef: row.driver?.toString().toLowerCase(), fullName: formatName(row.driver), team: row.team, row })}
                      >
                        {formatName(row.driver)}
                        <span className="ml-1 text-xs text-neutral-400 dark:text-neutral-500">ℹ</span>
                      </td>
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

    {/* Driver Info Modal */}
    {selectedDriver && (
      <div
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
        onClick={() => setSelectedDriver(null)}
      >
        <div
          className="relative bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-700 rounded-2xl shadow-2xl w-full max-w-md overflow-hidden"
          onClick={e => e.stopPropagation()}
        >
          {/* Team colour accent bar */}
          <div className="h-1.5 w-full" style={{ backgroundColor: getTeamColor(selectedDriver.team) }} />

          {/* Close button */}
          <button
            onClick={() => setSelectedDriver(null)}
            className="absolute top-3 right-3 text-neutral-400 hover:text-neutral-900 dark:hover:text-white text-2xl leading-none"
          >&times;</button>

          <div className="flex gap-5 p-6">
            {/* Photo */}
            <div className="shrink-0 w-28 h-36 rounded-xl overflow-hidden bg-neutral-100 dark:bg-neutral-800 flex items-center justify-center">
              {driverWikiLoading ? (
                <span className="text-neutral-400 text-xs animate-pulse">Loading…</span>
              ) : driverWiki?.thumbnail?.source ? (
                <img
                  src={driverWiki.thumbnail.source}
                  alt={selectedDriver.fullName}
                  className="w-full h-full object-cover object-top"
                />
              ) : (
                <span className="text-5xl">{selectedDriver.fullName.split(' ').slice(-1)[0].charAt(0)}</span>
              )}
            </div>

            {/* Info */}
            <div className="flex-1 min-w-0">
              <h3 className="text-xl font-bold text-neutral-900 dark:text-white leading-tight">{selectedDriver.fullName}</h3>
              <p className="text-sm mt-0.5" style={{ color: getTeamColor(selectedDriver.team) }}>{selectedDriver.team}</p>

              {/* Race result for this session */}
              <div className="mt-3 grid grid-cols-2 gap-2">
                {selectedDriver.row.position != null && (
                  <div className="bg-neutral-100 dark:bg-neutral-800 rounded-lg p-2 text-center">
                    <p className="text-xs text-neutral-500 uppercase tracking-wide">Position</p>
                    <p className="text-lg font-bold text-neutral-900 dark:text-white">P{selectedDriver.row.position}</p>
                  </div>
                )}
                {selectedDriver.row.points > 0 && (
                  <div className="bg-neutral-100 dark:bg-neutral-800 rounded-lg p-2 text-center">
                    <p className="text-xs text-neutral-500 uppercase tracking-wide">Points</p>
                    <p className="text-lg font-bold text-green-500">+{selectedDriver.row.points}</p>
                  </div>
                )}
                {selectedDriver.row.time && (
                  <div className="bg-neutral-100 dark:bg-neutral-800 rounded-lg p-2 text-center col-span-2">
                    <p className="text-xs text-neutral-500 uppercase tracking-wide">Time</p>
                    <p className="text-sm font-mono font-semibold text-neutral-900 dark:text-white">{selectedDriver.row.time}</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Wikipedia extract */}
          {driverWiki?.extract && (
            <div className="px-6 pb-5">
              <p className="text-xs text-neutral-500 uppercase tracking-wide mb-1">Biography</p>
              <p className="text-sm text-neutral-700 dark:text-neutral-300 leading-relaxed line-clamp-4">{driverWiki.extract}</p>
              {driverWiki.content_urls?.desktop?.page && (
                <a
                  href={driverWiki.content_urls.desktop.page}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-2 inline-block text-xs text-[rgb(var(--accent))] hover:underline"
                >
                  Read more on Wikipedia →
                </a>
              )}
            </div>
          )}
        </div>
      </div>
    )}
    </>
  );
}