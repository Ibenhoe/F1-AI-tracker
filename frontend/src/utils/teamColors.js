// Canonical team colors
export const TEAM_COLORS = {
  "Mercedes": "#00D7B6",
  "Red Bull Racing": "#4781D7",
  "Ferrari": "#ED1131",
  "McLaren": "#F47600",
  "Alpine": "#00A1E8",
  "RB": "#6C98FF",
  "Aston Martin": "#229971",
  "Williams": "#1868DB",
  "Kick Sauber": "#01C00E",
  "Haas F1 Team": "#9C9FA2",
};

const TEAM_ALIASES = {
  "red bull": "Red Bull Racing",
  "red bull racing": "Red Bull Racing",
  "red bull racing honda rbpt": "Red Bull Racing",

  "ferrari": "Ferrari",
  "scuderia ferrari": "Ferrari",

  "mercedes": "Mercedes",
  "mercedes-amg": "Mercedes",
  "mercedes-amg petronas": "Mercedes",
  "mercedes-amg petronas f1 team": "Mercedes",

  "mclaren": "McLaren",
  "mclaren f1 team": "McLaren",

  "alpine": "Alpine",
  "bwt alpine f1 team": "Alpine",

  "rb": "RB",
  "racing bulls": "RB",
  "visa cash app rb": "RB",
  "visa cash app rb f1 team": "RB",

  "aston martin": "Aston Martin",
  "aston martin aramco": "Aston Martin",
  "aston martin aramco f1 team": "Aston Martin",

  "williams": "Williams",
  "williams racing": "Williams",

  "sauber": "Kick Sauber",
  "kick sauber": "Kick Sauber",
  "stake": "Kick Sauber",
  "stake f1 team": "Kick Sauber",

  "haas": "Haas F1 Team",
  "haas f1": "Haas F1 Team",
  "haas f1 team": "Haas F1 Team",
};

export function normalizeTeamName(teamName) {
  if (!teamName) return null;
  const key = String(teamName).trim().toLowerCase();
  return TEAM_ALIASES[key] || teamName;
}

export function getTeamColor(teamName) {
  if (!teamName) return null;
  const normalized = normalizeTeamName(teamName);
  return TEAM_COLORS[normalized] ?? null;
}