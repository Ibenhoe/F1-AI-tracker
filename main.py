import pandas as pd
import xgboost as xgb
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# STAP 1: DATA LADEN & FEATURE ENGINEERING
# ---------------------------------------------------------

try:
    # We laden de ruwe data en passen de slimme features hier toe
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "unprocessed_f1_training_data.csv"))
    print(f"Data geladen! Totaal {len(df)} regels ruwe data.")
except FileNotFoundError:
    print("FOUT: Run eerst 'data_script.py' om de data te genereren!")
    exit()

# --- FEATURE ENGINEERING (Dezelfde logica als in je analyse) ---

# 1. Datum & Sorteren (Cruciaal voor ervaring berekening)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# --- HELPER FUNCTIONS ---

def parse_time_str(t_str):
    """Converts '1:30.123' or '90.123' to seconds (float)."""
    if pd.isna(t_str) or str(t_str).strip() == '' or '\\N' in str(t_str): return np.nan
    try:
        t_str = str(t_str).strip()
        if ':' in t_str:
            parts = t_str.split(':')
            return float(parts[0]) * 60 + float(parts[1])
        return float(t_str)
    except:
        return np.nan

def get_team_continuity_id(row):
    name = str(row.get('name_team', '')).lower()
    cid = row['constructorId']
    if 'mercedes' in name or 'brawn' in name or 'honda' in name or 'bar' in name: return 131
    if 'red bull' in name or 'jaguar' in name or 'stewart' in name: return 9
    if 'alpine' in name or 'renault' in name or 'lotus' in name: return 214
    if 'aston martin' in name or 'racing point' in name or 'force india' in name or 'spyker' in name or 'jordan' in name: return 117
    if 'rb' in name or 'alphatauri' in name or 'toro rosso' in name or 'minardi' in name: return 213
    if 'sauber' in name or 'alfa romeo' in name: return 15
    return cid

def calculate_elo_history(df):
    driver_elo = {}      # Huidige rating per coureur
    constructor_elo = {} # Huidige rating per team
    
    # Opslag voor de dataframe
    d_elo_vals = []
    c_elo_vals = []
    
    # Elo parameters
    START_ELO = 1500.0
    K_FACTOR = 20 # Hoe snel verandert de rating?
    
    # We itereren per race (chronologisch)
    for race_id, group in df.groupby('raceId', sort=False):
        # 1. Huidige ratings ophalen (DIT zijn de features voor de voorspelling)
        # Belangrijk: We gebruiken de rating VANAF DE VORIGE RACE. Niet de rating na deze race.
        current_d_elos = [driver_elo.get(d, START_ELO) for d in group['driverId']]
        current_c_elos = [constructor_elo.get(c, START_ELO) for c in group['team_continuity_id']]
        
        # Opslaan in de lijsten (in dezelfde volgorde als de groep rows)
        d_elo_vals.extend(current_d_elos)
        c_elo_vals.extend(current_c_elos)
        
        # 2. Ratings updaten op basis van uitslag
        # We vergelijken elke coureur met het gemiddelde van het veld die dag
        avg_d_elo = np.mean(current_d_elos)
        avg_c_elo = np.mean(current_c_elos)
        
        count = len(group)
        for d_id, c_id, pos in zip(group['driverId'], group['team_continuity_id'], group['positionOrder']):
            # Score: 1.0 voor winnaar, 0.0 voor laatste
            actual_score = (count - pos) / (count - 1) if count > 1 else 0.5
            
            # Expected Score: 1 / (1 + 10^((Opponent - Me)/400))
            d_curr = driver_elo.get(d_id, START_ELO)
            d_exp = 1 / (1 + 10 ** ((avg_d_elo - d_curr) / 400))
            driver_elo[d_id] = d_curr + K_FACTOR * (actual_score - d_exp)
            
            c_curr = constructor_elo.get(c_id, START_ELO)
            c_exp = 1 / (1 + 10 ** ((avg_c_elo - c_curr) / 400))
            constructor_elo[c_id] = c_curr + K_FACTOR * (actual_score - c_exp)
            
    return d_elo_vals, c_elo_vals

# --- APPLY HELPERS TO DATA ---

df['team_continuity_id'] = df.apply(get_team_continuity_id, axis=1)

print("   -> Calculating Elo Ratings (1950-Present)...")

# Elo berekenen op de HELE dataset (dus ook 1950-2014)
# Omdat df gesorteerd is op datum, werkt groupby('raceId', sort=False) chronologisch correct.
d_elos, c_elos = calculate_elo_history(df)

# Omdat groupby de volgorde van rijen binnen een groep kan veranderen t.o.v. origineel dataframe,
# is het veiliger om de waarden direct toe te wijzen via de index van de groepen.
# Echter, in Pandas behoudt groupby standaard de interne volgorde als sort=False.
# We gaan ervan uit dat de lengte matcht.
if len(d_elos) == len(df):
    df['driver_elo'] = d_elos
    df['constructor_elo'] = c_elos
else:
    print("LET OP: Elo berekening lengte mismatch. Feature overgeslagen.")
    df['driver_elo'] = 1500.0
    df['constructor_elo'] = 1500.0

# --- NIEUW: TRACK CHARACTERISTICS (FULL HISTORY) ---
# We gebruiken de hele geschiedenis om te bepalen wat voor type circuit dit is.
# Dit voldoet aan je wens: "keeping data about all the tracks could be usefull".

# 1. Data schoonmaken
if 'fastestLapSpeed' in df.columns:
    df['fastestLapSpeed_num'] = pd.to_numeric(df['fastestLapSpeed'], errors='coerce')
    
    # 2. Circuit Type bepalen (Snelheid)
    # We berekenen de gemiddelde snelheid van de snelste ronde per circuit over alle jaren.
    circuit_stats = df.groupby('circuitId')['fastestLapSpeed_num'].mean().reset_index()
    circuit_stats.rename(columns={'fastestLapSpeed_num': 'circuit_avg_speed'}, inplace=True)
    
    # We verdelen circuits in 3 categorieën op basis van snelheid
    low_threshold = circuit_stats['circuit_avg_speed'].quantile(0.33)
    high_threshold = circuit_stats['circuit_avg_speed'].quantile(0.66)
    
    def get_track_type(speed):
        if pd.isna(speed): return 'Medium' 
        if speed < low_threshold: return 'Twisty' # Monaco, Singapore
        if speed > high_threshold: return 'HighSpeed' # Monza, Spa
        return 'Medium'
    
    circuit_stats['track_type'] = circuit_stats['circuit_avg_speed'].apply(get_track_type)
    df = df.merge(circuit_stats[['circuitId', 'track_type']], on='circuitId', how='left')

    # --- NIEUW: EXTRA TRACK FEATURES (Length, Street, Altitude) ---
    # 1. Circuit Lengte (Schatting: Snelheid * Tijd)
    # Speed = km/h, Time = sec. Length = (Speed * Time/3600)
    # We nemen het gemiddelde per circuit om ruis weg te filteren
    if 'fastest_lap_seconds' not in df.columns:
        # Even tijdelijk parsen als het nog niet gebeurd is (volgorde dingetje)
        df['temp_seconds'] = pd.to_numeric(df['fastestLapTime'].str.replace(r'.*:', '', regex=True), errors='coerce')
    else:
        df['temp_seconds'] = df['fastest_lap_seconds']
        
    df['calc_length'] = (df['fastestLapSpeed_num'] * df['temp_seconds']) / 3600
    circuit_len = df.groupby('circuitId')['calc_length'].mean().rename('circuit_length_km')
    df = df.merge(circuit_len, on='circuitId', how='left')
    
    # 2. Street Circuit Flag (Handmatige lijst van bekende straatcircuits)
    # Monaco(6), Baku(73), Singapore(15), Jeddah(77), Miami(79), Las Vegas(80), Albert Park(1), Canada(7)
    street_circuits = [6, 73, 15, 77, 79, 80, 1, 7] 
    df['is_street_circuit'] = df['circuitId'].isin(street_circuits).astype(int)
    
    # 3. Altitude (Zit al in df['alt'], maar we zorgen dat het een getal is)
    df['alt'] = pd.to_numeric(df['alt'], errors='coerce').fillna(0)
    
else:
    df['track_type'] = 'Medium'
    df['circuit_length_km'] = 5.0
    df['is_street_circuit'] = 0
    df['alt'] = 0

# --- FILTER: NU PAS FILTEREN OP 2015+ ---
df = df[df['year'] >= 2015].copy()
print(f"Data gefilterd op seizoen 2015+. Totaal {len(df)} regels over voor training.")

# 2. Driver Experience (Aantal races gereden vóór deze race)
df['driver_experience'] = df.groupby('driverId').cumcount()

# 3. Home Race (Thuisvoordeel)
# Fallback: Als 'country' mist (oude data script), laad het uit circuits.csv
if 'country' not in df.columns:
    circuits = pd.read_csv('../F1_data_mangement/circuits.csv')
    df = pd.merge(df, circuits[['circuitId', 'country']], on='circuitId', how='left')

nationality_map = {
    'British': 'UK', 'German': 'Germany', 'Spanish': 'Spain', 'French': 'France',
    'Italian': 'Italy', 'Dutch': 'Netherlands', 'Australian': 'Australia', 
    'Monegasque': 'Monaco', 'American': 'USA', 'Japanese': 'Japan', 'Canadian': 'Canada',
    'Mexican': 'Mexico', 'Brazilian': 'Brazil'
}
df['mapped_nationality'] = df['nationality'].map(nationality_map).fillna(df['nationality'])
df['is_home_race'] = np.where(df['mapped_nationality'] == df['country'], 1, 0)

# 4. Binning (Categorieën maken)
df['grid_bin'] = pd.cut(df['grid'], bins=[-1, 1, 3, 10, 15, 25], labels=['Pole', 'Top3', 'Points', 'Midfield', 'Back'])
df['age_bin'] = pd.cut(df['driver_age'], bins=[17, 24, 30, 36, 60], labels=['Rookie', 'Prime', 'Experienced', 'Veteran'])

# 5. Encoding (Tekst naar getallen voor XGBoost)
grid_mapping = {'Pole': 0, 'Top3': 1, 'Points': 2, 'Midfield': 3, 'Back': 4}
age_mapping = {'Rookie': 0, 'Prime': 1, 'Experienced': 2, 'Veteran': 3}

df['grid_bin_code'] = df['grid_bin'].map(grid_mapping).fillna(4).astype(int)
df['age_bin_code'] = df['age_bin'].map(age_mapping).fillna(1).astype(int)

# --- NIEUW: OVERTAKING DIFFICULTY (CIRCUIT) ---
# We berekenen hoeveel posities er gemiddeld veranderen op een circuit.
# Veel verandering = makkelijk inhalen (of chaos). Weinig = Monaco.
df['pos_change_abs'] = (df['grid'] - df['positionOrder']).abs()
df['circuit_overtake_index'] = df.groupby('circuitId')['pos_change_abs'].transform(lambda x: x.expanding().mean().shift(1))
# Vul eventuele gaten (eerste race op circuit)
df['circuit_overtake_index'] = df['circuit_overtake_index'].fillna(df['pos_change_abs'].mean())

# 6. RECENT FORM (De belangrijkste toevoeging!)
# We berekenen het gemiddelde van de laatste 5 races voor coureur en team.
# Dit vertelt het model wie er NU in vorm is, in plaats van wie er 10 jaar geleden goed was.

def calculate_rolling_avg(group, window=5):
    # shift(1) zorgt dat we de uitslag van de huidige race niet gebruiken
    # We gebruiken een gewogen gemiddelde: recentere races tellen zwaarder (lineair: 1, 2, 3, 4, 5)
    def weighted_mean(x):
        weights = np.arange(1, len(x) + 1)
        return np.average(x, weights=weights)
        
    return group.shift(1).rolling(window, min_periods=1).apply(weighted_mean, raw=True)

df['driver_recent_form'] = df.groupby('driverId')['positionOrder'].transform(calculate_rolling_avg)
df['constructor_recent_form'] = df.groupby('team_continuity_id')['positionOrder'].transform(calculate_rolling_avg)

# Vul lege waarden (eerste races van seizoen) met een gemiddelde (bijv. P12)
df['driver_recent_form'] = df['driver_recent_form'].fillna(12.0)
df['constructor_recent_form'] = df['constructor_recent_form'].fillna(12.0)

# --- NIEUW: PITSTOP FEATURES ---
# We berekenen hoe snel een coureur/team gemiddeld is in de pitstraat (recente vorm)
if 'pit_stops_duration_ms' in df.columns and 'pit_stops_count' in df.columns:
    # Vul nulls (geen stops = 0)
    df['pit_stops_count'] = df['pit_stops_count'].fillna(0)
    df['pit_stops_duration_ms'] = df['pit_stops_duration_ms'].fillna(0)
    
    # Bereken gemiddelde tijd per stop in die race (voorkom delen door 0)
    df['avg_pit_duration'] = df.apply(lambda x: x['pit_stops_duration_ms'] / x['pit_stops_count'] if x['pit_stops_count'] > 0 else np.nan, axis=1)
    
    # Globaal gemiddelde (fallback voor als er geen data is, ca. 24000ms)
    global_pit_mean = df['avg_pit_duration'].mean()
    if pd.isna(global_pit_mean): global_pit_mean = 24000.0
    
    # Vul races zonder stops met het gemiddelde (zodat rolling avg niet breekt)
    df['avg_pit_duration_filled'] = df['avg_pit_duration'].fillna(global_pit_mean)

    # Rolling average berekenen (laatste 5 races)
    df['driver_recent_pit_avg'] = df.groupby('driverId')['avg_pit_duration_filled'].transform(calculate_rolling_avg)
    df['constructor_recent_pit_avg'] = df.groupby('team_continuity_id')['avg_pit_duration_filled'].transform(calculate_rolling_avg)
    
    # Eerste races vullen met fallback
    df['driver_recent_pit_avg'] = df['driver_recent_pit_avg'].fillna(global_pit_mean)
    df['constructor_recent_pit_avg'] = df['constructor_recent_pit_avg'].fillna(global_pit_mean)
else:
    print("LET OP: Geen pitstop data gevonden. Features worden standaardwaarden.")
    global_pit_mean = 24000.0
    df['driver_recent_pit_avg'] = global_pit_mean
    df['constructor_recent_pit_avg'] = global_pit_mean

# --- NIEUW: TIRE STRATEGY FEATURES ---
# 1. Circuit Degradation (Hoeveel stops zijn normaal op dit circuit?)
# We gebruiken expanding mean om data leakage te voorkomen.
df['circuit_avg_stops'] = df.groupby('circuitId')['pit_stops_count'].transform(lambda x: x.expanding().mean().shift(1))
df['circuit_avg_stops'] = df['circuit_avg_stops'].fillna(1.5) # Default 1-2 stops

# 2. Tire Strategy Proxy (Soft vs Hard Preference)
# We hebben geen letterlijke 'Soft/Medium/Hard' data, maar wel stint lengtes.
# Ratio > 1.0 = Langere stints dan gemiddeld (Harde band / Bandenspaarder).
# Ratio < 1.0 = Kortere stints dan gemiddeld (Zachte band / Agressief).

race_stint_avg = df.groupby('raceId')['avg_stint_length'].transform('mean')
df['tire_strategy_ratio'] = df['avg_stint_length'] / race_stint_avg
df['tire_strategy_ratio'] = df['tire_strategy_ratio'].fillna(1.0)

# We berekenen de 'Style' van de coureur: Kiest hij vaak voor lange stints?
df['driver_tire_strategy'] = df.groupby('driverId')['tire_strategy_ratio'].transform(calculate_rolling_avg)
df['driver_tire_strategy'] = df['driver_tire_strategy'].fillna(1.0)

# --- NIEUW: RELIABILITY (DNF KANS) ---
# We berekenen hoe vaak een coureur of auto de finish NIET haalt.
# Status 1 = Finished. Status 11, 12 etc = +1 Lap (ook gefinisht).
# We zoeken naar statussen die NIET 'Finished' of '+X Laps' zijn.
if 'status' in df.columns:
    # Regex: Matcht NIET met "Finished" of "+... Laps"
    df['is_dnf'] = ~df['status'].astype(str).str.match(r'(Finished|\+\d+\sLaps)').fillna(False)
    df['is_dnf'] = df['is_dnf'].astype(int)
    
    # Rolling average van DNF's (0.0 = altijd finish, 1.0 = altijd crash/pech)
    df['driver_dnf_rate'] = df.groupby('driverId')['is_dnf'].transform(calculate_rolling_avg)
    df['constructor_dnf_rate'] = df.groupby('team_continuity_id')['is_dnf'].transform(calculate_rolling_avg)
    
    df['driver_dnf_rate'] = df['driver_dnf_rate'].fillna(0.2) # Default 20% kans
    df['constructor_dnf_rate'] = df['constructor_dnf_rate'].fillna(0.2)
else:
    df['driver_dnf_rate'] = 0.2
    df['constructor_dnf_rate'] = 0.2

# --- NIEUW: QUALIFYING PACE (RELATIEF AAN POLE) ---
# Grid positie is "ordinaal" (1, 2, 3). Tijdverschil is "ratio" (0.1s, 0.5s).
# Dat laatste bevat veel meer informatie over de ware snelheid van de auto.

# 1. Parse de tijden (q1, q2, q3)
for col in ['q1', 'q2', 'q3']:
    if col in df.columns:
        df[f'{col}_sec'] = df[col].apply(parse_time_str)

# 2. Beste tijd per coureur bepalen (sommige halen Q3 niet)
df['best_quali_time'] = df[['q1_sec', 'q2_sec', 'q3_sec']].min(axis=1)

# 3. Pole tijd per race bepalen
pole_times = df.groupby('raceId')['best_quali_time'].min().rename('pole_time')
df = df.merge(pole_times, on='raceId', how='left')

# 4. Het percentage verschil berekenen (1.00 = Pole, 1.01 = 1% langzamer)
df['quali_pace_deficit'] = df['best_quali_time'] / df['pole_time']
df['quali_pace_deficit'] = df['quali_pace_deficit'].fillna(1.07) # 107% regel als fallback

# --- NIEUW: TEAMMATE BATTLE (SKILL ISOLATOR) ---
# Presteert deze coureur beter of slechter dan zijn teamgenoot in recente vorm?
# Dit filtert de "auto factor" eruit.
df['teammate_form_diff'] = df.groupby(['raceId', 'team_continuity_id'])['driver_recent_form'].transform(lambda x: x - x.mean())

# --- NIEUW: DRIVER STYLE AFFINITY (TRACK TYPE SPECIALIST) ---
# Is deze coureur een specialist op dit type circuit (Twisty vs HighSpeed)?
# We vergelijken hun vorm op dit TYPE circuit met hun algemene vorm.

# 1. Bereken vorm per track type (Expanding mean om toekomstkennis te voorkomen)
df['driver_type_form'] = df.groupby(['driverId', 'track_type'])['positionOrder'].transform(
    lambda x: x.shift(1).expanding().mean()
)
df['driver_type_form'] = df['driver_type_form'].fillna(12.0)

# 2. Bereken de "Style Bonus" (Positief = Beter op dit type baan dan gemiddeld)
df['driver_style_affinity'] = df['driver_recent_form'] - df['driver_type_form']

# --- WEERDATA VOORBEREIDEN ---
# We vullen ontbrekende weerdata op met logische standaardwaarden
if 'weather_temp_c' in df.columns:
    # UPDATE: Median is robuuster tegen uitschieters (smoother) dan Mean
    df['weather_temp_c'] = df['weather_temp_c'].fillna(df['weather_temp_c'].median())
    df['weather_precip_mm'] = df['weather_precip_mm'].fillna(0.0) # Geen regen
    df['weather_cloud_pct'] = df['weather_cloud_pct'].fillna(50.0) # Half bewolkt
else:
    # Fallback als kolommen nog niet bestaan (voor de zekerheid)
    df['weather_temp_c'] = 20.0
    df['weather_precip_mm'] = 0.0
    df['weather_cloud_pct'] = 50.0

# 7. TRACK HISTORY (Historie op dit specifieke circuit)
# We kijken naar hoe de coureur het in het verleden op DIT circuit heeft gedaan.

def calculate_expanding_mean(group):
    # shift(1) om huidige race niet mee te tellen (data leakage voorkomen)
    return group.shift(1).expanding().mean()

def calculate_expanding_sum(group):
    return group.shift(1).expanding().sum()

# Groepeer per coureur en circuit
track_groups = df.groupby(['driverId', 'circuitId'])

df['driver_track_avg_grid'] = track_groups['grid'].transform(calculate_expanding_mean)
df['driver_track_avg_finish'] = track_groups['positionOrder'].transform(calculate_expanding_mean)

# Podiums berekenen (Top 3)
df['is_podium'] = (df['positionOrder'] <= 3).astype(int)
df['driver_track_podiums'] = track_groups['is_podium'].transform(calculate_expanding_sum)

# Vul lege waarden (eerste keer op circuit) met neutrale waarden
df['driver_track_avg_grid'] = df['driver_track_avg_grid'].fillna(12.0)
df['driver_track_avg_finish'] = df['driver_track_avg_finish'].fillna(12.0)
df['driver_track_podiums'] = df['driver_track_podiums'].fillna(0.0)

# --- NIEUW: AGRESSIVITEIT & PUNTEN STAND ---
# We kijken of coureurs sneller worden (agressiviteit/skill) en hoeveel punten ze hebben (veilig vs risico).

if 'fastestLapTime' in df.columns:
    # 1. Zet tijd om naar seconden
    df['fastest_lap_seconds'] = df['fastestLapTime'].apply(parse_time_str)
    
    # 2. Normaliseren: Hoe snel t.o.v. de snelste raceronde die dag?
    # Dit maakt tijden vergelijkbaar tussen circuits (Monaco vs Spa)
    race_best_times = df.groupby('raceId')['fastest_lap_seconds'].min().reset_index().rename(columns={'fastest_lap_seconds': 'race_best_time'})
    df = pd.merge(df, race_best_times, on='raceId', how='left')
    
    # Ratio: 1.00 = Snelste van de dag. 1.05 = 5% langzamer.
    df['speed_ratio'] = df['fastest_lap_seconds'] / df['race_best_time']
    
    # Vul lege waarden (geen tijd gezet) met een trage ratio
    df['speed_ratio'] = df['speed_ratio'].fillna(1.10)
    
    # 3. Recente Snelheid: Worden ze sneller in de laatste 5 races?
    df['driver_recent_speed'] = df.groupby('driverId')['speed_ratio'].transform(calculate_rolling_avg)
    df['driver_recent_speed'] = df['driver_recent_speed'].fillna(1.10)
    
    # --- NIEUW: CIRCUIT KARAKTERISTIEKEN ---
    # We berekenen de gemiddelde rondetijd van een circuit over de historie.
    # FIX: Gebruik expanding().mean() om data leakage te voorkomen (geen toekomstkennis)
    # We groeperen per circuit en kijken alleen naar voorgaande races op dat circuit
    df['circuit_speed_index'] = df.groupby('circuitId')['fastest_lap_seconds'].transform(lambda x: x.expanding().mean().shift(1))
    
    # Vul missende waarden met het globale gemiddelde
    global_lap_mean = df['fastest_lap_seconds'].mean()
    df['circuit_speed_index'] = df['circuit_speed_index'].fillna(global_lap_mean)

    # --- NIEUW: AGGRESSION INDEX (Snelheid vs Resultaat) ---
    # Als je de snelste auto hebt (Rank 1) maar P10 finisht, ben je agressief/foutgevoelig.
    # Stap 1: Hoe snel was je in de race t.o.v. de rest? (Rank 1 = Snelste)
    df['fastest_lap_rank'] = df.groupby('raceId')['fastest_lap_seconds'].rank(method='min')
    
    # Stap 2: Verschil tussen snelheid en finish.
    # Positief getal (bv. Finish 15 - SpeedRank 1 = +14) = Veel snelheid, slecht resultaat (Agressief/Pech)
    # Negatief getal (bv. Finish 5 - SpeedRank 15 = -10) = Weinig snelheid, goed resultaat (Consistent/Geluk)
    df['aggression_score'] = df['positionOrder'] - df['fastest_lap_rank']
    
    # Stap 3: Dit is een eigenschap van de coureur, dus we nemen het gemiddelde van de laatste 5 races
    df['driver_aggression'] = df.groupby('driverId')['aggression_score'].transform(calculate_rolling_avg)
    df['driver_aggression'] = df['driver_aggression'].fillna(0.0)

    # --- NIEUW: CONSISTENCY (BATTLE DETECTOR) ---
    # We gebruiken de standaardafwijking van rondetijden (std_lap_time_ms).
    # Laag = Consistent (Vrije lucht). Hoog = Gevecht/Verkeer/Fouten.
    if 'std_lap_time_ms' in df.columns:
        df['driver_consistency'] = df.groupby('driverId')['std_lap_time_ms'].transform(calculate_rolling_avg)
        df['driver_consistency'] = df['driver_consistency'].fillna(5000.0) # Default onrustig
    else:
        df['driver_consistency'] = 5000.0
else:
    df['driver_recent_speed'] = 1.10
    df['circuit_speed_index'] = 90.0 # Default 1:30
    df['driver_consistency'] = 5000.0

# 4. Puntenstand voorafgaand aan de race (Strategie context)
# We berekenen de cumulatieve som min de punten van vandaag = punten bij start.
df['points_before_race'] = (df.groupby(['year', 'driverId'])['points'].cumsum() - df['points']).fillna(0)

# --- NIEUW: KWALIFICATIE DATA ---
# Gebruik quali_position als die bestaat, anders grid.
if 'quali_position' in df.columns:
    df['quali_pos_filled'] = df['quali_position'].fillna(df['grid'])
else:
    df['quali_pos_filled'] = df['grid']

# --- NIEUW: GRID PENALTY ---
# Verschil tussen Quali en Grid. (Positief = straf, Negatief = winst door andermans straf)
df['grid_penalty'] = df['grid'] - df['quali_pos_filled']

# --- NIEUW: WEATHER CONFIDENCE (Nat vs Droog Talent) ---
# We berekenen de gemiddelde finish in natte races apart van droge races.

# 1. Definieer wat 'Nat' is (meer dan 0.1mm regen)
df['is_wet_race'] = df['weather_precip_mm'] > 0.1

# 2. Bereken historisch gemiddelde voor NAT en DROOG (Expanding mean om toekomstkennis te voorkomen)
# We gebruiken transform zodat de index gelijk blijft aan de originele df
wet_avg = df[df['is_wet_race']].groupby('driverId')['positionOrder'].transform(lambda x: x.expanding().mean().shift(1))
dry_avg = df[~df['is_wet_race']].groupby('driverId')['positionOrder'].transform(lambda x: x.expanding().mean().shift(1))

# We zetten deze waarden terug in de hoofdtabel (dit is even puzzelen met indexen)
df['avg_finish_wet'] = wet_avg
df['avg_finish_dry'] = dry_avg

# Vul lege waarden (als je nog nooit in regen reed, is je regen-skill gelijk aan je droog-skill)
df['avg_finish_dry'] = df['avg_finish_dry'].fillna(12.0)
df['avg_finish_wet'] = df['avg_finish_wet'].fillna(df['avg_finish_dry'])

# 3. Het verschil: Negatief betekent BETER in regen (lagere positie is beter)
df['weather_confidence_diff'] = df['avg_finish_wet'] - df['avg_finish_dry']

# --- NIEUW: OVERTAKE RATE (INHAAL SKILL) ---
# Hoeveel plekken wint deze coureur gemiddeld? (Grid - Finish)
df['positions_gained'] = df['grid'] - df['positionOrder']
df['driver_overtake_rate'] = df.groupby('driverId')['positions_gained'].transform(calculate_rolling_avg).fillna(0.0)

# --- DEFINIEER FEATURES ---
feature_cols = [
    'grid', 
    'grid_penalty',
    'circuitId', 
    'circuit_speed_index',
    'circuit_overtake_index',
    'driver_age',
    'driver_experience',
    'is_home_race',
    'grid_bin_code',
    'age_bin_code',
    'driver_recent_form',
    'constructor_recent_form',
    'driver_track_avg_grid',
    'driver_track_avg_finish',
    'driver_track_podiums',
    'weather_temp_c',
    'weather_precip_mm',
    'weather_cloud_pct',
    'weather_confidence_diff', # <--- Nieuw: Is coureur beter in regen?
    'driver_aggression',       # <--- Nieuw: Rijdt coureur sneller dan zijn finish?
    'driver_overtake_rate',    # <--- Nieuw: Inhaalmachine?
    'driver_consistency',      # <--- Nieuw: Stabiele rondetijden?
    'driver_recent_pit_avg',
    'constructor_recent_pit_avg',
    'driver_dnf_rate',
    'constructor_dnf_rate',
    'circuit_avg_stops',      # <--- Nieuw
    'driver_tire_strategy',   # <--- Nieuw: Proxy voor Soft/Hard keuze
    'driver_recent_speed',
    'points_before_race',
    'quali_pace_deficit',     # <--- Nieuw: Echte snelheid
    'teammate_form_diff',     # <--- Nieuw: Skill vs Auto
    'driver_elo',             # <--- NIEUW: True Skill
    'constructor_elo',        # <--- NIEUW: True Car Performance
    'driver_style_affinity',  # <--- NIEUW: Rijstijl
    'circuit_length_km',      # <--- NIEUW: Verkeersdrukte
    'is_street_circuit',      # <--- NIEUW: Straatvechters
    'alt'                     # <--- NIEUW: Hoogte (Motor vermogen)
]

# Filter data: Alleen rijen met een geldig resultaat
train_df = df.dropna(subset=['positionOrder'] + feature_cols)

X = train_df[feature_cols]
y = train_df['positionOrder']

# ---------------------------------------------------------
# STAP 2: VOORSPELLING DATA VOORBEREIDEN (SPA 2024)
# ---------------------------------------------------------
# We bereiden EERST de input voor de komende race voor, zodat we die direct in de loop kunnen gebruiken.
print("\nInput data voorbereiden voor Spa 2024...")

# Hulpfunctie: Haal de huidige ervaring van een coureur op uit de historie
def get_experience(driver_id, df=df):
    if driver_id in df['driverId'].values:
        return df[df['driverId'] == driver_id]['driver_experience'].max() + 1
    return 0 # Nieuwe coureur

# Hulpfunctie: Haal de laatste Elo op
def get_latest_elo(id_col, id_val, target_col='driver_elo', df=df):
    # Pak de allerlaatste race van deze entiteit
    last_val = df[df[id_col] == id_val].sort_values(by='date').iloc[-1][target_col]
    return last_val if not pd.isna(last_val) else 1500.0

# Hulpfunctie: Haal de recente vorm op (gemiddelde laatste 5 races) uit de historie
def get_recent_form(id_col, id_val, target_col='positionOrder', window=5, default_val=12.0, df=df):
    # Pak de laatste N races van deze coureur/team uit de dataset
    history = df[df[id_col] == id_val].sort_values(by='date').tail(window)
    if len(history) == 0:
        return default_val # Geen historie? Dan gokken we de default
    
    # Gewogen gemiddelde berekenen voor de voorspelling
    values = history[target_col].values
    weights = np.arange(1, len(values) + 1)
    return np.average(values, weights=weights)

# Hulpfunctie: Haal de historie op dit circuit op
def get_track_history(driver_id, circuit_id, df=df):
    history = df[(df['driverId'] == driver_id) & (df['circuitId'] == circuit_id)]
    if len(history) == 0:
        return 12.0, 12.0, 0.0 # Default waarden
    
    avg_grid = history['grid'].mean()
    avg_finish = history['positionOrder'].mean()
    podiums = len(history[history['positionOrder'] <= 3])
    return avg_grid, avg_finish, podiums

# Hulpfunctie: Haal de track type affinity op
def get_style_affinity(driver_id, track_type, df=df):
    # Haal alle races van deze coureur op dit type circuit
    type_history = df[(df['driverId'] == driver_id) & (df['track_type'] == track_type)]
    if len(type_history) == 0: return 0.0
    
    avg_pos_on_type = type_history['positionOrder'].mean()
    recent_form = get_recent_form('driverId', driver_id)
    return recent_form - avg_pos_on_type # Positief = Beter op dit type

upcoming_race_dict = {
    'driver_name':   ['Leclerc', 'Perez', 'Hamilton', 'Norris', 'Piastri', 'Russell', 'Sainz', 'Alonso', 'Ocon', 'Albon', 'Verstappen', 'Gasly', 'Ricciardo', 'Bottas', 'Stroll', 'Hulkenberg', 'Magnussen', 'Sargeant', 'Guanyu', 'Tsunoda'],
    'grid':          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'driverId':      [844, 815, 1, 846, 857, 847, 832, 4, 839, 848, 830, 842, 817, 822, 840, 807, 825, 858, 855, 852],
    'constructorId': [6, 9, 131, 1, 1, 131, 6, 117, 214, 3, 9, 214, 213, 15, 117, 210, 210, 3, 15, 213],
    'circuitId':     [13] * 20,
    'year':          [2024] * 20,
    'driver_age':    [26, 34, 39, 24, 23, 26, 29, 43, 27, 28, 26, 28, 35, 34, 25, 37, 31, 23, 25, 24],
    'nationality':   ['Monegasque', 'Mexican', 'British', 'British', 'Australian', 'British', 'Spanish', 'Spanish', 'French', 'Thai', 'Dutch', 'French', 'Australian', 'Finnish', 'Canadian', 'German', 'Danish', 'American', 'Chinese', 'Japanese'],
    'country':       ['Belgium'] * 20,
    # Weerbericht voor Spa (voorbeeld: 18 graden, lichte regen)
    'weather_temp_c':    [18.0] * 20,
    'weather_precip_mm': [0.5] * 20,
    'weather_cloud_pct': [80.0] * 20,
    'alt':               [400] * 20 # Spa ligt op ~400m hoogte
}

# Voeg teamnamen toe voor mapping (fictief, maar nodig voor de functie)
upcoming_race_dict['name_team'] = ['Ferrari', 'Red Bull', 'Mercedes', 'McLaren', 'McLaren', 'Mercedes', 'Ferrari', 'Aston Martin', 'Alpine', 'Williams', 'Red Bull', 'Alpine', 'RB', 'Sauber', 'Aston Martin', 'Haas', 'Haas', 'Williams', 'Sauber', 'RB']

# Voeg quali info toe (fictief voor Spa 2024, we nemen grid over)
upcoming_race_dict['quali_position'] = upcoming_race_dict['grid']

X_next = pd.DataFrame(upcoming_race_dict)

# Feature Engineering toepassen op de nieuwe data
X_next['driver_experience'] = X_next['driverId'].apply(get_experience)

# Home Race logic
X_next['mapped_nationality'] = X_next['nationality'].map(nationality_map).fillna(X_next['nationality'])
X_next['is_home_race'] = np.where(X_next['mapped_nationality'] == X_next['country'], 1, 0)

# Bins & Encoding (Gebruik dezelfde encoders als bij training!)
X_next['grid_bin'] = pd.cut(X_next['grid'], bins=[-1, 1, 3, 10, 15, 25], labels=['Pole', 'Top3', 'Points', 'Midfield', 'Back'])
X_next['age_bin'] = pd.cut(X_next['driver_age'], bins=[17, 24, 30, 36, 60], labels=['Rookie', 'Prime', 'Experienced', 'Veteran'])

X_next['grid_bin_code'] = X_next['grid_bin'].map(grid_mapping).fillna(4).astype(int)
X_next['age_bin_code'] = X_next['age_bin'].map(age_mapping).fillna(1).astype(int)

# Team Mapping toepassen op voorspelling
X_next['team_continuity_id'] = X_next.apply(get_team_continuity_id, axis=1)

# Elo Ratings ophalen (De belangrijkste stap!)
X_next['driver_elo'] = X_next['driverId'].apply(lambda x: get_latest_elo('driverId', x, 'driver_elo'))
X_next['constructor_elo'] = X_next['team_continuity_id'].apply(lambda x: get_latest_elo('team_continuity_id', x, 'constructor_elo'))

# Recent Form toevoegen aan de voorspelling
X_next['driver_recent_form'] = X_next['driverId'].apply(lambda x: get_recent_form('driverId', x, default_val=12.0))
# Let op: we gebruiken nu team_continuity_id voor de constructor form
X_next['constructor_recent_form'] = X_next['team_continuity_id'].apply(lambda x: get_recent_form('team_continuity_id', x, default_val=12.0))

# Pitstop Form toevoegen (gebruikt de 'avg_pit_duration_filled' kolom uit de historie)
X_next['driver_recent_pit_avg'] = X_next['driverId'].apply(lambda x: get_recent_form('driverId', x, target_col='avg_pit_duration_filled', default_val=global_pit_mean))
X_next['constructor_recent_pit_avg'] = X_next['team_continuity_id'].apply(lambda x: get_recent_form('team_continuity_id', x, target_col='avg_pit_duration_filled', default_val=global_pit_mean))

# Tire Strategy Features voor voorspelling
# 1. Circuit Avg Stops (Spa = Circuit 13)
spa_avg_stops = df[df['circuitId'] == 13]['circuit_avg_stops'].mean()
X_next['circuit_avg_stops'] = spa_avg_stops if not pd.isna(spa_avg_stops) else 2.0
# 2. Driver Tire Strategy (Historie van stint ratio)
X_next['driver_tire_strategy'] = X_next['driverId'].apply(lambda x: get_recent_form('driverId', x, target_col='tire_strategy_ratio', default_val=1.0))

# Track History toevoegen
track_stats = X_next.apply(lambda x: get_track_history(x['driverId'], x['circuitId']), axis=1)
X_next['driver_track_avg_grid'] = [x[0] for x in track_stats]
X_next['driver_track_avg_finish'] = [x[1] for x in track_stats]
X_next['driver_track_podiums'] = [x[2] for x in track_stats]

# Recente Podiums (Vormpiek)
# We berekenen hoe vaak ze in de laatste 5 races podium haalden (Top 3)
X_next['recent_podium_rate'] = X_next['driverId'].apply(lambda x: get_recent_form('driverId', x, target_col='is_podium', default_val=0.0))

# Nieuwe features toevoegen aan voorspelling
# 1. Recente snelheid (gebruikt de historie van fastest laps)
X_next['driver_recent_speed'] = X_next['driverId'].apply(lambda x: get_recent_form('driverId', x, target_col='speed_ratio', default_val=1.10))

# 2. Circuit Speed Index (Spa is bekend in data, dus we pakken het gemiddelde van circuitId 13)
spa_avg_time = df[df['circuitId'] == 13]['fastest_lap_seconds'].mean()
if pd.isna(spa_avg_time): spa_avg_time = 105.0 # Fallback
X_next['circuit_speed_index'] = spa_avg_time

# 3. Circuit Overtake Index (Spa is bekend)
spa_overtake = df[df['circuitId'] == 13]['circuit_overtake_index'].mean()
if pd.isna(spa_overtake): spa_overtake = 2.5 # Fallback
X_next['circuit_overtake_index'] = spa_overtake

# 4. Aggression (Historie van verschil tussen snelheid en finish)
# We gebruiken de eerder berekende 'aggression_score' kolom uit de historie
X_next['driver_aggression'] = X_next['driverId'].apply(lambda x: get_recent_form('driverId', x, target_col='aggression_score', default_val=0.0))

# 5. Overtake Rate & Consistency
X_next['driver_overtake_rate'] = X_next['driverId'].apply(lambda x: get_recent_form('driverId', x, target_col='positions_gained', default_val=0.0))
X_next['driver_consistency'] = X_next['driverId'].apply(lambda x: get_recent_form('driverId', x, target_col='std_lap_time_ms', default_val=5000.0))

# 3. DNF Rates
X_next['driver_dnf_rate'] = X_next['driverId'].apply(lambda x: get_recent_form('driverId', x, target_col='is_dnf', default_val=0.2))
X_next['constructor_dnf_rate'] = X_next['team_continuity_id'].apply(lambda x: get_recent_form('team_continuity_id', x, target_col='is_dnf', default_val=0.2))

# 4. Quali
X_next['quali_pos_filled'] = X_next['quali_position']
X_next['grid_penalty'] = X_next['grid'] - X_next['quali_pos_filled']

# 5. Weather Confidence (Voor Spa voorspelling)
# We halen het historische verschil op tussen nat en droog
X_next['weather_confidence_diff'] = X_next['driverId'].apply(lambda x: get_recent_form('driverId', x, target_col='weather_confidence_diff', default_val=0.0))


# 5. Huidige puntenstand (totaal van dit jaar tot nu toe)
def get_current_points(driver_id, year):
    return df[(df['driverId'] == driver_id) & (df['year'] == year)]['points'].sum()

X_next['points_before_race'] = X_next.apply(lambda x: get_current_points(x['driverId'], x['year']), axis=1)

# --- NIEUWE FEATURES VOOR VOORSPELLING ---
# 1. Quali Pace Deficit (Schatting op basis van grid, want we hebben geen Q3 tijden in de dict)
X_next['quali_pace_deficit'] = 1.0 + (X_next['grid'] - 1) * 0.003 # Aanname: 0.3% tijdverlies per gridplek
# 2. Teammate Diff (Berekend uit de reeds bepaalde form)
X_next['teammate_form_diff'] = X_next.groupby('team_continuity_id')['driver_recent_form'].transform(lambda x: x - x.mean())

# 3. Driver Style Affinity (Spa = HighSpeed)
# We moeten weten wat voor type Spa is. We halen het uit de circuit_stats logic of hardcoden het.
# Spa (13) is HighSpeed.
spa_type = 'HighSpeed'
if 'track_type' in df.columns:
    spa_row = df[df['circuitId'] == 13]
    if not spa_row.empty: spa_type = spa_row.iloc[0]['track_type']

X_next['driver_style_affinity'] = X_next['driverId'].apply(lambda x: get_style_affinity(x, spa_type))

# Nieuwe Track Features voor Spa
X_next['circuit_length_km'] = 7.004 # Spa is lang!
X_next['is_street_circuit'] = 0     # Spa is permanent

# Selecteer input
X_next_input = X_next[feature_cols]

# ---------------------------------------------------------
# STAP 3: TRAINING & EVALUATIE
# ---------------------------------------------------------

# --- ACTUAL RESULTS (SPA 2024) ---
actual_positions = {
    "Hamilton": 1, "Piastri": 2, "Leclerc": 3, "Verstappen": 4, "Norris": 5,
    "Sainz": 6, "Perez": 7, "Alonso": 8, "Ocon": 9, "Ricciardo": 10,
    "Stroll": 11, "Albon": 12, "Gasly": 13, "Magnussen": 14, "Bottas": 15,
    "Tsunoda": 16, "Sargeant": 17, "Hulkenberg": 18, "Guanyu": 19, "Russell": 20
}

# --- MODEL 1: POSITIE VOORSPELLEN (REGRESSIE) ---
print("\n==================================================")
print(" 1. TRAINING POSITION MODEL (XGBRanker - Learning to Rank) ")
print("==================================================")

# --- STAP 3A: DATA VOORBEREIDING VOOR RANKING ---
# XGBRanker heeft data nodig die gegroepeerd is per race.
# Het model leert: "In deze groep (race), wie is beter dan wie?"

# 1. Sorteren op RaceID (Cruciaal voor groepering)
# We maken een nieuwe gesorteerde set specifiek voor de Ranker
train_sorted = train_df.sort_values(by=['raceId', 'positionOrder'])

# 2. Filter finishers (We ranken alleen snelheid, crashes doen we apart)
mask_finishers = train_sorted['is_dnf'] == 0
X_rank = train_sorted[mask_finishers][feature_cols]

# 3. Target omzetten naar Relevance (Hoger is beter)
# Voor ranking werkt 'punten' beter dan positie. P1 = 20, P20 = 1.
y_rank_score = 21 - train_sorted[mask_finishers]['positionOrder']

# 4. Groepen maken (Hoeveel auto's in elke race?)
# We gebruiken de gesorteerde data om te tellen hoe groot elke 'query' (race) is.
groups = train_sorted[mask_finishers].groupby('raceId', sort=False).size().to_numpy()

print(f"   -> Training data: {len(X_rank)} rijen verdeeld over {len(groups)} races.")

print("   -> Training Ranker Model...")
rank_model = xgb.XGBRanker(
    objective='rank:ndcg', # Normalized Discounted Cumulative Gain (Optimaliseert de Top 3 zwaarder)
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist" # Sneller
)

rank_model.fit(
    X_rank, 
    y_rank_score, 
    group=groups, 
    verbose=True
)
print("   -> Model getraind!")

print("\n==================================================")
print(" 3. VOORSPELLING SPA 2024 (GECOMBINEERD MODEL) ")
print("==================================================")

# 1. Voorspel Posities (Snelheid)
# Ranker geeft een 'score' (hoe hoger hoe beter). Wij moeten dit omzetten naar rangorde (1, 2, 3).
pred_scores = rank_model.predict(X_next_input)

# We zetten de scores om naar posities (Hoogste score = P1)
pred_pos = pd.Series(pred_scores).rank(ascending=False, method='first').astype(int).values

# TIJDELIJK: DNF uitgeschakeld
pred_dnf_label = np.zeros(len(X_next_input), dtype=int)

# Resultaten samenvoegen
result_df = X_next.copy()
result_df['actual_pos'] = result_df['driver_name'].map(actual_positions)

# --- UITSLAG (Met DNF Logic) ---
print("\n" + "="*50)
print(" UITSLAG: XGBOOST (Inclusief DNF Model)")
print("="*50)

xgb_df = result_df.copy()
xgb_df['Score'] = pred_pos
xgb_df['DNF_Pred'] = pred_dnf_label

# Sorteren: DNF's onderaan (999), anders op score
xgb_df['Sort'] = np.where(xgb_df['DNF_Pred'] == 1, 999, xgb_df['Score'])
xgb_df = xgb_df.sort_values(by='Sort')

xgb_df['Rank'] = range(1, len(xgb_df) + 1)
xgb_df['Error'] = (xgb_df['Rank'] - xgb_df['actual_pos']).abs()
xgb_df['Status'] = np.where(xgb_df['DNF_Pred'] == 1, "DNF", "Finish")

print(f"Totale Fout XGBoost: {xgb_df['Error'].sum()}")
print(xgb_df[['Rank', 'driver_name', 'actual_pos', 'Error', 'Status', 'Score']].to_string(index=False))

print("-" * 60)
print("\nKlaar! Model geëvalueerd.")