import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

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
le_grid = LabelEncoder()
le_age = LabelEncoder()

# .astype(str) zorgt dat we geen crashes krijgen op lege waarden
df['grid_bin_code'] = le_grid.fit_transform(df['grid_bin'].astype(str))
df['age_bin_code'] = le_age.fit_transform(df['age_bin'].astype(str))

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
df['constructor_recent_form'] = df.groupby('constructorId')['positionOrder'].transform(calculate_rolling_avg)

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
    df['constructor_recent_pit_avg'] = df.groupby('constructorId')['avg_pit_duration_filled'].transform(calculate_rolling_avg)
    
    # Eerste races vullen met fallback
    df['driver_recent_pit_avg'] = df['driver_recent_pit_avg'].fillna(global_pit_mean)
    df['constructor_recent_pit_avg'] = df['constructor_recent_pit_avg'].fillna(global_pit_mean)
else:
    print("LET OP: Geen pitstop data gevonden. Features worden standaardwaarden.")
    global_pit_mean = 24000.0
    df['driver_recent_pit_avg'] = global_pit_mean
    df['constructor_recent_pit_avg'] = global_pit_mean

# --- WEERDATA VOORBEREIDEN ---
# We vullen ontbrekende weerdata op met logische standaardwaarden
if 'weather_temp_c' in df.columns:
    df['weather_temp_c'] = df['weather_temp_c'].fillna(df['weather_temp_c'].mean()) # Gemiddelde temp
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

def parse_fastest_lap(t_str):
    if pd.isna(t_str) or str(t_str).strip() == '\\N': return np.nan
    try:
        # Formaat M:SS.mmm (bijv 1:30.123)
        parts = str(t_str).split(':')
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(t_str)
    except:
        return np.nan

if 'fastestLapTime' in df.columns:
    # 1. Zet tijd om naar seconden
    df['fastest_lap_seconds'] = df['fastestLapTime'].apply(parse_fastest_lap)
    
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
else:
    df['driver_recent_speed'] = 1.10

# 4. Puntenstand voorafgaand aan de race (Strategie context)
# We berekenen de cumulatieve som min de punten van vandaag = punten bij start.
df['points_before_race'] = (df.groupby(['year', 'driverId'])['points'].cumsum() - df['points']).fillna(0)

# --- DEFINIEER FEATURES ---
feature_cols = [
    'grid', 
    'circuitId', 
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
    'driver_recent_pit_avg',
    'constructor_recent_pit_avg',
    'driver_recent_speed',
    'points_before_race'
]

# Filter data: Alleen rijen met een geldig resultaat
train_df = df.dropna(subset=['positionOrder'] + feature_cols)

X = train_df[feature_cols]
y = train_df['positionOrder']

# ---------------------------------------------------------
# STAP 2: MODELLEN VERGELIJKEN & TRAINEN
# ---------------------------------------------------------

print(f"\nStart training met {len(feature_cols)} features...")

# We splitsen in Train en Test (op tijd gebaseerd, niet random, want F1 is tijdreeks)
# We pakken de laatste 20% van de races als test set om te kijken welk model het beste is
split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

models = {
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42),
    "Linear Regression": LinearRegression()
}

best_model = None
best_mae = float('inf')
best_name = ""

print("\n--- Model Resultaten (MAE - Lager is beter) ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"{name}: {mae:.4f}")
    
    if mae < best_mae:
        best_mae = mae
        best_model = model
        best_name = name

print(f"\nWINNAAR: {best_name} met MAE {best_mae:.4f}")
print("Hertrainen van het beste model op ALLE data...")
best_model.fit(X, y)

# ---------------------------------------------------------
# STAP 3: VOORSPELLING KOMENDE RACE (SPA 2024)
# ---------------------------------------------------------
print("\nVoorspelling voorbereiden voor komende race...")

# Hulpfunctie: Haal de huidige ervaring van een coureur op uit de historie
def get_experience(driver_id):
    if driver_id in df['driverId'].values:
        return df[df['driverId'] == driver_id]['driver_experience'].max() + 1
    return 0 # Nieuwe coureur

# Hulpfunctie: Haal de recente vorm op (gemiddelde laatste 5 races) uit de historie
def get_recent_form(id_col, id_val, target_col='positionOrder', window=5, default_val=12.0):
    # Pak de laatste N races van deze coureur/team uit de dataset
    history = df[df[id_col] == id_val].sort_values(by='date').tail(window)
    if len(history) == 0:
        return default_val # Geen historie? Dan gokken we de default
    
    # Gewogen gemiddelde berekenen voor de voorspelling
    values = history[target_col].values
    weights = np.arange(1, len(values) + 1)
    return np.average(values, weights=weights)

# Hulpfunctie: Haal de historie op dit circuit op
def get_track_history(driver_id, circuit_id):
    history = df[(df['driverId'] == driver_id) & (df['circuitId'] == circuit_id)]
    if len(history) == 0:
        return 12.0, 12.0, 0.0 # Default waarden
    
    avg_grid = history['grid'].mean()
    avg_finish = history['positionOrder'].mean()
    podiums = len(history[history['positionOrder'] <= 3])
    return avg_grid, avg_finish, podiums

upcoming_race_dict = {
    'driver_name':   ['Leclerc', 'Perez', 'Hamilton', 'Norris', 'Piastri', 'Russell', 'Sainz', 'Alonso', 'Ocon', 'Albon', 'Verstappen', 'Gasly', 'Ricciardo', 'Bottas', 'Stroll', 'Hulkenberg', 'Magnussen', 'Sargeant', 'Zhou', 'Tsunoda'],
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
    'weather_cloud_pct': [80.0] * 20
}

X_next = pd.DataFrame(upcoming_race_dict)

# Feature Engineering toepassen op de nieuwe data
X_next['driver_experience'] = X_next['driverId'].apply(get_experience)

# Home Race logic
X_next['mapped_nationality'] = X_next['nationality'].map(nationality_map).fillna(X_next['nationality'])
X_next['is_home_race'] = np.where(X_next['mapped_nationality'] == X_next['country'], 1, 0)

# Bins & Encoding (Gebruik dezelfde encoders als bij training!)
X_next['grid_bin'] = pd.cut(X_next['grid'], bins=[-1, 1, 3, 10, 15, 25], labels=['Pole', 'Top3', 'Points', 'Midfield', 'Back'])
X_next['age_bin'] = pd.cut(X_next['driver_age'], bins=[17, 24, 30, 36, 60], labels=['Rookie', 'Prime', 'Experienced', 'Veteran'])

X_next['grid_bin_code'] = le_grid.transform(X_next['grid_bin'].astype(str))
X_next['age_bin_code'] = le_age.transform(X_next['age_bin'].astype(str))

# Recent Form toevoegen aan de voorspelling
X_next['driver_recent_form'] = X_next['driverId'].apply(lambda x: get_recent_form('driverId', x, default_val=12.0))
X_next['constructor_recent_form'] = X_next['constructorId'].apply(lambda x: get_recent_form('constructorId', x, default_val=12.0))

# Pitstop Form toevoegen (gebruikt de 'avg_pit_duration_filled' kolom uit de historie)
X_next['driver_recent_pit_avg'] = X_next['driverId'].apply(lambda x: get_recent_form('driverId', x, target_col='avg_pit_duration_filled', default_val=global_pit_mean))
X_next['constructor_recent_pit_avg'] = X_next['constructorId'].apply(lambda x: get_recent_form('constructorId', x, target_col='avg_pit_duration_filled', default_val=global_pit_mean))

# Track History toevoegen
track_stats = X_next.apply(lambda x: get_track_history(x['driverId'], x['circuitId']), axis=1)
X_next['driver_track_avg_grid'] = [x[0] for x in track_stats]
X_next['driver_track_avg_finish'] = [x[1] for x in track_stats]
X_next['driver_track_podiums'] = [x[2] for x in track_stats]

# Nieuwe features toevoegen aan voorspelling
# 1. Recente snelheid (gebruikt de historie van fastest laps)
X_next['driver_recent_speed'] = X_next['driverId'].apply(lambda x: get_recent_form('driverId', x, target_col='speed_ratio', default_val=1.10))

# 2. Huidige puntenstand (totaal van dit jaar tot nu toe)
def get_current_points(driver_id, year):
    return df[(df['driverId'] == driver_id) & (df['year'] == year)]['points'].sum()

X_next['points_before_race'] = X_next.apply(lambda x: get_current_points(x['driverId'], x['year']), axis=1)

# Selecteer input
X_next_input = X_next[feature_cols]

# ---------------------------------------------------------
# STAP 4: VOORSPELLEN & RANKING
# ---------------------------------------------------------

# Voorspel de 'score' (lagere score is beter)
predictions = best_model.predict(X_next_input)

# Plak de score terug aan de dataframe zodat we namen kunnen zien
X_next['ai_score'] = predictions

# Sorteren: Laagste score bovenaan = P1
final_ranking = X_next.sort_values(by='ai_score', ascending=True)

# Maak een net dashboard lijstje
full_dashboard = final_ranking[['driver_name', 'grid', 'ai_score', 'driver_recent_form', 'driver_track_podiums']]
full_dashboard.index = range(1, len(full_dashboard) + 1) # Nummers 1 t/m N ervoor zetten

print("\n========================================")
print("   AI PREDICTIE: UITSLAG KOMENDE RACE   ")
print("========================================")
print(full_dashboard)
print("\nKlaar! Model getraind op echte Ergast data.")