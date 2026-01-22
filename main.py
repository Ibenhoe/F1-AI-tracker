import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import numpy as np

# ---------------------------------------------------------
# STAP 1: DATA LADEN & FEATURE ENGINEERING
# ---------------------------------------------------------

try:
    # We laden de ruwe data en passen de slimme features hier toe
    df = pd.read_csv("unprocessed_f1_training_data.csv")
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
    # shift(1) zorgt dat we de uitslag van de huidige race niet gebruiken om zichzelf te voorspellen
    return group.shift(1).rolling(window, min_periods=1).mean()

df['driver_recent_form'] = df.groupby('driverId')['positionOrder'].transform(calculate_rolling_avg)
df['constructor_recent_form'] = df.groupby('constructorId')['positionOrder'].transform(calculate_rolling_avg)

# Vul lege waarden (eerste races van seizoen) met een gemiddelde (bijv. P12)
df['driver_recent_form'] = df['driver_recent_form'].fillna(12.0)
df['constructor_recent_form'] = df['constructor_recent_form'].fillna(12.0)

# --- DEFINIEER FEATURES ---
feature_cols = [
    'grid', 
    'circuitId', 
    'driver_age',
    'driver_experience',
    'is_home_race',
    'grid_bin_code',
    'age_bin_code',
    'driver_recent_form',      # NIEUW: Vorm van de coureur
    'constructor_recent_form'  # NIEUW: Vorm van de auto
]

# Filter data: Alleen rijen met een geldig resultaat
train_df = df.dropna(subset=['positionOrder'] + feature_cols)

X = train_df[feature_cols]
y = train_df['positionOrder']

# ---------------------------------------------------------
# STAP 2: HET MODEL & GRID SEARCH
# ---------------------------------------------------------

print(f"\nStart training met {len(feature_cols)} features...")

model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    'n_estimators': [40, 50, 60, 100, 200],
    'max_depth': [1, 2, 3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5, 
    scoring='neg_mean_absolute_error',
    verbose=1
)

grid_search.fit(X, y)

print(f"\nBeste parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

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
def get_recent_form(id_col, id_val, target_col='positionOrder', window=5):
    # Pak de laatste N races van deze coureur/team uit de dataset
    history = df[df[id_col] == id_val].sort_values(by='date').tail(window)
    if len(history) == 0:
        return 12.0 # Geen historie? Dan gokken we middenveld (P12)
    return history[target_col].mean()

upcoming_race_dict = {
    'driver_name':   ['Leclerc', 'Perez', 'Hamilton', 'Norris', 'Piastri', 'Russell', 'Sainz', 'Alonso', 'Ocon', 'Albon', 'Verstappen', 'Gasly', 'Ricciardo', 'Bottas', 'Stroll', 'Hulkenberg', 'Magnussen', 'Sargeant', 'Zhou', 'Tsunoda'],
    'grid':          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'driverId':      [844, 815, 1, 846, 857, 847, 832, 4, 839, 848, 830, 842, 817, 822, 840, 807, 825, 858, 855, 852],
    'constructorId': [6, 9, 131, 1, 1, 131, 6, 117, 214, 3, 9, 214, 213, 15, 117, 210, 210, 3, 15, 213],
    'circuitId':     [13] * 20,
    'year':          [2024] * 20,
    'driver_age':    [26, 34, 39, 24, 23, 26, 29, 43, 27, 28, 26, 28, 35, 34, 25, 37, 31, 23, 25, 24],
    'nationality':   ['Monegasque', 'Mexican', 'British', 'British', 'Australian', 'British', 'Spanish', 'Spanish', 'French', 'Thai', 'Dutch', 'French', 'Australian', 'Finnish', 'Canadian', 'German', 'Danish', 'American', 'Chinese', 'Japanese'],
    'country':       ['Belgium'] * 20
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
X_next['driver_recent_form'] = X_next['driverId'].apply(lambda x: get_recent_form('driverId', x))
X_next['constructor_recent_form'] = X_next['constructorId'].apply(lambda x: get_recent_form('constructorId', x))

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
top_10_dashboard = final_ranking[['driver_name', 'grid', 'ai_score', 'driver_recent_form', 'constructor_recent_form']].head(10)
top_10_dashboard.index = range(1, 11) # Nummers 1 t/m 10 ervoor zetten

print("\n========================================")
print("   AI PREDICTIE: UITSLAG KOMENDE RACE   ")
print("========================================")
print(top_10_dashboard)
print("\nKlaar! Model getraind op echte Ergast data.")