# %% [markdown]
# # F1 Data Science Laboratorium 🧪
# Dit script analyseert welke data-strategie het beste werkt.
# 1. Vergelijking van Imputatie (Null waarden vullen)
# 2. Feature Importance (Welke data telt echt?)
# 3. Feature Selectie (Wat kan weg?)

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# %%
# 1. DATA LADEN & BASIS VOORBEREIDING
print("--- 1. DATA LADEN ---")
df = pd.read_csv('unprocessed_f1_training_data.csv')

# Sorteren op datum is cruciaal voor Time Series
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# Filter op modern tijdperk (2015+) voor relevantere analyse
df = df[df['year'] >= 2015].copy()
print(f"Analyse dataset: {len(df)} races (2015-heden)")

# %%
# 2. EXPERIMENT: IMPUTATIE STRATEGIEËN (Null waarden vullen)
# We testen wat het beste werkt voor 'weather_temp_c' (vaak null): Mean, Median of Zero.

print("\n--- 2. EXPERIMENT: IMPUTATIE (Mean vs Median) ---")

# Maak een test-set met alleen rijen waar we wél weerdata hebben om te valideren
valid_weather_df = df.dropna(subset=['weather_temp_c']).copy()

# Simuleer gaten in de data (we verwijderen random 20% van de temp)
np.random.seed(42)
mask = np.random.rand(len(valid_weather_df)) < 0.2
valid_weather_df.loc[mask, 'temp_missing'] = np.nan
valid_weather_df.loc[~mask, 'temp_missing'] = valid_weather_df['weather_temp_c']

strategies = ['mean', 'median', 'constant'] # constant = 0
results = {}

for strat in strategies:
    imputer = SimpleImputer(strategy=strat, fill_value=0)
    # Reshape omdat imputer 2D array verwacht
    filled = imputer.fit_transform(valid_weather_df[['temp_missing']])
    
    # Bereken hoe ver we er naast zitten t.o.v. de echte waarde
    mae = mean_absolute_error(valid_weather_df['weather_temp_c'], filled)
    results[strat] = mae

print("Foutmarge per strategie (lager is beter):")
for strat, score in results.items():
    print(f" - {strat.capitalize()}: {score:.4f} graden afwijking")

best_strat = min(results, key=results.get)
print(f"CONCLUSIE: '{best_strat.upper()}' is de beste methode om gaten te vullen!")

# %%
# 3. FEATURE IMPORTANCE & SELECTIE
# We trainen een XGBoost model om te zien welke features écht tellen.

print("\n--- 3. FEATURE IMPORTANCE ANALYSE ---")

# Eerst even snel de features bouwen (zoals in main.py)
df['grid_penalty'] = df['grid'] - df['quali_position'].fillna(df['grid'])
df['driver_experience'] = df.groupby('driverId').cumcount()

# Rolling averages (Form)
def calculate_rolling_avg(group):
    return group.shift(1).rolling(5, min_periods=1).mean()

df['driver_recent_form'] = df.groupby('driverId')['positionOrder'].transform(calculate_rolling_avg).fillna(12)

# --- NIEUWE FEATURES VOOR ANALYSE ---
# 1. Aggression (Snelheid vs Finish)
if 'fastestLapTime' in df.columns:
    # Parse tijd naar seconden (simpel voor analyse)
    df['fastest_lap_seconds'] = pd.to_numeric(df['fastestLapTime'].str.replace(r'.*:', '', regex=True), errors='coerce')
    df['fastest_lap_rank'] = df.groupby('raceId')['fastest_lap_seconds'].rank(method='min')
    df['aggression_score'] = df['positionOrder'] - df['fastest_lap_rank']
    df['driver_aggression'] = df.groupby('driverId')['aggression_score'].transform(calculate_rolling_avg).fillna(0)
else:
    df['driver_aggression'] = 0

# 2. Weather Confidence (Nat vs Droog)
df['is_wet'] = df['weather_precip_mm'] > 0.1
df['weather_confidence'] = df.groupby('driverId')['positionOrder'].transform(lambda x: x.rolling(10, min_periods=1).mean()) # Simpele proxy voor analyse

# --- NIEUW: BATTLE & CONSISTENCY FEATURES ---
# 1. Overtake Rate: Gemiddeld aantal posities gewonnen per race
df['positions_gained'] = df['grid'] - df['positionOrder']
df['driver_overtake_rate'] = df.groupby('driverId')['positions_gained'].transform(calculate_rolling_avg).fillna(0)

# 2. Consistency: Hoe stabiel zijn de rondetijden? (std_lap_time_ms)
# Lager is beter (stabieler). Als dit hoog is, is er vaak een gevecht of probleem.
if 'std_lap_time_ms' in df.columns:
    df['driver_consistency'] = df.groupby('driverId')['std_lap_time_ms'].transform(calculate_rolling_avg).fillna(5000)
else:
    df['driver_consistency'] = 5000

# 3. Punten voorafgaand aan race (ipv punten van VANDAAG, want dat is valsspelen!)
df['points_before_race'] = (df.groupby(['year', 'driverId'])['points'].cumsum() - df['points']).fillna(0)

# Features selecteren voor analyse
features = [
    'grid', 'grid_penalty', 'driver_age', 'driver_experience',
    'driver_recent_form', 'driver_aggression', 'weather_confidence',
    'weather_precip_mm', 'points_before_race', 
    'driver_overtake_rate', 'driver_consistency'
]

# Data opschonen voor het model
analysis_df = df.dropna(subset=['positionOrder'] + features).copy()

X = analysis_df[features]
y = analysis_df['positionOrder']

# Model trainen
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=4, random_state=42)
model.fit(X, y)

# Belangrijkheid ophalen
importance = model.feature_importances_
feat_imp = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)

print("\nTop Features (Invloed op voorspelling):")
print(feat_imp)

# Visualisatie
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
plt.title('Welke data vindt het AI model het belangrijkst?')
plt.show()

# %%
# 4. DROP CANDIDATES (Wat kan weg?)
# Features met < 0.01 importance voegen vaak alleen maar ruis toe.

threshold = 0.01
drop_candidates = feat_imp[feat_imp['Importance'] < threshold]['Feature'].tolist()

print("\n--- 4. ADVIES: TE VERWIJDEREN FEATURES ---")
if drop_candidates:
    print(f"De volgende features hebben nauwelijks invloed (< {threshold}):")
    print(drop_candidates)
    print("-> Overweeg deze uit 'main.py' te halen voor een sneller en schoner model.")
else:
    print("Alle features lijken relevant genoeg (> 1% invloed). Goed bezig!")

# %%
# 5. SMOOTHNESS CHECK (Rolling Windows)
# Testen of een window van 3, 5 of 10 races beter werkt voor 'Recent Form'.

print("\n--- 5. EXPERIMENT: BESTE WINDOW SIZE (Form) ---")

windows = [3, 5, 10]
correlations = {}

for w in windows:
    col_name = f'form_w{w}'
    # Bereken rolling mean
    df[col_name] = df.groupby('driverId')['positionOrder'].transform(
        lambda x: x.shift(1).rolling(w, min_periods=1).mean()
    )
    # Bereken correlatie met de daadwerkelijke finish positie
    # We willen een HOGE correlatie (dicht bij 1.0)
    corr = df[[col_name, 'positionOrder']].corr().iloc[0, 1]
    correlations[w] = corr

print("Correlatie met uitslag (Hoger is beter):")
for w, corr in correlations.items():
    print(f" - Window {w} races: {corr:.4f}")

best_window = max(correlations, key=correlations.get)
print(f"CONCLUSIE: Een terugblik van {best_window} races geeft de beste voorspelling.")
