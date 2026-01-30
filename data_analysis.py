# %% [markdown]
# # F1 Data Analysis & Visualization
# Dit bestand werkt als een Notebook. Elke sectie met '# %%' is een cel.
# Je kunt op 'Run Cell' klikken (in VS Code) om alleen dat stukje te draaien.

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

# %%
# 1. DATA LADEN
print("Bestand laden...")
# We laden nu direct de samengevoegde file
df = pd.read_csv('unprocessed_f1_training_data.csv')

print(f"Data geladen. Totaal aantal rijen: {len(df)}")

# %%
# 3. SCHOONMAKEN & FEATURE ENGINEERING

# Datums en Leeftijd
df['date'] = pd.to_datetime(df['date'])
df['dob'] = pd.to_datetime(df['dob'])
df['driver_age'] = (df['date'] - df['dob']).dt.days / 365.25

# Vervang \N door NaN en converteer naar getallen
df = df.replace(r'\\N', np.nan, regex=True)

# Belangrijke kolommen numeriek maken voor correlatie
numeric_cols = ['grid', 'positionOrder', 'points_constructor', 'alt', 'year', 'driver_age',
                'quali_position', 'pit_stops_count', 'pit_stops_duration_ms', 'avg_lap_time_ms']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Verwijder rijen zonder finish positie of grid
df_clean = df.dropna(subset=['positionOrder', 'grid', 'driver_age'])

# %%
# 4. VISUALISATIE: CORRELATIE MATRIX
# Hiermee zie je welke features invloed hebben op 'positionOrder' (de finish positie)

plt.figure(figsize=(10, 8))
correlation_matrix = df_clean[numeric_cols].corr()

# Heatmap tekenen
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlatie Matrix: Wat hangt samen met de Finish Positie?")
plt.show()

# %%
# 5. VISUALISATIE: FEATURE IMPORTANCE (XGBoost)
# Laat het model vertellen wat het belangrijkst vindt

# We voegen de nieuwe features toe aan de analyse
# Let op: sommige features (zoals pit stops) zijn pas NA de race bekend.
# Voor analyse is dat prima, voor voorspelling (prediction) moet je oppassen.
features_to_use = ['grid', 'points_constructor', 'alt', 'year', 'driver_age', 
                   'quali_position', 'pit_stops_count', 'pit_stops_duration_ms', 'avg_lap_time_ms']
X = df_clean[features_to_use]
y = df_clean['positionOrder']

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=80, max_depth=4, learning_rate=0.1)
model.fit(X, y)

xgb.plot_importance(model)
plt.title("Welke features vindt het AI model het belangrijkst?")
plt.show()

# %%
# 6. DATA KWALITEIT & OPSCHONING CHECKS
# Hier kijken we naar wat er 'vies' is aan de data om te beslissen wat we verwijderen.

# A. Missende waarden (Null / NaN) visualiseren
# Geel = Data mist. Dit helpt je beslissen of je kolommen als 'q3' moet weggooien.
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title("Missende data in de dataset (Geel = Missend)")
plt.show()

# B. Outliers (Uitschieters) zoeken met Boxplots
# Check 1: Pitstops. Zijn er stops van 20+ minuten (Rode vlag)? Die verpesten je gemiddelde.
plt.figure(figsize=(12, 5))
sns.boxplot(x=df_clean['pit_stops_duration_ms'])
plt.title("Boxplot: Pitstop Duur (Let op extreme uitschieters door rode vlaggen)")
plt.show()

# Check 2: Rondetijden.
plt.figure(figsize=(12, 5))
sns.boxplot(x=df_clean['avg_lap_time_ms'])
plt.title("Boxplot: Gemiddelde Rondetijden")
plt.show()

# C. Logische Checks
# Grid positie 0 betekent vaak start vanuit pitstraat. Is dat relevant of ruis?
pitlane_starts = df_clean[df_clean['grid'] == 0]
print(f"\n--- Aantal starts vanuit pitlane (Grid=0): {len(pitlane_starts)} ---")

# Check hoeveel data we hebben per jaar (is data uit 1950 nog nuttig?)
print("\n--- Aantal races per decennium ---")
print(df_clean['year'].apply(lambda x: (x//10)*10).value_counts().sort_index())