# 1. Imports: De gereedschapskist
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------
# STAP 1: DATA LADEN & VOORBEREIDEN
# ---------------------------------------------------------

# Stel je hebt een CSV met historische data (2018-2023)
# Kolommen: ['driver_id', 'constructor_id', 'grid_pos', 'circuit_id', 'rain_prob', 'track_temp', 'final_position']
df = pd.read_csv("historical_race_data.csv")

# X = De Features (waarop we baseren)
# Let op: Tekst (zoals 'Verstappen') moet eerst omgezet worden naar getallen (Encoding)
X = df[['grid_pos', 'driver_id', 'constructor_id', 'circuit_id', 'rain_prob', 'track_temp']]

# y = De Target (wat we willen voorspellen)
y = df['final_position']

# ---------------------------------------------------------
# STAP 2: HET MODEL & GRID SEARCH (Jouw kernvraag)
# ---------------------------------------------------------

# We kiezen XGBoost Regressor (omdat we een positie 1 t/m 20 voorspellen)
model = xgb.XGBRegressor(objective='reg:squarederror')

# Hyperparameters om te testen (Grid Search probeert alle combinaties)
param_grid = {
    'n_estimators': [100, 150, 200],     # Hoeveel 'bomen' bouwt het model?
    'max_depth': [3, 5, 7],         # Hoe complex mag elke boom zijn?
    'learning_rate': [0.01, 0.1]    # Hoe snel leert het model?
}

# Hier gebeurt de magie: cv=5 betekent 5-Fold Cross Validation
# Het model wordt 5x getraind op 80% data en getest op 20%
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,                      # <-- Dit is jouw eis: 5x trainen/validatie
    scoring='neg_mean_absolute_error', # We willen zo dicht mogelijk bij de echte positie zitten
    verbose=1
)

print("Start met trainen... dit kan even duren.")
grid_search.fit(X, y)

print(f"Beste parameters gevonden: {grid_search.best_params_}")

# Het 'beste' model wordt automatisch opgeslagen in best_model
best_model = grid_search.best_estimator_

# ---------------------------------------------------------
# STAP 3: VOORSPELLING VOOR DE AANKOMENDE RACE (MVP)
# ---------------------------------------------------------

# We laden de dataset van de race van komend weekend (vóór de start)
# Deze data haal je uit je API (FastF1/Ergast) na de kwalificatie
next_race_data = pd.read_csv("upcoming_race_spa_2024.csv") 

# We selecteren dezelfde kolommen als bij het trainen
X_next = next_race_data[['grid_pos', 'driver_id', 'constructor_id', 'circuit_id', 'rain_prob', 'track_temp']]

# Laat het model voorspellen
# Het model geeft getallen, bv: Verstappen -> 1.2, Norris -> 2.4, Hamilton -> 2.9
predictions = best_model.predict(X_next)

# Voeg voorspellingen toe aan de data tabel
next_race_data['predicted_position_score'] = predictions

# ---------------------------------------------------------
# STAP 4: RANKING & TOP 5 GENEREREN
# ---------------------------------------------------------

# Sorteer de lijst van laag naar hoog (Laagste score = Plek 1)
final_ranking = next_race_data.sort_values(by='predicted_position_score', ascending=True)

# Selecteer alleen de bovenste 5 en de relevante kolommen voor je dashboard
top_5_dashboard = final_ranking[['driver_name', 'team', 'predicted_position_score']].head(5)

print("\n--- AI VOORSPELLING: TOP 5 ---")
# Reset index zodat het netjes 1 t/m 5 toont
top_5_dashboard.index = range(1, 6) 
print(top_5_dashboard)

# TODO: Stuur deze 'top_5_dashboard' dataset naar je Frontend (React/Vue/Dashboard)