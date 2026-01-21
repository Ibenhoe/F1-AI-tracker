import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import numpy as np

# ---------------------------------------------------------
# STAP 1: DATA LADEN (De samengevoegde Ergast Data)
# ---------------------------------------------------------

# Laad de data die we in de vorige stap hebben gemaakt
try:
    df = pd.read_csv("processed_f1_training_data.csv")
    print(f"Data geladen! Totaal {len(df)} regels historische data.")
except FileNotFoundError:
    print("FOUT: Run eerst het 'preprocessing' script om 'processed_f1_training_data.csv' te maken!")
    exit()

# DEFINIEER FEATURES (X) en TARGET (y)
# We gebruiken de kolomnamen zoals ze in Ergast/Processed file staan.
# Let op: We hebben 'rain_prob' verwijderd omdat Ergast dat niet standaard heeft.
feature_cols = ['grid', 'driverId', 'constructorId', 'circuitId', 'year', 'driver_age']

X = df[feature_cols]
y = df['positionOrder'] # Dit is de uiteindelijke ranking (1 = Winnaar)

# ---------------------------------------------------------
# STAP 2: HET MODEL & GRID SEARCH
# ---------------------------------------------------------

print("\nStart Grid Search... even geduld.")

model = xgb.XGBRegressor(objective='reg:squarederror')

# We testen deze combinaties (je kunt dit later uitbreiden)
param_grid = {
    'n_estimators': [70, 80, 90, 100, 150],   # Aantal bomen
    'max_depth': [1, 3, 4, 5, 7],   # Diepte van bomen
    'learning_rate': [0.01, 0.07, 0.08, 0.09, 0.1]  # Leersnelheid
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5, # 5-Fold Cross Validation (Jouw eis)
    scoring='neg_mean_absolute_error',
    verbose=1
)

grid_search.fit(X, y)

print(f"\nBeste parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# ---------------------------------------------------------
# STAP 3: DATA VOORBEREIDEN VOOR "KOMENDE RACE"
# ---------------------------------------------------------
# Omdat we nog geen echte API-koppeling hebben voor de race van volgende week,
# maken we hier handmatig de "Line-up" voor een fictieve race (bijv. Spa 2024).

print("\nVoorspelling voorbereiden voor komende race...")

# Voorbeeld: We simuleren een startgrid met 5 coureurs
# Je moet hier de echte ID's gebruiken uit je drivers.csv en constructors.csv!
# Bijv: Hamilton = 1, Verstappen = 830 (of 33/1 afhankelijk van je file versie), Norris = 846
upcoming_race_dict = {
    'driver_name':   ['Verstappen', 'Norris', 'Hamilton', 'Leclerc', 'Piastri'],
    'grid':          [11,            4,        3,          1,         5],       # Startpositie
    'driverId':      [830,          846,      1,          16,        857],     # Ergast IDs
    'constructorId': [9,            1,        131,        6,         1],       # Red Bull=9, McLaren=1 etc.
    'circuitId':     [6,            6,        6,          6,         6],       # Spa = ID 6 (voorbeeld)
    'year':          [2024,         2024,     2024,       2024,      2024],
    'driver_age':    [26,           24,       39,         26,        23]       # Geschatte leeftijd
}

X_next = pd.DataFrame(upcoming_race_dict)

# We selecteren alleen de kolommen die het model kent (dus zonder 'driver_name')
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
top_5_dashboard = final_ranking[['driver_name', 'grid', 'ai_score']].head(5)
top_5_dashboard.index = range(1, 6) # Nummers 1 t/m 5 ervoor zetten

print("\n========================================")
print("   AI PREDICTIE: UITSLAG KOMENDE RACE   ")
print("========================================")
print(top_5_dashboard)
print("\nKlaar! Model getraind op echte Ergast data.")