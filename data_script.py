import pandas as pd
import numpy as np
from datetime import datetime

# 1. DATA LADEN
# Zorg dat je csv bestanden in dezelfde map staan als dit script
print("Bestanden laden...")
results = pd.read_csv('../F1_data_mangement/results.csv')
races = pd.read_csv('../F1_data_mangement/races.csv')
drivers = pd.read_csv('../F1_data_mangement/drivers.csv')
constructors = pd.read_csv('../F1_data_mangement/constructors.csv')

# 2. DATA SAMENVOEGEN (MERGEN)
# We beginnen met 'results' als basis, want daar staat elke finish in.

# Stap A: Race info toevoegen (Datum, Circuit)
df = pd.merge(results, races[['raceId', 'year', 'circuitId', 'date']], on='raceId', how='left')

# Stap B: Driver info toevoegen (Naam, Geboortedatum)
df = pd.merge(df, drivers[['driverId', 'dob', 'nationality']], on='driverId', how='left')

# Stap C: Constructor info toevoegen (Teamnaam)
df = pd.merge(df, constructors[['constructorId', 'name']], on='constructorId', how='left', suffixes=('', '_team'))

# 3. FEATURE ENGINEERING (De data slim maken voor AI)

# A. Leeftijd van coureur berekenen op moment van race
# We moeten datums converteren naar echte datum-objecten
df['date'] = pd.to_datetime(df['date'])
df['dob'] = pd.to_datetime(df['dob'])
df['driver_age'] = (df['date'] - df['dob']).dt.days / 365.25

# B. 'Null' waarden opschonen (in Ergast is \N vaak een lege waarde)
df = df.replace(r'\\N', np.nan, regex=True)

# 4. SELECTEREN VAN KOLOMMEN VOOR XGBOOST
# We hebben alleen getallen nodig. Tekst (zoals 'Dutch') moeten we later nog omzetten.

final_df = df[[
    'raceId',           # Om later te weten welke race het is
    'year',             # Feature: Jaar
    'grid',             # Feature: Startpositie (BELANGRIJK!)
    'circuitId',        # Feature: Welk circuit
    'constructorId',    # Feature: Welk team
    'driverId',         # Feature: Welke coureur
    'driver_age',       # Feature: Leeftijd
    'positionOrder'     # TARGET: De uiteindelijke ranking (1 t/m 20)
]]

# Verwijder rijen waar cruciale data mist
final_df = final_df.dropna()

print("\n--- DATA PREPARED ---")
print(final_df.head())
print(f"\nTotaal aantal rijen om op te trainen: {len(final_df)}")

# 5. OPSLAAN VOOR JE MODEL
final_df.to_csv('processed_f1_training_data.csv', index=False)
print("Bestand 'processed_f1_training_data.csv' is aangemaakt!")