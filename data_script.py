import pandas as pd
import numpy as np
from datetime import datetime
import os

# 1. DATA LADEN
# Zorg dat je csv bestanden in dezelfde map staan als dit script
print("Bestanden laden...")

# We gebruiken het pad van dit script om de data map te vinden, ongeacht waar je het script start
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../F1_data_mangement')

results = pd.read_csv(os.path.join(data_dir, 'results.csv'))
races = pd.read_csv(os.path.join(data_dir, 'races.csv'))
drivers = pd.read_csv(os.path.join(data_dir, 'drivers.csv'))
constructors = pd.read_csv(os.path.join(data_dir, 'constructors.csv'))
circuits = pd.read_csv(os.path.join(data_dir, 'circuits.csv'))
constructor_results = pd.read_csv(os.path.join(data_dir, 'constructor_results.csv'))
status = pd.read_csv(os.path.join(data_dir, 'status.csv'))
lap_times = pd.read_csv(os.path.join(data_dir, 'lap_times.csv'))
pit_stops = pd.read_csv(os.path.join(data_dir, 'pit_stops.csv'))
qualifying = pd.read_csv(os.path.join(data_dir, 'qualifying.csv'))
seasons = pd.read_csv(os.path.join(data_dir, 'seasons.csv'))

# Weerdata laden (indien beschikbaar)
weather_file = os.path.join(script_dir, 'f1_weather_data.csv')
if os.path.exists(weather_file):
    print("Weerdata gevonden en aan het laden...")
    weather_data = pd.read_csv(weather_file)
else:
    print("LET OP: 'f1_weather_data.csv' niet gevonden. Run eerst 'fetch_weather.py'!")
    weather_data = pd.DataFrame()

# 2. DATA SAMENVOEGEN (MERGEN)
# We beginnen met 'results' als basis, want daar staat elke finish in.

# Stap A: Race info toevoegen (Datum, Circuit)
df = pd.merge(results, races[['raceId', 'year', 'circuitId', 'date']], on='raceId', how='left')

# Stap B: Driver info toevoegen (Naam, Geboortedatum)
df = pd.merge(df, drivers[['driverId', 'driverRef', 'dob', 'nationality']], on='driverId', how='left')

# Stap C: Constructor info toevoegen (Teamnaam)
df = pd.merge(df, constructors[['constructorId', 'name']], on='constructorId', how='left', suffixes=('', '_team'))

df = pd.merge(df, circuits[['circuitId', 'alt', 'country']], on='circuitId', how='left')

df = pd.merge(df, constructor_results[['raceId', 'constructorId', 'points']], on=['raceId', 'constructorId'], how='left', suffixes=('', '_constructor'))

# Stap E: Status info (bv. 'Finished', 'Collision')
df = pd.merge(df, status[['statusId', 'status']], on='statusId', how='left')

# Stap F: Seasons info (url)
df = pd.merge(df, seasons[['year', 'url']], on='year', how='left', suffixes=('', '_season'))

# Stap G: Qualifying info
# We hernoemen 'position' naar 'quali_position' om verwarring met de race-uitslag te voorkomen
qualifying = qualifying.rename(columns={'position': 'quali_position'})
df = pd.merge(df, qualifying[['raceId', 'driverId', 'quali_position', 'q1', 'q2', 'q3']], on=['raceId', 'driverId'], how='left')

# Stap H: Pit Stops (Aggregatie)
# We tellen het aantal stops en de totale tijd in de pit per coureur per race
pit_stops_agg = pit_stops.groupby(['raceId', 'driverId']).agg(
    pit_stops_count=('stop', 'count'),
    pit_stops_duration_ms=('milliseconds', 'sum')
).reset_index()
df = pd.merge(df, pit_stops_agg, on=['raceId', 'driverId'], how='left')

# Stap I: Lap Times (Aggregatie)
# We berekenen de gemiddelde rondetijd en consistentie (std dev)
lap_times_agg = lap_times.groupby(['raceId', 'driverId']).agg(
    avg_lap_time_ms=('milliseconds', 'mean'),
    std_lap_time_ms=('milliseconds', 'std')
).reset_index()
df = pd.merge(df, lap_times_agg, on=['raceId', 'driverId'], how='left')

# Stap J: Weer toevoegen
if not weather_data.empty:
    df = pd.merge(df, weather_data, on='raceId', how='left')

# 3. FEATURE ENGINEERING (De data slim maken voor AI)

# A. Leeftijd van coureur berekenen op moment van race
# We moeten datums converteren naar echte datum-objecten
df['date'] = pd.to_datetime(df['date'])
df['dob'] = pd.to_datetime(df['dob'])
df['driver_age'] = (df['date'] - df['dob']).dt.days / 365.25

# B. 'Null' waarden opschonen (in Ergast is \N vaak een lege waarde)
df = df.replace(r'\\N', np.nan, regex=True)

# C. TIRE STRATEGY (STINT LENGTHS)
# We berekenen gemiddelde rondes per stint. (Totaal rondes / (Stops + 1))
df['laps'] = pd.to_numeric(df['laps'], errors='coerce').fillna(0)
df['pit_stops_count'] = df['pit_stops_count'].fillna(0)
df['avg_stint_length'] = df['laps'] / (df['pit_stops_count'] + 1)

# 4. SELECTEREN VAN KOLOMMEN VOOR XGBOOST
# We hebben alleen getallen nodig. Tekst (zoals 'Dutch') moeten we later nog omzetten.

# We selecteren nu ALLES, zodat we later kunnen kiezen wat we weggooien.
final_df = df

# We verwijderen GEEN rijen met dropna(), omdat we alle ruwe data willen zien.
# (Bijv. oude races hebben geen pitstop data, die zouden anders verdwijnen)
# final_df = final_df.dropna()

print("\n--- DATA PREPARED ---")
print(final_df.head())
print(f"\nTotaal aantal rijen om op te trainen: {len(final_df)}")

# 5. OPSLAAN VOOR JE MODEL
output_path = os.path.join(script_dir, 'unprocessed_f1_training_data.csv')
final_df.to_csv(output_path, index=False)
print(f"Bestand '{output_path}' is aangemaakt!")