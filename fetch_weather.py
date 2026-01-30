import pandas as pd
import requests
import time
import os

# 1. INSTELLINGEN
RACES_PATH = '../F1_data_mangement/races.csv'
CIRCUITS_PATH = '../F1_data_mangement/circuits.csv'
OUTPUT_FILE = 'f1_weather_data.csv'

def get_weather_data():
    print("--- START WEER DOWNLOAD SCRIPT ---")
    
    # Controleren of bestanden bestaan
    if not os.path.exists(RACES_PATH) or not os.path.exists(CIRCUITS_PATH):
        print(f"FOUT: Kan invoerbestanden niet vinden op {RACES_PATH} of {CIRCUITS_PATH}")
        return

    print("Races en Circuits laden...")
    races = pd.read_csv(RACES_PATH)
    circuits = pd.read_csv(CIRCUITS_PATH)

    # Samenvoegen om coordinaten (lat/lng) bij de race te krijgen
    df = pd.merge(races, circuits[['circuitId', 'lat', 'lng']], on='circuitId', how='left')
    
    # We hebben alleen races nodig met een datum
    df = df.dropna(subset=['date'])
    
    total_races = len(df)
    print(f"{total_races} races gevonden. Start downloaden via Open-Meteo API...")
    print("Dit duurt even (ca. 0.2s per race) om de API niet te overbelasten.")

    weather_list = []

    for i, row in df.iterrows():
        race_id = row['raceId']
        lat = row['lat']
        lng = row['lng']
        date_str = row['date']
        time_str = row['time']
        
        # Standaard starttijd is 14:00 als er geen tijd bekend is (vaak bij oude races)
        hour = 14
        if pd.notna(time_str) and str(time_str) != '\\N':
            try:
                # Tijdformaat is meestal HH:MM:SS. We pakken het uur.
                hour = int(str(time_str).split(':')[0])
            except ValueError:
                pass

        # API Aanroep (Open-Meteo Archive)
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lng,
            "start_date": date_str,
            "end_date": date_str,
            "hourly": "temperature_2m,precipitation,rain,cloudcover",
            "timezone": "UTC" # F1 data is meestal UTC
        }

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                
                if 'hourly' in data:
                    hourly = data['hourly']
                    # We zoeken de index die overeenkomt met ons startuur (0-23)
                    if hour < len(hourly['time']):
                        weather_entry = {
                            'raceId': race_id,
                            'weather_temp_c': hourly['temperature_2m'][hour],
                            'weather_precip_mm': hourly['precipitation'][hour],
                            'weather_rain_mm': hourly['rain'][hour],
                            'weather_cloud_pct': hourly['cloudcover'][hour]
                        }
                        weather_list.append(weather_entry)
        except Exception as e:
            print(f"Fout bij race {race_id}: {e}")

        # Even wachten (Rate limiting)
        time.sleep(0.2)
        
        if (i + 1) % 50 == 0:
            print(f"Voortgang: {i + 1}/{total_races} races verwerkt...")

    # Opslaan naar CSV
    weather_df = pd.DataFrame(weather_list)
    weather_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nKlaar! {len(weather_df)} weer-records opgeslagen in '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    get_weather_data()
    