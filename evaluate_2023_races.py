import pandas as pd
import xgboost as xgb
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ==============================================================================
# 1. DATA VOORBEREIDING (Kopie van main.py logica)
# ==============================================================================
def prepare_data():
    print("Data laden en features bouwen...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "unprocessed_f1_training_data.csv"))
    
    # Filter 2015+
    df = df[df['year'] >= 2015].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    # --- FEATURE ENGINEERING ---
    # 1. Basis Features
    df['driver_experience'] = df.groupby('driverId').cumcount()
    
    # 2. Binning & Encoding
    df['grid_bin'] = pd.cut(df['grid'], bins=[-1, 1, 3, 10, 15, 25], labels=['Pole', 'Top3', 'Points', 'Midfield', 'Back'])
    le = LabelEncoder()
    df['grid_bin_code'] = le.fit_transform(df['grid_bin'].astype(str))
    
    # 3. Team Continuity (Simpel)
    df['team_continuity_id'] = df['constructorId'] # Voor deze test houden we het simpel

    # 4. Rolling Averages (Vorm)
    def calculate_rolling_avg(group, window=5):
        return group.shift(1).rolling(window, min_periods=1).mean()

    df['driver_recent_form'] = df.groupby('driverId')['positionOrder'].transform(calculate_rolling_avg).fillna(12)
    df['constructor_recent_form'] = df.groupby('constructorId')['positionOrder'].transform(calculate_rolling_avg).fillna(12)
    df['driver_vs_team_form'] = df['constructor_recent_form'] - df['driver_recent_form']

    # --- PARSE FASTEST LAP (Nodig voor Circuit Speed Index) ---
    def parse_fastest_lap(t_str):
        if pd.isna(t_str) or str(t_str).strip() == '\\N': return np.nan
        try:
            parts = str(t_str).split(':')
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(t_str)
        except:
            return np.nan

    if 'fastestLapTime' in df.columns:
        df['fastest_lap_seconds'] = df['fastestLapTime'].apply(parse_fastest_lap)
    else:
        df['fastest_lap_seconds'] = np.nan

    # 5. Circuit Stats (Expanding Mean)
    df['circuit_speed_index'] = df.groupby('circuitId')['fastest_lap_seconds'].transform(lambda x: x.expanding().mean().shift(1)).fillna(90.0)
    df['circuit_overtake_index'] = df.groupby('circuitId')['grid'].transform(lambda x: (x - df['positionOrder']).abs().expanding().mean().shift(1)).fillna(2.0)

    # 6. Weather Confidence (Nat vs Droog Skill)
    df['is_wet_race'] = df['weather_precip_mm'] > 0.1
    wet_avg = df[df['is_wet_race']].groupby('driverId')['positionOrder'].transform(lambda x: x.expanding().mean().shift(1))
    dry_avg = df[~df['is_wet_race']].groupby('driverId')['positionOrder'].transform(lambda x: x.expanding().mean().shift(1))
    df['avg_finish_wet'] = wet_avg.fillna(12)
    df['avg_finish_dry'] = dry_avg.fillna(12)
    df['weather_confidence_diff'] = df['avg_finish_wet'] - df['avg_finish_dry']

    # 7. Aggression & Overtake
    df['positions_gained'] = df['grid'] - df['positionOrder']
    df['driver_overtake_rate'] = df.groupby('driverId')['positions_gained'].transform(calculate_rolling_avg).fillna(0)
    
    if 'fastestLapTime' in df.columns:
        # Simpele parse
        df['fastest_lap_rank'] = df.groupby('raceId')['rank'].rank(method='min')
        df['aggression_score'] = df['positionOrder'] - df['fastest_lap_rank']
        df['driver_aggression'] = df.groupby('driverId')['aggression_score'].transform(calculate_rolling_avg).fillna(0)
    else:
        df['driver_aggression'] = 0

    # 8. Punten vooraf
    df['points_before_race'] = (df.groupby(['year', 'driverId'])['points'].cumsum() - df['points']).fillna(0)
    
    # 9. Grid Penalty
    df['quali_pos_filled'] = df['quali_position'].fillna(df['grid'])
    df['grid_penalty'] = df['grid'] - df['quali_pos_filled']

    # 10. Weer imputatie
    if 'weather_temp_c' in df.columns:
        df['weather_temp_c'] = df['weather_temp_c'].fillna(df['weather_temp_c'].median())
        df['weather_precip_mm'] = df['weather_precip_mm'].fillna(0.0)
    
    return df

# ==============================================================================
# 2. EVALUATIE FUNCTIE
# ==============================================================================
def evaluate_race(df, year, circuit_id, race_name):
    print(f"\n{'='*60}")
    print(f" ANALYSE: {race_name} {year}")
    print(f"{'='*60}")

    # 1. Selecteer de specifieke race
    target_race = df[(df['year'] == year) & (df['circuitId'] == circuit_id)]
    if len(target_race) == 0:
        print("Race niet gevonden in dataset!")
        return

    race_date = target_race.iloc[0]['date']
    print(f"Datum: {race_date.date()} | Circuit ID: {circuit_id}")
    
    # 2. Split Data (Train = Alles VOOR deze datum)
    train_mask = df['date'] < race_date
    test_mask = (df['year'] == year) & (df['circuitId'] == circuit_id)
    
    # Filter alleen finishers voor training (om crashes niet als 'snelheid' te leren)
    # Maar voor test willen we iedereen voorspellen
    train_df = df[train_mask].dropna(subset=['positionOrder'])
    test_df = df[test_mask].copy()

    # Features
    features = [
        'grid', 'grid_penalty', 'driver_experience', 'grid_bin_code',
        'driver_recent_form', 'constructor_recent_form', 'driver_vs_team_form',
        'circuit_speed_index', 'circuit_overtake_index',
        'weather_temp_c', 'weather_precip_mm', 'weather_confidence_diff',
        'driver_aggression', 'driver_overtake_rate', 'points_before_race'
    ]

    X_train = train_df[features]
    y_train = train_df['positionOrder']
    
    X_test = test_df[features]
    
    # Actuals (voor vergelijking)
    actuals = test_df[['driverRef', 'positionOrder', 'grid', 'status']].copy()
    actuals['driverRef'] = actuals['driverRef'].str.capitalize()

    # 3. Train Model (Weighted: P1 is belangrijker)
    weights = (22 - y_train).clip(lower=1)
    
    model = xgb.XGBRegressor(
        n_estimators=200, 
        max_depth=4, 
        learning_rate=0.05, 
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train, y_train, sample_weight=weights)
    
    # --- XGBOOST DNF MODEL (Simulatie) ---
    # In main.py hebben we een apart DNF model. Hier simuleren we dat simpel.
    # We trainen op alle data (niet alleen finishers) voor DNF
    train_df_all = df[train_mask].copy()
    # DNF = Status niet finished
    y_dnf_all = ~train_df_all['statusId'].isin([1, 11, 12, 13, 14]).astype(int) # Simplificatie statusId
    X_train_all = train_df_all[features]
    
    # (Voor deze evaluatie houden we het even bij puur positie voorspelling om de code niet te complex te maken,
    #  maar we voegen wel de NN Classifier toe voor de volledigheid van je vraag)

    # --- NIEUW: NEURAL NETWORK (MLP) ---
    # NN heeft geschaalde data nodig (mean=0, std=1)
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median') # Veiligheid
    
    # Pipeline: Impute -> Scale
    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_scaled = scaler.transform(imputer.transform(X_test))

    nn_model = MLPRegressor(
        hidden_layer_sizes=(64, 32), # 2 lagen met neuronen
        activation='relu',
        solver='adam',
        max_iter=2000, # Genoeg tijd geven om te leren
        random_state=42
    )
    nn_model.fit(X_train_scaled, y_train)
    
    # NN Crash Model (Op alle data)
    X_train_all_scaled = scaler.transform(imputer.transform(X_train_all))
    nn_class = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    nn_class.fit(X_train_all_scaled, y_dnf_all)

    # 4. Predict (Beide modellen)
    preds = model.predict(X_test)
    nn_preds = nn_model.predict(X_test_scaled)
    
    # Crash kansen (NN)
    nn_crash_prob = nn_class.predict_proba(X_test_scaled)[:, 1]
    
    # --- NIEUW: ENSEMBLE (GEMIDDELDE) ---
    ens_preds = (preds + nn_preds) / 2

    actuals['Predicted_Raw'] = preds
    actuals['Predicted_Pos'] = actuals['Predicted_Raw'].rank().astype(int)
    
    actuals['NN_Raw'] = nn_preds
    # Als NN denkt dat crash kans > 50% is, zet onderaan (99)
    actuals['NN_Pos_Raw'] = np.where(nn_crash_prob > 0.5, 99, actuals['NN_Raw'])
    # Ranken op basis van de nieuwe score (met crashes onderaan)
    actuals['NN_Pos'] = actuals['NN_Pos_Raw'].rank().astype(int)
    
    actuals['Ens_Raw'] = ens_preds
    actuals['Ens_Pos'] = actuals['Ens_Raw'].rank().astype(int)
    
    # 5. Resultaten Tabel
    actuals['Error'] = (actuals['Predicted_Pos'] - actuals['positionOrder']).abs()
    actuals['NN_Error'] = (actuals['NN_Pos'] - actuals['positionOrder']).abs()
    actuals['Ens_Error'] = (actuals['Ens_Pos'] - actuals['positionOrder']).abs()
    actuals = actuals.sort_values(by='Predicted_Pos')
    
    print(f"\n--- VERGELIJKING: XGBOOST vs NEURAL NETWORK vs ENSEMBLE ---")
    print(f"{'Driver':<15} | {'XGB':<3} | {'NN':<3} | {'ENS':<3} | {'Act':<3} | {'Grid':<4} | {'XGB Err':<7} | {'NN Err':<7} | {'ENS Err':<7} | {'Status'}")
    print("-" * 105)
    
    total_error_xgb = 0
    total_error_nn = 0
    total_error_ens = 0

    for _, row in actuals.iterrows():
        driver = row['driverRef']
        pred = row['Predicted_Pos']
        nn_pred = row['NN_Pos']
        ens_pred = row['Ens_Pos']
        act = row['positionOrder']
        grid = row['grid']
        status = row['status']
        
        # DNF handling in display
        if "Finished" not in str(status) and "+" not in str(status):
            act_str = "DNF"
            error_str = "-"
            nn_error_str = "-"
            ens_error_str = "-"
        else:
            act_str = str(int(act))
            error = abs(pred - int(act))
            nn_error = abs(nn_pred - int(act))
            ens_error = abs(ens_pred - int(act))
            
            total_error_xgb += error
            total_error_nn += nn_error
            total_error_ens += ens_error
            
            error_str = str(error)
            nn_error_str = str(nn_error)
            ens_error_str = str(ens_error)
            
            # Markeer grote fouten
            if error > 5: error_str += " ⚠️"
            if nn_error > 5: nn_error_str += " ⚠️"
            if ens_error > 5: ens_error_str += " ⚠️"
        
        print(f"{driver:<15} | {pred:<3} | {nn_pred:<3} | {ens_pred:<3} | {act_str:<3} | {grid:<4} | {error_str:<7} | {nn_error_str:<7} | {ens_error_str:<7} | {status}")

    print("-" * 105)
    print(f"Totale Fout XGBoost:       {total_error_xgb}")
    print(f"Totale Fout Neural Net:    {total_error_nn}")
    print(f"Totale Fout Ensemble:      {total_error_ens}")
    
    best_score = min(total_error_xgb, total_error_nn, total_error_ens)
    if best_score == total_error_ens:
        print("🏆 CONCLUSIE: Het Ensemble model (Combinatie) wint!")
    elif best_score == total_error_nn:
        print("🏆 CONCLUSIE: Het Neural Network wint deze ronde!")
    else:
        print("🏆 CONCLUSIE: XGBoost is nog steeds de kampioen.")
    
    # 6. Analyse van de missers
    print("\n--- ANALYSE VAN DE MODEL FOUTEN ---")
    if race_name == "Zandvoort":
        print("CONTEXT: Zandvoort 2023 was een CHAOS race (Regen + Rode Vlag).")
        print("Verwachting: Het model zal moeite hebben met verrassingen zoals Gasly (P3) of Albon (P8).")
        print("Reden: Het model leunt op 'Grid' en 'Droge Vorm'. In regen telt strategie en geluk zwaarder.")
    elif race_name == "Monza":
        print("CONTEXT: Monza 2023 was een SNELHEID race (Droog + Weinig incidenten).")
        print("Verwachting: Het model zou hier heel accuraat moeten zijn.")
        print("Reden: Op Monza wint meestal de snelste auto (Red Bull). Weinig random factoren.")

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    df = prepare_data()
    
    # ZANDVOORT 2023 (Circuit 39)
    # Werkelijkheid: Verstappen (1), Alonso (2), Gasly (3), Perez (4)
    evaluate_race(df, 2023, 39, "Zandvoort")
    
    # MONZA 2023 (Circuit 14)
    # Werkelijkheid: Verstappen (1), Perez (2), Sainz (3), Leclerc (4)
    evaluate_race(df, 2023, 14, "Monza")
'''
### Wat gaat dit script je laten zien?

1.  **Bij Monza (Orde):**
    *   Ik verwacht dat je model **Sainz en Leclerc** goed voorspelt (Ferrari is altijd snel op Monza).
    *   De foutmarge zou laag moeten zijn (bijv. totaal < 30).

2.  **Bij Zandvoort (Chaos):**
    *   Het model zal waarschijnlijk **Gasly (P3)** compleet missen. Alpine was dat seizoen matig, en Gasly startte P12. Het model zal hem waarschijnlijk rond P10-P12 voorspellen.
    *   Dit is **bewijs** voor je thesis: *"AI modellen gebaseerd op historische data falen in chaotische weersomstandigheden omdat ze geen rekening kunnen houden met live strategie-beslissingen."*

Run het script en kijk naar de kolom `Error`. Waar staan de ⚠️ tekens? Dat zijn je discussiepunten!

<!--
[PROMPT_SUGGESTION]Can you explain why the model missed Gasly's podium in Zandvoort so badly?[/PROMPT_SUGGESTION]
[PROMPT_SUGGESTION]How can I add a 'Chaos Factor' feature to the model to handle races like Zandvoort better?[/PROMPT_SUGGESTION]
-->

'''