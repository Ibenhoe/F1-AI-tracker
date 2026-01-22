#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[F1-CHAMP] F1 RACE PREDICTOR - Interactive Live Simulation
Selecteer een 2024 race en zie realtime predictions per lap
"""

import sys
import os
import io
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from continuous_model_learner import ContinuousModelLearner
from fastf1_data_fetcher import FastF1DataFetcher
import pandas as pd


# 2024 F1 Races (meest recente beschikbare data)
F1_2024_RACES = {
    1: ("Bahrain", "Sakhir"),
    2: ("Saudi Arabia", "Jeddah"),
    3: ("Australia", "Melbourne"),
    4: ("Japan", "Suzuka"),
    5: ("China", "Shanghai"),
    6: ("Miami", "USA"),
    7: ("Monaco", "Monte Carlo"),
    8: ("Canada", "Montreal"),
    9: ("Spain", "Barcelona"),
    10: ("Austria", "Spielberg"),
    11: ("United Kingdom", "Silverstone"),
    12: ("Hungary", "Budapest"),
    13: ("Belgium", "Spa"),
    14: ("Netherlands", "Zandvoort"),
    15: ("Italy", "Monza"),
    16: ("Azerbaijan", "Baku"),
    17: ("Singapore", "Marina Bay"),
    18: ("Austin", "USA"),
    19: ("Mexico", "Mexico City"),
    20: ("Brazil", "Interlagos"),
    21: ("Abu Dhabi", "Yas Island"),
}


def show_race_menu():
    """Toon beschikbare races met betere opmaak"""
    print("\n" + "="*70)
    print("[F1-RACES] F1 2024 SEIZOEN - BESCHIKBARE RACES")
    print("="*70)
    
    # Groepeer per etappe
    print("\n[PART-1] EERSTE HELFT:")
    for num in range(1, 11):
        country, circuit = F1_2024_RACES[num]
        print(f"    {num:2d}. {country:15s} ({circuit})")
    
    print("\n[PART-2] TWEEDE HELFT:")
    for num in range(11, 21):
        country, circuit = F1_2024_RACES[num]
        print(f"    {num:2d}. {country:15s} ({circuit})")
    
    print("\n[FINAL-RND] FINAL:")
    for num in range(21, 22):
        country, circuit = F1_2024_RACES[num]
        print(f"    {num:2d}. {country:15s} ({circuit})")
    
    print("\n" + "-"*70)


def select_race():
    """Vraag gebruiker welke race te simuleren"""
    show_race_menu()
    
    while True:
        try:
            choice = input("\n  Kies race nummer (1-21): ").strip()
            race_num = int(choice)
            
            if race_num not in F1_2024_RACES:
                print(" Ongeldige keuze, probeer opnieuw")
                continue
            
            country, circuit = F1_2024_RACES[race_num]
            print(f"\n Je hebt gekozen: {country} - {circuit} (Race {race_num})")
            return race_num
        
        except ValueError:
            print(" Voer een getal in")


def predict_lap_winner(model, lap_data, lap_num, total_laps, show_all=False):
    """Voorspel met VERBETERD scoring - kijkt naar werkelijke F1 performance
    
    Scoring factors:
    - Position (35%): Huidige positie
    - Pace (35%): Lap time vs average
    - Tire Strategy (20%): Compound + age + pit impact
    - Consistency (10%): Pit stop penalty
    """
    if not lap_data:
        return None
    
    predictions = []
    
    # Bereken lap times gemiddelde om pace gap te bepalen
    lap_times = []
    for driver_data in lap_data:
        lap_time = driver_data.get('lap_time')
        if lap_time:
            try:
                if hasattr(lap_time, 'total_seconds'):
                    lap_sec = lap_time.total_seconds()
                else:
                    lap_sec = float(lap_time)
                if lap_sec > 0:
                    lap_times.append(lap_sec)
            except:
                pass
    
    avg_lap_time = sum(lap_times) / len(lap_times) if lap_times else 0
    
    for driver_data in lap_data:
        driver_id = driver_data.get('driver', '?')
        actual_pos = driver_data.get('position', 20)
        lap_time = driver_data.get('lap_time')
        pit_stop = driver_data.get('is_pit_lap', False)
        tire_compound = driver_data.get('tire_compound', 'UNKNOWN')
        tire_age = driver_data.get('tire_age', 0)
        
        # Model prediction - DIEPER: predict final position
        try:
            pred_pos = model.predict(driver_data)
            if pred_pos is None:
                pred_pos = actual_pos
        except:
            pred_pos = actual_pos
        
        # ===== VERBETERD SCORING SYSTEEM =====
        
        # 1. Position score (35%) - hoe beter de huidige positie
        position_score = max(0, (20 - actual_pos) / 20 * 100)
        
        # 2. Pace score (35%) - hoe dicht bij gemiddelde lap time
        pace_score = 0
        if lap_time and avg_lap_time > 0:
            try:
                if hasattr(lap_time, 'total_seconds'):
                    lap_time_sec = lap_time.total_seconds()
                else:
                    lap_time_sec = float(lap_time)
                time_diff = abs(lap_time_sec - avg_lap_time)
                pace_score = max(0, 100 - (time_diff / avg_lap_time * 100))
            except:
                pace_score = 0
        
        # 3. TIRE STRATEGY score (20%) - beter dan voor
        tire_strategy_score = 50  # baseline
        
        # Tire compound factor (SOFT sneller, HARD duurzamer)
        if tire_compound == 'SOFT':
            tire_strategy_score += 15
        elif tire_compound == 'MEDIUM':
            tire_strategy_score += 10
        elif tire_compound == 'HARD':
            tire_strategy_score += 5
        elif tire_compound == 'INTERMEDIATE':
            tire_strategy_score += 12
        elif tire_compound == 'WET':
            tire_strategy_score -= 5
        
        # Tire age effect (jong = goed, oud = slecht)
        if tire_age and tire_age > 0:
            age_score = max(0, 100 - (tire_age * 5))  # -5% per lap oud
            tire_strategy_score = (tire_strategy_score * 0.7) + (age_score * 0.3)
        
        tire_strategy_score = max(0, min(100, tire_strategy_score))
        
        # 4. Consistency score (10%) - pit stop penalty
        consistency_score = 100 if not pit_stop else 40
        
        # ===== GECOMBINEERDE SCORE =====
        total_accuracy = (
            position_score * 0.35 +
            pace_score * 0.35 +
            tire_strategy_score * 0.20 +
            consistency_score * 0.10
        )
        
        predictions.append({
            'driver': driver_id,
            'actual_pos': actual_pos,
            'pred_pos': int(pred_pos) if pred_pos else actual_pos,
            'accuracy': total_accuracy,
            'pos_score': position_score,
            'pace_score': pace_score,
            'tire_score': tire_strategy_score,
            'pit': pit_stop,
            'tire_compound': tire_compound,
            'tire_age': tire_age
        })
    
    # Sort by accuracy
    predictions.sort(key=lambda x: x['accuracy'], reverse=True)
    
    if show_all:
        return predictions
    else:
        # Voor live updates: alleen top realistic tonen
        top_realistic = []
        for pred in predictions:
            if pred['actual_pos'] <= 10 and pred['accuracy'] > 40:
                top_realistic.append(pred)
        
        return (top_realistic[:5] if len(top_realistic) >= 3 else predictions[:5])


def analyze_tire_strategy(laps_by_number):
    """Analyseer welke band strategie het snelste was"""
    tire_performance = {}  # {compound: [lap_times]}
    
    for lap_num in laps_by_number:
        for driver_data in laps_by_number[lap_num]:
            compound = driver_data.get('tire_compound', 'UNKNOWN')
            lap_time = driver_data.get('lap_time')
            
            if compound not in tire_performance:
                tire_performance[compound] = []
            
            try:
                if hasattr(lap_time, 'total_seconds'):
                    lap_sec = lap_time.total_seconds()
                else:
                    lap_sec = float(lap_time)
                if lap_sec > 0:
                    tire_performance[compound].append(lap_sec)
            except:
                pass
    
    # Bereken gemiddelde per band
    tire_stats = {}
    for compound, times in tire_performance.items():
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            tire_stats[compound] = {
                'avg': avg_time,
                'min': min_time,
                'max': max_time,
                'samples': len(times)
            }
    
    return tire_stats


def analyze_final_predictions(model, final_lap_data):
    """Get all drivers' predictions for final classification"""
    # Use model to get realistic predictions based on pace
    predictions = model.predict_winner(final_lap_data)
    
    result = []
    if 'predictions' in predictions:
        for driver_id, pred_data in predictions['predictions'].items():
            result.append({
                'driver': str(driver_id),
                'pred_pos': int(pred_data.get('predicted_position', 15)),
                'current_pos': int(pred_data.get('current_position', 15)),
                'accuracy': pred_data.get('confidence', 65.0)  # Use realistic confidence (max 85%)
            })
    
    return result


def main():
    print("\n" + "="*70)
    print("[F1-CHAMP] F1 RACE PREDICTOR 2024")
    print("   Live Per-Lap Predictions")
    print("="*70)
    # Create output folder if needed
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)
    # Selecteer race
    race_num = select_race()
    
    # Load data
    print(f"\n[LOAD] Data laden voor race {race_num}...")
    fetcher = FastF1DataFetcher()
    
    if not fetcher.fetch_race(2024, race_num):
        print(" Kon race data niet laden")
        return
    
    race_info = fetcher.get_race_summary()
    print(f"[OK] Geladen: {race_info['event']}")
    
    # Process laps
    print("\n[PROCESS] Laps verwerken...")
    laps_data = fetcher.process_race_laps_streaming()
    
    if not laps_data:
        print(" Geen lap data gevonden")
        return
    
    print(f"[OK] {len(laps_data)} lap records gevonden")
    
    # Group by lap
    laps_by_number = {}
    for lap_data in laps_data:
        lap_num = int(lap_data['lap_number'])
        if lap_num not in laps_by_number:
            laps_by_number[lap_num] = []
        laps_by_number[lap_num].append(lap_data)
    
    sorted_laps = sorted(laps_by_number.keys())
    total_laps = len(sorted_laps)
    
    print(f" Georganiseerd in {total_laps} laps")
    
    # Initialize model
    model = ContinuousModelLearner(learning_decay=0.85, min_samples_to_train=15)
    
    # Pre-train on historical data to improve confidence
    print("\n" + "="*70)
    print("[PRETRAIN] Loading historical F1 race data...")
    print("="*70)
    model.pretrain_on_historical_data('processed_f1_training_data.csv')
    
    print("\n" + "="*70)
    print(f"[RACE-START] LIVE PREDICTION - {race_info['event']}")
    print(f"   Totaal laps: {total_laps}")
    print("="*70)
    
    # Initialize output storage
    predictions_log = []
    
    # Train and predict per lap
    for lap_num in sorted_laps:
        lap_data_list = laps_by_number[lap_num]
        model.add_lap_data(lap_num, lap_data_list)
        
        # Update model
        if lap_num >= 2:
            model.update_model(target_variable='position', epochs=3)
        
        # Show predictions every lap after lap 5
        if lap_num >= 5:
            top5 = predict_lap_winner(model, lap_data_list, lap_num, total_laps)
            
            if top5:
                progress = (lap_num / total_laps) * 100
                print(f"\n[LAP] {lap_num}/{total_laps} ({progress:.0f}%)")
                print("   Top Winners (Position + Pace + Tire Strategy):")
                print("   " + "-"*70)
                
                medals = ['[1st]', '[2nd]', '[3rd]', '     ', '     ']
                lap_prediction = f"LAP {lap_num}: "
                for rank, pred in enumerate(top5):
                    driver = pred['driver']
                    accuracy = pred['accuracy']
                    pos = pred['actual_pos']
                    pace = pred['pace_score']
                    tire_score = pred.get('tire_score', 0)
                    pit = "[PIT]" if pred['pit'] else "     "
                    
                    bar_len = int(accuracy / 5)
                    bar = "#" * bar_len + "-" * (20 - bar_len)
                    
                    print(f"   {medals[rank]} #{rank+1} | Driver {driver:3s} | P:{pos:2.0f} | Pace:{pace:5.1f}% | Tire:{tire_score:5.1f}% | Score:{accuracy:5.1f}% {pit} | {bar}")
                    lap_prediction += f"{rank+1}. Driver {driver} ({accuracy:.1f}%) | "
                
                predictions_log.append(lap_prediction)
    
    # Final classification
    print("\n" + "="*70)
    print("[FINISH] FINAL CLASSIFICATION - ALLE DRIVERS")
    print("="*70)
    
    final_lap_data = laps_by_number[sorted_laps[-1]]
    
    # Get AI predictions for all drivers
    all_predictions = analyze_final_predictions(model, final_lap_data)
    
    # Sort by actual position
    actual_finishers = sorted(final_lap_data, key=lambda x: x.get('position', 999))
    
    print("\nActual Finishers | AI Prediction | Current Pos | Pred Pos | Accuracy:")
    print("-"*70)
    final_results = []
    for i, driver_data in enumerate(actual_finishers, 1):
        driver_id = driver_data.get('driver', '?')
        actual_pos = driver_data.get('position', 999)
        tire_final = driver_data.get('tire_compound', 'UNKNOWN')
        
        # Find prediction for this driver
        pred_data = None
        for pred in all_predictions:
            if pred['driver'] == driver_id:
                pred_data = pred
                break
        
        if pred_data:
            pred_pos = pred_data['pred_pos']
            accuracy = pred_data['accuracy']
            print(f"   {i:2d}. Driver {driver_id:3s} ({tire_final:6s}) | Pos:{actual_pos:2.0f}->{pred_pos:2d} | Current:{actual_pos:2.0f} | Pred:{pred_pos:2d} | {accuracy:5.1f}%")
            final_results.append({
                'pos': i,
                'driver': driver_id,
                'actual': actual_pos,
                'predicted': pred_pos,
                'accuracy': accuracy,
                'tire': tire_final
            })
        else:
            print(f"   {i:2d}. Driver {driver_id:3s} ({tire_final:6s}) | Pos:{actual_pos:2.0f}->?? | Current:{actual_pos:2.0f} | Pred:?? | --.--%")
    
    # TIRE STRATEGY ANALYSIS
    print("\n" + "="*70)
    print("[TIRES] STRATEGY ANALYSIS")
    print("="*70)
    tire_stats = analyze_tire_strategy(laps_by_number)
    print("\nBand Performance (Average Lap Time):")
    print("-"*70)
    
    tire_results = []
    for compound in sorted(tire_stats.keys()):
        stats = tire_stats[compound]
        avg_time = stats['avg']
        min_time = stats['min']
        max_time = stats['max']
        samples = stats['samples']
        
        # Bepaal snelheid relatief
        if compound == 'SOFT':
            speed_indicator = "[SOFT] (Grip)"
        elif compound == 'MEDIUM':
            speed_indicator = "[MED] (Balanced)"
        elif compound == 'HARD':
            speed_indicator = "[HARD] (Durable)"
        else:
            speed_indicator = "[UNK] (Unknown)"
        
        print(f"   {compound:10s} {speed_indicator}: Avg {avg_time:6.2f}s | Min {min_time:6.2f}s | Max {max_time:6.2f}s | Samples: {samples}")
        tire_results.append({
            'compound': compound,
            'avg': avg_time,
            'min': min_time,
            'max': max_time,
            'samples': samples
        })
    
    # Find fastest tire strategy
    if tire_stats:
        fastest_compound = min(tire_stats.items(), key=lambda x: x[1]['avg'])
        print(f"\n[BEST] FASTEST TIRE STRATEGY: {fastest_compound[0]} (Avg {fastest_compound[1]['avg']:.2f}s)")
    
    # DEEP AI ANALYSIS - All drivers current vs predicted
    print("\n" + "="*70)
    print("[AI] DEEP AI ANALYSIS - CURRENT vs PREDICTED FINISH")
    print("="*70)
    print("\nAll Drivers Position Evolution:")
    print("-"*70)
    
    # Sort by current position
    all_predictions_sorted = sorted(all_predictions, key=lambda x: x['current_pos'])
    
    print("Driver | Current | Predicted | Confidence | Pace | Notes")
    print("-"*70)
    for pred in all_predictions_sorted:
        driver_id = pred['driver']
        current_pos = pred['current_pos']
        pred_pos = pred['pred_pos']
        accuracy = pred['accuracy']
        
        # Determine position change
        if pred_pos < current_pos:
            direction = "[UP] GAIN"
        elif pred_pos > current_pos:
            direction = "[DOWN] LOSE"
        else:
            direction = "[SAME] HOLD"
        
        print(f"{driver_id:3s} | {current_pos:7.0f} | {pred_pos:9.0f} | {accuracy:10.1f}% | {accuracy:4.1f}% | {direction}")
    
    print("\n" + "="*70)
    print("[OK] Simulatie voltooid!")
    print("="*70 + "\n")
    
    # Save comprehensive results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_folder, f"race_{race_num:02d}_{timestamp}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"[F1-CHAMP] F1 RACE PREDICTOR - {race_info['event']}\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Race: {race_info['event']}\n")
        f.write(f"Totaal laps: {total_laps}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PREDICTION EVOLUTION PER LAP:\n")
        f.write("-"*70 + "\n")
        for pred in predictions_log:
            f.write(pred + "\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("FINAL CLASSIFICATION - ALLE DRIVERS\n")
        f.write("="*70 + "\n")
        f.write("Driver | Actual Pos | Predicted Pos | Confidence | Tire\n")
        f.write("-"*70 + "\n")
        for result in final_results:
            f.write(f"   {result['pos']:2d}. Driver {result['driver']:3s} | {result['actual']:10.0f} | {result['predicted']:13d} | {result['accuracy']:10.1f}% | {result['tire']}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("TIRE STRATEGY ANALYSIS\n")
        f.write("="*70 + "\n")
        for tire in tire_results:
            f.write(f"{tire['compound']:10s}: Avg {tire['avg']:6.2f}s | Min {tire['min']:6.2f}s | Max {tire['max']:6.2f}s | Samples {tire['samples']}\n")
        
        if tire_results:
            fastest = min(tire_results, key=lambda x: x['avg'])
            f.write(f"\nFastest Tire: {fastest['compound']} ({fastest['avg']:.2f}s)\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("DEEP AI ANALYSIS - CURRENT vs PREDICTED\n")
        f.write("="*70 + "\n")
        f.write("Driver | Current Pos | Predicted Pos | Confidence\n")
        f.write("-"*70 + "\n")
        for pred in all_predictions_sorted:
            f.write(f"   {pred['driver']:3s} | {pred['current_pos']:11.0f} | {pred['pred_pos']:13.0f} | {pred['accuracy']:10.1f}%\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"[SAVE] Output opgeslagen: outputs/race_{race_num:02d}_{timestamp}.txt\n")


if __name__ == "__main__":
    main()
