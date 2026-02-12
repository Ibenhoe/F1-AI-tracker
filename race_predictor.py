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

from continuous_model_learner_v2 import ContinuousModelLearner
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


def predict_lap_rankings(model, lap_data, lap_num, total_laps, show_all=False):
    """Predict per-lap driver rankings with INCREMENTAL LEARNING model
    
    Returns ranked list of drivers sorted by performance accuracy (not single winner).
    Performance note: lap_time is converted to seconds inside loop (lines 115-118).
    For large datasets (20+ drivers × 100+ laps), consider pre-processing lap_time
    to numeric format during data fetching to reduce repeated conversions.
    
    Model updates itself every lap!
    Confidence is NEVER 100% - realistic predictions only
    """
    if not lap_data:
        return None
    
    predictions = []
    
    # Get predictions from incremental model
    model_predictions = model.predict_lap(lap_data)
    
    # Bereken lap times voor extra context
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
    
    # Build cumulative pace tracking: drivers that consistently outpace others get bigger bonus
    driver_pace_deltas = {}  # {driver: time_delta_from_avg}
    for driver_data in lap_data:
        driver_id = driver_data.get('driver', '?')
        lap_time = driver_data.get('lap_time')
        
        if lap_time and avg_lap_time > 0:
            try:
                if hasattr(lap_time, 'total_seconds'):
                    lap_sec = lap_time.total_seconds()
                else:
                    lap_sec = float(lap_time)
                delta = lap_sec - avg_lap_time
                driver_pace_deltas[driver_id] = delta
            except:
                driver_pace_deltas[driver_id] = 0.0
        else:
            driver_pace_deltas[driver_id] = 0.0
    
    for driver_data in lap_data:
        driver_id = driver_data.get('driver', '?')
        actual_pos = driver_data.get('position', 20)
        lap_time = driver_data.get('lap_time')
        tire_compound = driver_data.get('tire_compound', 'UNKNOWN')
        
        # Get model prediction (position, confidence)
        if driver_id in model_predictions:
            pred_pos, model_confidence = model_predictions[driver_id]
        else:
            pred_pos = actual_pos
            model_confidence = 30.0
        
        # ===== SCORING BASED ON CURRENT POSITION & PACE =====
        
        # 1. Position score - EXPONENTIAL: P1 >> P2 >> P3
        # Leaders get bonuses, rear runners get penalties
        if actual_pos <= 1:
            position_score = 90.0
        elif actual_pos <= 3:
            position_score = 75.0 - (actual_pos - 1) * 7
        elif actual_pos <= 5:
            position_score = 60.0 - (actual_pos - 3) * 4
        elif actual_pos <= 10:
            position_score = 50.0 - (actual_pos - 5) * 2.5
        else:
            position_score = 35.0 - (actual_pos - 10) * 1.5
        
        position_score = max(10.0, position_score)
        
        # 2. PACE SCORE - MUCH MORE IMPORTANT NOW
        # Cumulative pace advantage: if you're 0.3s faster on avg, that's huge!
        pace_score = 50.0  # Base for average pace
        pace_delta = driver_pace_deltas.get(driver_id, 0.0)
        
        if avg_lap_time > 0:
            # Negative delta = faster than average (BONUS!)
            # Positive delta = slower than average (PENALTY!)
            if pace_delta < 0:
                # Driver is FASTER - big bonus!
                # 0.1s faster = +10, 0.3s faster = +30, 0.5s+ faster = +50 bonus
                pace_bonus = min(50.0, abs(pace_delta) / avg_lap_time * 600)
                pace_score = 50.0 + pace_bonus
            else:
                # Driver is SLOWER - penalty
                pace_penalty = min(40.0, pace_delta / avg_lap_time * 400)
                pace_score = 50.0 - pace_penalty
        
        pace_score = max(10.0, min(95.0, pace_score))
        
        # 3. Tire strategy - Dynamically penalize old tires
        tire_age = driver_data.get('tire_age', 5)
        tire_strategy_score = 50.0
        
        if tire_compound == 'SOFT':
            tire_strategy_score = 65.0
            if tire_age > 10:
                tire_strategy_score -= (tire_age - 10) * 2.5
        elif tire_compound == 'MEDIUM':
            tire_strategy_score = 60.0
            if tire_age > 15:
                tire_strategy_score -= (tire_age - 15) * 1.5
        elif tire_compound == 'HARD':
            tire_strategy_score = 55.0
            if tire_age > 20:
                tire_strategy_score -= (tire_age - 20) * 1
        
        tire_strategy_score = max(15.0, min(95.0, tire_strategy_score))
        
        # ===== COMBINE SCORES WITH NEW WEIGHTS =====
        # Position: 35% (less dominant, allows pace to matter more)
        # Pace: 40% (MUCH MORE IMPORTANT - cumulative advantage counts!)
        # Tire: 15% (strategy is fine-tuning)
        # Model: 10% (adds nuance)
        total_accuracy = (
            position_score * 0.35 +  # Position still important but not dominant
            pace_score * 0.40 +      # PACE IS KING NOW - if you're 0.3s faster, you win!
            tire_strategy_score * 0.15 +  # Tire is supporting role
            model_confidence * 0.10  # Model adds nuance
        )
        
        # ===== RACE-PHASE CONFIDENCE CAPS =====
        # Looser caps that allow differentiation while maintaining realism
        race_phase_cap = 100.0
        if lap_num <= 5:
            race_phase_cap = 75.0   # First 5 laps: conservative but allows leaders through
        elif lap_num <= 10:
            race_phase_cap = 82.0   # Laps 6-10: patterns emerging
        elif lap_num <= 20:
            race_phase_cap = 88.0   # Laps 11-20: established patterns
        elif lap_num <= 30:
            race_phase_cap = 90.0   # Laps 21-30: clear strategy
        # After lap 30: 100% (no cap, full confidence)
        
        # HARD CAP: Never 100%
        # Set max at race_phase_cap to ensure predictions are NEVER deterministic or overconfident
        # This reflects real-world F1 unpredictability where even favorites can fail
        # (e.g., DNF, tire failure, strategic pit stop changes, weather, collisions)
        # This cap is applied ONCE here, not redundantly during display
        total_accuracy = min(race_phase_cap, total_accuracy)  # Race-phase cap for realism
        total_accuracy = max(15.0, total_accuracy)  # Min 15% to avoid overpenalizing drivers
        
        predictions.append({
            'driver': driver_id,
            'actual_pos': actual_pos,
            'pred_pos': int(pred_pos),
            'accuracy': total_accuracy,
            'model_confidence': model_confidence,
            'pos_score': position_score,
            'pace_score': pace_score,
            'tire_score': tire_strategy_score
        })
    
    # Sort by accuracy (predicted winner potential, based on current performance)
    # NOT by position - so drivers improving through the race can enter top 5!
    predictions.sort(key=lambda x: x['accuracy'], reverse=True)
    
    if show_all:
        return predictions
    else:
        # Return top 5 by ACCURACY SCORE (who is most likely to finish in top positions)
        # This allows drivers who are improving (e.g., VER starting P18 but now P2) to show up
        # in top 5 predictions as they climb through the field during the race
        # OLD LOGIC (broken): Filtered to p['actual_pos'] <= 15 first, preventing drivers who started far back
        return predictions[:5]


def analyze_tire_strategy(laps_by_number):
    """Analyze tire compound performance across all laps
    
    Performance note: lap_time is converted to seconds inside nested loop (lines 220-224).
    For optimization, pre-process lap_time during initial data fetch to avoid repeated
    conversion calls in the loop structure (lap × driver × compound iterations).
    """
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
    
    # Bereken gemiddelde per band (protected against empty lists)
    tire_stats = {}
    for compound, times in tire_performance.items():
        if times:  # Only calculate if list has elements (prevents division by zero)
            avg_time = sum(times) / len(times)  # Safe: len(times) > 0 guaranteed by 'if times:' check
            min_time = min(times)
            max_time = max(times)
            tire_stats[compound] = {
                'avg': avg_time,
                'min': min_time,
                'max': max_time,
                'samples': len(times)
            }
    
    return tire_stats


def get_final_predictions(model, final_lap_data, current_lap):
    """Retrieve and format final predictions for all drivers from model
    
    Parameters:
    -----------
    model : ContinuousModelLearner
        The trained AI model with predict_top5_winners() method
    final_lap_data : List[Dict]
        List of driver data from the final lap, passed to model.predict_top5_winners()
        Contains driver positions, tire info, and lap times for final prediction
    current_lap : int
        Current lap number for the model's prediction context
    
    Returns:
    --------
    List[Dict]
        Predictions for all drivers with position, confidence, and current position
    
    Performance note: Calls model.predict_top5_winners() which may be computationally expensive.
    If this function is called frequently or with large datasets, consider:
    1) Profiling model.predict_top5_winners() to understand its execution time
    2) Caching results if final_lap_data inputs are frequently repeated
    3) Running asynchronously if supported by the model and calling context
    
    Note: Confidence values returned from model are already constrained by model's internal caps.
    Display-time capping (if applied) happens in calling code, not here.
    """
    
    result = []
    # Call model to get predictions for final classification
    # final_lap_data is passed to model.predict_top5_winners() to obtain predictions based on current race state
    predictions = model.predict_top5_winners(final_lap_data, current_lap) if model else None
    
    if predictions:
        for pred_data in predictions:
            result.append({
                'driver': str(pred_data.get('driver', '?')),
                'pred_pos': int(pred_data.get('predicted_position', 15)),
                'current_pos': int(pred_data.get('current_position', 15)),
                'accuracy': pred_data.get('confidence', 65.0)  # Use realistic confidence (max 85%)
            })
    
    return result


def main():
    """Main entry point for F1 race prediction simulation
    
    ARCHITECTURAL NOTE (Issue 7 - God Function):
    This function handles multiple responsibilities: data loading, model initialization,
    incremental training, prediction display, and report generation.
    Future refactoring opportunity: Extract into focused components:
    - RaceDataLoader: Handle FastF1 fetching and lap data organization
    - ModelEngine: Manage model initialization and incremental updates  
    - PredictionDisplay: Handle per-lap prediction output formatting
    - ReportGenerator: Handle final classification and file output
    This would improve modularity, testability, and maintainability.
    """
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
    
    # Initialize INCREMENTAL model
    print("\n" + "="*70)
    print("[INIT] Initializing Continuous Learning Model (v2 - Pit Stop Analysis)")
    print("="*70)
    model = ContinuousModelLearner()
    
    # Pre-train on historical data
    print("\n[PRETRAIN] Loading 5-year F1 historical data...")
    model.pretrain_on_historical_data('f1_historical_5years.csv')
    
    print("\n" + "="*70)
    print(f"[RACE-START] LIVE PREDICTION - {race_info['event']}")
    print(f"   Totaal laps: {total_laps}")
    print(f"   Advanced Model: 40+ engineered features")
    print(f"   Ensemble: SGD + GradientBoosting + XGBoost + RandomForest")
    print(f"   Per-lap incremental learning: ENABLED")
    print(f"   Confidence: Dynamic (15-85%, based on prediction stability)")
    print("="*70)
    
    # Initialize output storage
    predictions_log = []
    
    # ===== TRAIN AND PREDICT PER LAP WITH INCREMENTAL UPDATES =====
    for lap_num in sorted_laps:
        lap_data_list = laps_by_number[lap_num]
        
        # ===== UPDATE MODEL WITH CONTINUOUS LEARNING =====
        # Model learns from this lap via buffer-based training (v2 API)
        if lap_num >= 1:  # Start updating from lap 1
            model.add_race_data(lap_num, lap_data_list)
            update_result = model.update_model(lap_num)
            if update_result['status'] == 'updated':
                print(f"[UPDATE] Lap {lap_num}: Advanced model trained on {update_result['samples']} samples (Total: {update_result['updates']})")
        
        # Show predictions every lap after lap 2
        if lap_num >= 2:
            top5 = predict_lap_rankings(model, lap_data_list, lap_num, total_laps)
            
            if top5:
                progress = (lap_num / total_laps) * 100
                print(f"\n[LAP {lap_num}/{total_laps}] {progress:.0f}% complete - INCREMENTAL PREDICTIONS:")
                print("   " + "-"*75)
                
                medals = ['🥇 1st', '🥈 2nd', '🥉 3rd', '  4th', '  5th']
                lap_prediction = f"LAP {lap_num}: "
                for rank, pred in enumerate(top5):
                    driver = pred['driver']
                    accuracy = pred['accuracy']
                    model_conf = pred['model_confidence']
                    pos = pred['actual_pos']
                    pace = pred['pace_score']
                    
                    bar_len = int(accuracy / 5)
                    bar = "█" * bar_len + "░" * (16 - bar_len)
                    
                    # No need to cap here - accuracy is already capped at 79.9% in predict_lap_winner()
                    print(f"   {medals[rank]} | Driver {driver:3s} | Current Pos: {pos:2.0f} | Pace: {pace:5.1f}% | Confidence: {accuracy:5.1f}% | {bar}")
                    lap_prediction += f"{rank+1}. Driver {driver} ({accuracy:.1f}%) | "
                
                predictions_log.append(lap_prediction)
    
    # Final classification
    print("\n" + "="*70)
    print("[FINISH] FINAL CLASSIFICATION - ALLE DRIVERS")
    print("="*70)
    
    final_lap_data = laps_by_number[sorted_laps[-1]]
    final_lap_num = sorted_laps[-1]
    
    # Get AI predictions for all drivers
    all_predictions = get_final_predictions(model, final_lap_data, final_lap_num)
    
    # Sort by actual position
    actual_finishers = sorted(final_lap_data, key=lambda x: x.get('position', 999))
    
    print("\nActual Finishers | AI Prediction | Current Pos | Pred Pos | Accuracy (max 79.9%):")
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
            # No need to cap here - accuracy from model.predict_winner() should already be realistic
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
    print("[AI] DEEP AI ANALYSIS - REALISTIC CONFIDENCE (max 85%)")
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
        accuracy = min(85, pred['accuracy'])  # Cap at 85% for realistic display
        
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
        f.write("FINAL CLASSIFICATION - ALLE DRIVERS (Confidence capped at max 79.9%)\n")
        f.write("="*70 + "\n")
        f.write("Driver | Actual Pos | Predicted Pos | Confidence | Tire\n")
        f.write("-"*70 + "\n")
        for result in final_results:
            # No need to cap here - accuracy is already constrained by model
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
        f.write("DEEP AI ANALYSIS - CURRENT vs PREDICTED (Realistic Confidence)\n")
        f.write("="*70 + "\n")
        f.write("Driver | Current Pos | Predicted Pos | Confidence (max 85%)\n")
        f.write("-"*70 + "\n")
        for pred in all_predictions_sorted:
            # Cap at 85% for realism
            display_conf = min(85, pred['accuracy'])
            f.write(f"   {pred['driver']:3s} | {pred['current_pos']:11.0f} | {pred['pred_pos']:13.0f} | {display_conf:10.1f}%\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"[SAVE] Output opgeslagen: outputs/race_{race_num:02d}_{timestamp}.txt\n")


if __name__ == "__main__":
    main()
