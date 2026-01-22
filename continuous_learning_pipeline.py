"""
Continuous Learning Pipeline
Verbindt FastF1 data, lap processor, en model voor real-time updates
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional
from datetime import datetime
from fastf1_data_fetcher import FastF1DataFetcher
from lap_processor import LapDataProcessor, RaceRealtimeSimulator
from continuous_model_learner import ContinuousModelLearner
import json
from queue import Queue
import threading


class LapDataQueue:
    """
    Thread-safe queue voor lap data
    Handig voor concurrent reading/writing
    """
    
    def __init__(self, maxsize: int = 100):
        self.queue = Queue(maxsize=maxsize)
        self.all_items = []  # Keep voor reference
        
    def put(self, item: Dict):
        """Voeg lap data toe aan queue"""
        self.queue.put(item)
        self.all_items.append(item)
    
    def get(self, block: bool = True, timeout: float = None) -> Dict:
        """Haal lap data uit queue"""
        return self.queue.get(block=block, timeout=timeout)
    
    def size(self) -> int:
        """Huidige queue size"""
        return self.queue.qsize()
    
    def empty(self) -> bool:
        """Is queue leeg?"""
        return self.queue.empty()


class ModelPredictor:
    """
    Wrapper voor model prediction
    Handelt errors af en logging
    """
    
    def __init__(self, model):
        self.model = model
        self.predictions_log = []
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Maak predictions met error handling
        """
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            print(f"[ERROR] Prediction error: {e}")
            return None
    
    def predict_proba(self, X: np.ndarray):
        """Probabilities (als model dit support)"""
        try:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
        except:
            pass
        return None
    
    def log_prediction(self, lap_number: int, driver: str, prediction: float):
        """Log prediction voor analyse"""
        self.predictions_log.append({
            'lap_number': lap_number,
            'driver': driver,
            'prediction': float(prediction),
            'timestamp': datetime.now().isoformat()
        })


class ContinuousLearningPipeline:
    """
    Hoofd pipeline voor continuous learning
    
    Flow:
    1. Fetch race data van FastF1
    2. Lap-voor-lap data extracten
    3. Features engineeren
    4. Stuur naar model voor predictions
    5. Log results voor retraining
    """
    
    def __init__(self, model=None, enable_continuous_learning: bool = True):
        self.fetcher = FastF1DataFetcher()
        self.processor = LapDataProcessor()
        self.model = ModelPredictor(model) if model else None
        
        # NEW: Continuous model learner
        self.enable_continuous_learning = enable_continuous_learning
        self.model_learner = ContinuousModelLearner(model=model) if enable_continuous_learning else None
        
        self.data_queue = LapDataQueue()
        self.predictions = []
        self.performance_log = []
        
    
    def load_race(self, year: int, round_number: int) -> bool:
        """
        Load race van FastF1
        """
        return self.fetcher.fetch_race(year, round_number)
    
    
    def extract_lap_data(self) -> List[Dict]:
        """
        Extract alle lap data van race
        """
        return self.fetcher.process_race_laps_streaming()
    
    
    def process_and_predict_laps(self, 
                                laps_data: List[Dict],
                                callback: Optional[Callable] = None,
                                max_laps: int = None) -> Dict:
        """
        Verwerk race lap-voor-lap, maak predictions
        
        Args:
            laps_data: Alle lap data
            callback: Functie die per lap wordt aangeroepen
            max_laps: Limit voor testing (None = alles)
            
        Returns:
            Summary dict met results
        """
        
        print("\n" + "="*70)
        print("[RACE-START] CONTINUOUS LEARNING PIPELINE START")
        print("="*70)
        
        # Initialize
        race_info = self.fetcher.get_race_summary()
        print(f"\n[LAP-DATA] Race: {race_info.get('event')} ({race_info.get('year')})")
        
        drivers = list(set([l['driver'] for l in laps_data]))
        self.processor.initialize_driver_history(drivers)
        print(f"[DRIVERS] Drivers: {len(drivers)}")
        
        # Group by lap
        laps_by_number = {}
        for lap_data in laps_data:
            lap_num = int(lap_data['lap_number'])
            if lap_num not in laps_by_number:
                laps_by_number[lap_num] = []
            laps_by_number[lap_num].append(lap_data)
        
        total_laps = len(laps_by_number)
        print(f"🔢 Total laps: {total_laps}\n")
        
        if max_laps:
            total_laps = min(total_laps, max_laps)
        
        # Process per lap
        lap_count = 0
        predictions_made = 0
        
        for lap_num in sorted(laps_by_number.keys())[:total_laps]:
            lap_data_list = laps_by_number[lap_num]
            
            # Process lap
            processed_df = self.processor.process_full_lap_round({
                'lap_number': lap_num,
                'drivers_data': lap_data_list
            })
            
            # Prepare voor model
            X, features_df = self.processor.prepare_for_model(processed_df)
            
            # ====== NEW: Add to model learner ======
            if self.model_learner:
                self.model_learner.add_lap_data(lap_num, lap_data_list)
                
                # Update model AFTER EVERY LAP (for continuous learning)
                if lap_num >= 2:  # Need minimum 2 laps worth data
                    learning_result = self.model_learner.update_model(
                        target_variable='position',
                        epochs=5  # Fewer epochs per update since doing it more often
                    )
                    if learning_result['status'] != 'skipped':
                        print(f"     [PROCESS] Model updated: {learning_result['status']} | "
                              f"Samples: {learning_result.get('samples_used', 0)} | "
                              f"MAE: {learning_result.get('mae', 0):.3f}")
            # ========================================
            
            # Predictions
            predictions = None
            if self.model:
                predictions = self.model.predict(X)
                if predictions is not None:
                    predictions_made += len(predictions)
            elif self.model_learner:
                # Use continuous learner for predictions if no static model
                predictions = self.model_learner.predict_lap(lap_data_list)
                if predictions:
                    predictions_made += len([p for p in predictions.values() if p is not None])
            
            # Log data
            lap_result = {
                'lap_number': lap_num,
                'num_drivers': len(lap_data_list),
                'processed_data': processed_df,
                'predictions': predictions,
                'X': X
            }
            
            self.performance_log.append(lap_result)
            
            # Callback (voor custom processing)
            if callback:
                callback(lap_result)
            
            # Progress
            lap_count += 1
            if lap_count % 10 == 0 or lap_count == 1:
                status = f"  ✓ Lap {lap_num}/{total_laps}"
                if predictions is not None:
                    status += f" - {len(predictions)} predictions"
                print(status)
        
        # Summary
        summary = {
            'race_year': race_info.get('year'),
            'race_round': race_info.get('round'),
            'event': race_info.get('event'),
            'total_drivers': len(drivers),
            'total_laps_processed': lap_count,
            'total_predictions': predictions_made,
            'has_model': self.model is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        print("\n" + "="*70)
        print("[OK] PIPELINE COMPLETE")
        print("="*70)
        print(f"  Laps processed: {summary['total_laps_processed']}")
        print(f"  Predictions made: {summary['total_predictions']}")
        print(f"  Drivers tracked: {summary['total_drivers']}")
        
        return summary
    
    
    def export_predictions(self, filename: str = None) -> str:
        """
        Export prediction log
        """
        if not self.performance_log:
            print("Geen predictions om te exporteren")
            return None
        
        if filename is None:
            filename = f"f1_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Flatten performance log
        rows = []
        for lap_result in self.performance_log:
            lap_num = lap_result['lap_number']
            processed_df = lap_result['processed_data']
            predictions = lap_result['predictions']
            
            for idx, row in processed_df.iterrows():
                row_dict = row.to_dict()
                row_dict['lap_number'] = lap_num
                
                if predictions is not None and idx < len(predictions):
                    row_dict['model_prediction'] = predictions[idx]
                
                rows.append(row_dict)
        
        df_export = pd.DataFrame(rows)
        df_export.to_csv(filename, index=False)
        
        print(f"📁 Predictions exported to: {filename}")
        return filename
    
    
    def export_lap_statistics(self, filename: str = None) -> str:
        """
        Export lap statistics
        """
        if not self.performance_log:
            print("Geen data om te exporteren")
            return None
        
        if filename is None:
            filename = f"f1_lap_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        stats = []
        for lap_result in self.performance_log:
            lap_num = lap_result['lap_number']
            num_drivers = lap_result['num_drivers']
            predictions = lap_result['predictions']
            
            lap_stat = {
                'lap_number': int(lap_num),
                'drivers': int(num_drivers),
                'predictions_made': int(len(predictions)) if predictions is not None else 0,
            }
            
            if predictions is not None and len(predictions) > 0:
                lap_stat['avg_prediction'] = float(np.mean(predictions))
                lap_stat['min_prediction'] = float(np.min(predictions))
                lap_stat['max_prediction'] = float(np.max(predictions))
            
            stats.append(lap_stat)
        
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"[LAP-DATA] Statistics exported to: {filename}")
        return filename
    
    
    def get_driver_telemetry_evolution(self, driver: str) -> pd.DataFrame:
        """
        Get evolution van een driver door de race
        """
        driver_data = []
        
        for lap_result in self.performance_log:
            processed_df = lap_result['processed_data']
            
            # Filter voor deze driver
            driver_rows = processed_df[processed_df['driver'] == driver]
            
            if not driver_rows.empty:
                for _, row in driver_rows.iterrows():
                    driver_data.append(row)
        
        if not driver_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(driver_data)
        df = df.sort_values('lap_number')
        return df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_demo_pipeline():
    """
    Demo van volledige pipeline
    """
    print("\n" + "="*70)
    print("🚀 CONTINUOUS LEARNING PIPELINE DEMO")
    print("="*70)
    
    # Create pipeline
    pipeline = ContinuousLearningPipeline()
    
    # Fetch race
    print("\n1️⃣ Fetching race data from FastF1...")
    
    # Probeer 2024 Abu Dhabi
    success = pipeline.load_race(2024, 24)
    if not success:
        print("  Trying 2023 Abu Dhabi...")
        success = pipeline.load_race(2023, 22)
    
    if not success:
        print("[ERROR] Could not load race data")
        return None
    
    # Extract laps
    print("\n2️⃣ Extracting lap data...")
    laps_data = pipeline.extract_lap_data()
    
    if not laps_data:
        print("[ERROR] No lap data extracted")
        return None
    
    # Process with callback
    def lap_callback(lap_result):
        """Custom processing per lap"""
        lap_num = lap_result['lap_number']
        num_drivers = lap_result['num_drivers']
        predictions = lap_result['predictions']
        
        # Toon top 5 drivers
        df = lap_result['processed_data'].sort_values('position')
        top_5 = df[['driver', 'position', 'lap_time_seconds']].head(5)
    
    # Process en predict
    print("\n3️⃣ Processing laps and making predictions...")
    summary = pipeline.process_and_predict_laps(
        laps_data,
        callback=lap_callback,
        max_laps=10  # Beperken voor demo
    )
    
    # Export
    print("\n4️⃣ Exporting results...")
    pipeline.export_predictions()
    pipeline.export_lap_statistics()
    
    # Show driver evolution
    print("\n5️⃣ Driver evolution (sample - Max Verstappen):")
    driver_evolution = pipeline.get_driver_telemetry_evolution('VER')
    if not driver_evolution.empty:
        print(driver_evolution[['lap_number', 'position', 'lap_time_seconds']].head(10).to_string(index=False))
    
    return pipeline


if __name__ == "__main__":
    pipeline = create_demo_pipeline()
