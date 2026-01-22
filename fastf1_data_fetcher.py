"""
FastF1 Data Fetcher Module
Haalt F1 race data op en verwerkt het per lap voor continuous learning
"""

import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Tuple
import warnings
import os
import tempfile
warnings.filterwarnings("ignore")

# Cache FastF1 data locally to speed up development
# Use a cross-platform temp directory
cache_dir = os.path.join(tempfile.gettempdir(), 'fastf1_cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)


class FastF1DataFetcher:
    """
    Haalt race data van FastF1 op en verwerkt het lap-voor-lap
    """
    
    def __init__(self):
        self.session = None
        self.race_year = None
        self.race_round = None
        self.laps_data = []
        
    def fetch_race(self, year: int, round_number: int) -> bool:
        """
        Haalt een specifieke race op van FastF1
        
        Args:
            year: Seizoensjaar (bijv 2024)
            round_number: Race nummer (bijv 1 = Bahrain)
            
        Returns:
            bool: True als succesvol, False als fout
        """
        try:
            print(f"[LOAD] Bezig met laden van {year} Race {round_number}...")
            
            self.session = fastf1.get_session(year, round_number, 'R')
            self.session.load()
            
            self.race_year = year
            self.race_round = round_number
            
            print(f"[OK] Race geladen! Locatie: {self.session.event['Location']}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Fout bij laden race: {e}")
            return False
    
    
    def get_drivers_in_race(self) -> List[str]:
        """Geeft lijst van coureurs in de race"""
        if self.session is None:
            return []
        drivers = self.session.drivers
        # Handle both numpy arrays and lists
        if hasattr(drivers, 'tolist'):
            return drivers.tolist()
        return list(drivers)
    
    
    def extract_lap_features(self, lap_data: pd.Series, driver: str) -> Dict:
        """
        Extraheert features uit 1 lap data
        Dit zijn de features die iedere lap beschikbaar zijn
        """
        try:
            # Get lap number safely
            lap_number = lap_data.get('LapNumber', np.nan)
            if pd.isna(lap_number):
                return None
            
            # Get position safely
            position = lap_data.get('Position', np.nan)
            if pd.isna(position):
                return None
            
            # Convert to int safely
            try:
                lap_number = int(lap_number)
                position = int(position)
            except (ValueError, TypeError):
                return None
            
            # Get lap time safely - use pd.NaT instead of np.nan!
            lap_time = lap_data.get('LapTime', pd.NaT)
            if isinstance(lap_time, pd.Timedelta):
                lap_time_seconds = lap_time.total_seconds()
            else:
                try:
                    lap_time_seconds = float(lap_time) if not pd.isna(lap_time) else np.nan
                except:
                    lap_time_seconds = np.nan
            
            # Get tire life safely
            tire_life = lap_data.get('TyreLife', np.nan)
            try:
                tire_age = float(tire_life) if not pd.isna(tire_life) else np.nan
            except:
                tire_age = np.nan
            
            # Get DRS safely
            drs = lap_data.get('DRS', 0)
            try:
                drs_available = int(drs) if not pd.isna(drs) else 0
            except:
                drs_available = 0
            
            features = {
                'driver': str(driver),
                'lap_number': lap_number,
                'lap_time': lap_time_seconds,
                'position': position,
                'is_pit_lap': bool(lap_data.get('PitInTime', pd.NaT) is not pd.NaT) or bool(lap_data.get('PitOutTime', pd.NaT) is not pd.NaT),
                'tire_compound': str(lap_data.get('Compound', 'UNKNOWN')),
                'tire_age': tire_age,
                'track_status': str(lap_data.get('TrackStatus', 'UNKNOWN')),
                'drs_available': drs_available,
                'fresh_tires': int(bool(lap_data.get('FreshTyre', False))),
            }
            
            return features
        
        except Exception as e:
            return None
    
    
    def process_race_laps_streaming(self, show_progress: bool = True) -> List[Dict]:
        """
        Verwerkt alle laps van de race en retourneert ze als stream
        Dit simuleert real-time lap data
        
        Returns:
            List van dictionaries met per-lap features
        """
        if self.session is None:
            print("[ERROR] Geen race geladen! Roep eerst fetch_race() aan")
            return []
        
        all_laps_data = []
        
        try:
            # Get all laps at once
            all_laps = self.session.laps
            
            if all_laps is None or len(all_laps) == 0:
                print("[ERROR] No laps in session!")
                return []
            
            print(f"  [LAP-DATA] Total laps in session: {len(all_laps)}")
            
            # DEBUG: Print first lap
            if len(all_laps) > 0:
                first_lap = all_laps.iloc[0]
                print(f"\n  DEBUG - First lap sample:")
                print(f"    LapNumber: {first_lap.get('LapNumber')} (type: {type(first_lap.get('LapNumber'))})")
                print(f"    Position: {first_lap.get('Position')} (type: {type(first_lap.get('Position'))})")
                print(f"    Driver: {first_lap.get('Driver')} / DriverNumber: {first_lap.get('DriverNumber')}")
                print(f"    LapTime: {first_lap.get('LapTime')} (type: {type(first_lap.get('LapTime'))})")
                print()
            
            # Group by driver
            drivers = self.get_drivers_in_race()
            print(f"  [DRIVERS] Drivers: {drivers}")
            
            for driver in drivers:
                try:
                    # Pick laps for this driver - try both string and int
                    laps = None
                    try:
                        laps = all_laps.pick_driver(driver)
                    except:
                        try:
                            laps = all_laps.pick_driver(int(driver))
                        except:
                            laps = all_laps[all_laps['DriverNumber'] == int(driver)] if driver.isdigit() else None
                    
                    if laps is None or len(laps) == 0:
                        continue
                    
                    print(f"  [DRIVER] Driver {driver}: {len(laps)} laps")
                    
                    # Extract features
                    lap_count = 0
                    for idx, (_, lap) in enumerate(laps.iterrows()):
                        lap_features = self.extract_lap_features(lap, driver)
                        if lap_features is not None:
                            all_laps_data.append(lap_features)
                            lap_count += 1
                    
                    if lap_count == 0:
                        print(f"      [WARN] No valid laps extracted for driver {driver}")
                    
                except Exception as e:
                    print(f"  [WARN] Error with driver {driver}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Sort chronologically
            all_laps_data.sort(key=lambda x: (x['lap_number'], x['driver']))
            
            print(f"\n[OK] {len(all_laps_data)} laps processed!")
            
            return all_laps_data
            
        except Exception as e:
            print(f"[ERROR] Error processing laps: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    
    def group_laps_by_lap_number(self, laps_data: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Groepeer lap data per lap nummer (bijv lap 1 van alle drivers)
        Dit is beter voor het model om per lap voorbij te gaan
        
        Returns:
            Dict met lap_number als key en list van drivers als value
        """
        grouped = {}
        
        for lap_data in laps_data:
            lap_num = int(lap_data['lap_number'])
            
            if lap_num not in grouped:
                grouped[lap_num] = []
            
            grouped[lap_num].append(lap_data)
        
        return grouped
    
    
    def export_to_csv(self, laps_data: List[Dict], filename: str = None):
        """
        Exporteert lap data naar CSV voor analyse
        """
        if not laps_data:
            print("Geen lap data om te exporteren")
            return
        
        if filename is None:
            filename = f"f1_race_{self.race_year}_r{self.race_round}_laps.csv"
        
        df = pd.DataFrame(laps_data)
        df.to_csv(filename, index=False)
        print(f"📁 Data geëxporteerd naar: {filename}")
        
        return df
    
    
    def get_race_summary(self) -> Dict:
        """
        Geeft samenvatting van de race
        """
        if self.session is None:
            return {}
        
        return {
            'year': self.race_year,
            'round': self.race_round,
            'event': self.session.event['EventName'],
            'location': self.session.event['Location'],
            'date': str(self.session.event['EventDate']),
            'total_drivers': len(self.get_drivers_in_race()),
        }


class LapStreamSimulator:
    """
    Simuleert real-time lap data streaming
    Handig voor testing van het model zonder live data
    """
    
    def __init__(self, laps_data: List[Dict]):
        self.laps_data = laps_data
        self.lap_groups = self._group_by_lap()
        self.current_lap = 1
        
    def _group_by_lap(self) -> Dict[int, List[Dict]]:
        """Groepeer per lap nummer"""
        grouped = {}
        for lap in self.laps_data:
            lap_num = int(lap['lap_number'])
            if lap_num not in grouped:
                grouped[lap_num] = []
            grouped[lap_num].append(lap)
        return grouped
    
    def stream_lap_by_lap(self):
        """
        Generator die lap-voor-lap data uitgeeft
        Handig voor model training per lap
        """
        for lap_num in sorted(self.lap_groups.keys()):
            lap_data = self.lap_groups[lap_num]
            yield {
                'lap_number': lap_num,
                'drivers_data': lap_data,
                'timestamp': datetime.now().isoformat()
            }
    
    def stream_driver_data(self, driver: str):
        """
        Generator voor 1 specifieke coureur
        """
        driver_laps = [l for l in self.laps_data if l['driver'] == driver]
        driver_laps.sort(key=lambda x: x['lap_number'])
        
        for lap_data in driver_laps:
            yield lap_data


# ============================================================================
# DEMONSTRATIE / TEST FUNCTIE
# ============================================================================

def demo_fetch_and_process():
    """
    Demonstreert hoe FastF1 data op te halen en te verwerken
    """
    print("=" * 60)
    print("FastF1 Data Fetcher - DEMO")
    print("=" * 60)
    
    # Initialize fetcher
    fetcher = FastF1DataFetcher()
    
    # Fetch vorige race (2024 Abu Dhabi / Vorige jaar seizoen einde)
    success = fetcher.fetch_race(2024, 24)  # Abu Dhabi
    
    if not success:
        print("Probeer met 2023...")
        success = fetcher.fetch_race(2023, 22)  # Abu Dhabi 2023
    
    if success:
        # Get race info
        summary = fetcher.get_race_summary()
        print(f"\n[LAP-DATA] Race Gegevens:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Process alle laps
        laps_data = fetcher.process_race_laps_streaming()
        
        # Group per lap nummer
        laps_by_number = fetcher.group_laps_by_lap_number(laps_data)
        
        print(f"\n[FINISH] Lap Summary:")
        print(f"  Totale laps: {len(laps_by_number)}")
        print(f"  Totale lap records: {len(laps_data)}")
        
        # Toon eerste paar laps
        print(f"\n📍 Eerste 3 laps voorbeeld:")
        for lap_num in sorted(list(laps_by_number.keys())[:3]):
            print(f"\n  Lap {lap_num}:")
            for driver_data in laps_by_number[lap_num][:3]:  # Eerste 3 drivers
                print(f"    {driver_data['driver']}: " +
                      f"Pos {driver_data['position']}, " +
                      f"Time {driver_data['lap_time']:.2f}s, " +
                      f"Tire: {driver_data['tire_compound']}")
        
        # Export to CSV
        df = fetcher.export_to_csv(laps_data)
        
        # Toon statistieken
        print(f"\n📈 Data Statistieken:")
        print(df.describe())
        
        return laps_data
    
    return None


if __name__ == "__main__":
    demo_laps = demo_fetch_and_process()
