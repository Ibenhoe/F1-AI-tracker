#!/usr/bin/env python3
"""
Test the actual API function to see why it's not working
"""

import fastf1
import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def fetch_qualifying_grid_debug(race_num):
    """Exact replica of the backend function for testing"""
    try:
        print(f"  [GRID] Attempting to fetch FastF1 qualifying data for race {race_num}...")
        
        with time_limit(10):
            print(f"  [GRID] Creating session object...")
            qual_session = fastf1.get_session(2024, race_num, 'Q')
            
            print(f"  [GRID] Loading session (disabling telemetry/weather for speed)...")
            qual_session.load(telemetry=False, weather=False)
            
            print(f"  [GRID] Processing results...")
            grid = []
            if qual_session.results is not None and len(qual_session.results) > 0:
                for grid_idx, (_, row) in enumerate(qual_session.results.iterrows()):
                    driver_code = str(row.get('Abbreviation', ''))
                    if driver_code and driver_code != 'nan':
                        grid_pos = grid_idx + 1
                        grid.append({
                            'driver': driver_code,
                            'number': int(row.get('DriverNumber', 0)),
                            'team': str(row.get('TeamName', 'Unknown')),
                            'grid_pos': grid_pos
                        })
                
                print(f"  [GRID] ✓ Successfully loaded {len(grid)} drivers from FastF1 qualifying")
                print(f"  [GRID] Top 5 drivers:")
                for g in grid[:5]:
                    print(f"         P{g['grid_pos']:2d}: {g['driver']} ({g['team']})")
                return grid
            else:
                print(f"  [GRID] No qualifying results found in FastF1 for race {race_num}")
                return None
            
    except TimeoutException:
        print(f"  [GRID] TIMEOUT: FastF1 API took too long for race {race_num}")
        return None
    except Exception as e:
        print(f"  [GRID] ERROR fetching FastF1 data: {type(e).__name__}: {str(e)[:100]}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("\n" + "="*80)
    print("[TEST] Testing _fetch_qualifying_grid() function for race 5 (China)")
    print("="*80 + "\n")
    
    result = fetch_qualifying_grid_debug(5)
    
    print("\n" + "="*80)
    if result:
        print("[RESULT] ✓ Function returned data successfully")
    else:
        print("[RESULT] ✗ Function returned None")
    print("="*80 + "\n")
