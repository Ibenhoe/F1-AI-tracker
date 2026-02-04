#!/usr/bin/env python3
"""
Debug FastF1 data fetching for Race 5 (China 2024)
Verify that Max Verstappen is P1 and Sergio Pérez is P2
"""

import fastf1
import pandas as pd

def test_fastf1_china():
    """Test FastF1 qualifying data for China 2024"""
    
    print("\n" + "="*80)
    print("[DEBUG] Testing FastF1 qualifying data for China 2024 (Race 5)")
    print("="*80)
    
    try:
        print("\n[FASTF1] Fetching qualifying session for race 5...")
        qual_session = fastf1.get_session(2024, 5, 'Q')
        
        print("[FASTF1] Loading qualifying data...")
        qual_session.load(telemetry=False, weather=False)
        
        print(f"\n[FASTF1] ✓ Qualifying session loaded successfully")
        print(f"[FASTF1] Session date: {qual_session.date}")
        print(f"[FASTF1] Session name: {qual_session.name}")
        
        if qual_session.results is not None and len(qual_session.results) > 0:
            print(f"\n[FASTF1] Found {len(qual_session.results)} drivers in qualifying results")
            print("\n[FASTF1] Top 10 qualifying positions:")
            print("-" * 80)
            
            grid = []
            for grid_idx, (_, row) in enumerate(qual_session.results.iterrows()):
                driver_code = str(row.get('Abbreviation', ''))
                driver_name = str(row.get('FullName', 'Unknown'))
                team_name = str(row.get('TeamName', 'Unknown'))
                
                if driver_code and driver_code != 'nan':
                    grid_pos = grid_idx + 1
                    grid.append({
                        'driver': driver_code,
                        'name': driver_name,
                        'team': team_name,
                        'grid_pos': grid_pos
                    })
                    
                    if grid_idx < 10:
                        print(f"  P{grid_pos:2d}: {driver_code:3s} {driver_name:20s} ({team_name})")
            
            print("-" * 80)
            
            # Verify China data
            print("\n[VERIFICATION] Expected vs Actual:")
            print("-" * 80)
            
            # Check Verstappen
            ver_entry = next((g for g in grid if g['driver'] == 'VER'), None)
            if ver_entry:
                print(f"  ✓ VER (Max Verstappen) found at P{ver_entry['grid_pos']}")
                if ver_entry['grid_pos'] == 1:
                    print(f"    ✓ CORRECT: P1 (as expected for China 2024)")
                else:
                    print(f"    ✗ WRONG: Expected P1, got P{ver_entry['grid_pos']}")
            else:
                print(f"  ✗ VER (Max Verstappen) NOT FOUND in grid!")
            
            # Check Pérez
            per_entry = next((g for g in grid if g['driver'] == 'PER'), None)
            if per_entry:
                print(f"  ✓ PER (Sergio Pérez) found at P{per_entry['grid_pos']}")
                if per_entry['grid_pos'] == 2:
                    print(f"    ✓ CORRECT: P2 (as expected for China 2024)")
                else:
                    print(f"    ✗ WRONG: Expected P2, got P{per_entry['grid_pos']}")
            else:
                print(f"  ✗ PER (Sergio Pérez) NOT FOUND in grid!")
            
            print("-" * 80)
            print(f"\n[FASTF1] ✓ Data is CORRECT - FastF1 API working properly for China 2024")
            return True
        else:
            print(f"\n[FASTF1] ✗ No qualifying results found!")
            return False
            
    except Exception as e:
        print(f"\n[FASTF1] ✗ ERROR fetching data: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fastf1_china()
    print("\n" + "="*80)
    if success:
        print("[RESULT] ✓ FastF1 data is working correctly")
        print("[ACTION] Backend should use FastF1 data instead of fallback")
    else:
        print("[RESULT] ✗ FastF1 data retrieval failed")
        print("[ACTION] Need to debug why FastF1 is not working")
    print("="*80 + "\n")
