#!/usr/bin/env python3
"""Direct test of telemetry extraction"""
import sys
import json
from fastf1_data_fetcher import FastF1DataFetcher

try:
    print("\n[1/3] Initializing fetcher...")
    fetcher = FastF1DataFetcher()
    
    print("[2/3] Loading race 21 (Brazil 2024)...")
    success = fetcher.fetch_race(year=2024, round_number=21)
    if not success:
        print("[ERROR] Failed to load race!")
        sys.exit(1)
    
    print("[3/3] Extracting lap data with telemetry...")
    laps = fetcher.process_race_laps_streaming(show_progress=False)
    
    # Analyze results
    print(f"\n[OK] Total laps extracted: {len(laps)}")
    
    # Count telemetry success
    tel_laps = [l for l in laps if l.get('x') is not None]
    tel_count = len(tel_laps)
    rate = 100 * tel_count // len(laps) if laps else 0
    
    print(f"[OK] Telemetry extracted: {tel_count}/{len(laps)} ({rate}%)")
    
    # Show samples
    if tel_laps:
        print("\n[SAMPLES] Laps with telemetry:")
        for lap in tel_laps[:3]:
            print(f"  {lap['driver']} L{int(lap['lap_number'])}: "
                  f"x={lap.get('x', 0):.1f}, y={lap.get('y', 0):.1f}, "
                  f"speed={lap.get('speed')}, gear={lap.get('gear')}")
        print("\n[SUCCESS] TELEMETRY EXTRACTION WORKING!")
    else:
        print("\n[ERROR] ERROR: No telemetry extracted at all!")
        print(f"\nFirst 3 laps (should have None x,y): ")
        for lap in laps[:3]:
            print(f"  {lap['driver']} L{int(lap['lap_number'])}: x={lap.get('x')}, y={lap.get('y')}")
            
except Exception as e:
    print(f"\n[ERROR] EXCEPTION: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
