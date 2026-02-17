#!/usr/bin/env python3
import fastf1
import pandas as pd
import os
import tempfile

# Enable cache
cache_dir = os.path.join(tempfile.gettempdir(), 'fastf1_cache')
fastf1.Cache.enable_cache(cache_dir)

try:
    print("[TEST] Loading Brazil 2024 race telemetry...")
    session = fastf1.get_session(2024, 21, 'R')
    session.load(telemetry=True, weather=False)  # Enable telemetry!
    
    print(f"[OK] Session loaded")
    print(f"[TEST] Session has telemetry: {hasattr(session, 'telemetry')}")
    print(f"[TEST] Session has car_data: {hasattr(session, 'car_data')}")
    print(f"[TEST] Session has laps: {hasattr(session, 'laps')}")
    
    # Check what data we have
    if hasattr(session, 'telemetry'):
        tel = session.telemetry
        print(f"[TEST] Telemetry type: {type(tel)}")
        print(f"[TEST] Telemetry shape: {tel.shape if hasattr(tel, 'shape') else 'N/A'}")
        print(f"[TEST] Telemetry columns: {list(tel.columns) if hasattr(tel, 'columns') else 'N/A'}")
        print(f"[TEST] Telemetry head:\n{tel.head()}")
    
    if hasattr(session, 'car_data'):
        car_data = session.car_data
        print(f"\n[TEST] Car data type: {type(car_data)}")
        print(f"[TEST] Car data shape: {car_data.shape if hasattr(car_data, 'shape') else 'N/A'}")
        print(f"[TEST] Car data columns: {list(car_data.columns) if hasattr(car_data, 'columns') else 'N/A'}")
    
    # Check laps telemetry
    if hasattr(session, 'laps'):
        laps = session.laps
        print(f"\n[TEST] Laps shape: {laps.shape}")
        # Get first lap and its telemetry
        lap1 = laps.iloc[0]
        print(f"[TEST] First lap has telemetry: {hasattr(lap1, 'telemetry')}")
        if hasattr(lap1, 'telemetry') and lap1.telemetry is not None:
            print(f"[TEST] First lap telemetry shape: {lap1.telemetry.shape}")
            print(f"[TEST] First lap telemetry columns: {list(lap1.telemetry.columns)}")
            print(f"[TEST] First lap telemetry head:\n{lap1.telemetry.head()}")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
