#!/usr/bin/env python3
"""
Test the server predictions to verify the race-phase cap fix (75% for lap 2)
"""
import requests
import time
import json

BASE_URL = "http://localhost:5000"

try:
    # Health check
    resp = requests.get(f"{BASE_URL}/api/health", timeout=5)
    print(f"[OK] Server health: {resp.status_code}")
    
    # Get races
    resp = requests.get(f"{BASE_URL}/api/races", timeout=5)
    races = resp.json()
    print(f"[OK] Available races: {len(races)}")
    
    # Initialize race 5 (China)
    print(f"\n[INIT] Initializing Race 5 (China)...")
    resp = requests.post(f"{BASE_URL}/api/race/init", json={"race_number": 5}, timeout=5)
    print(f"  Status: {resp.json()['status']}")
    
    # Wait for init
    time.sleep(3)
    
    # Check init status
    resp = requests.get(f"{BASE_URL}/api/race/init-status?race=5", timeout=5)
    status_data = resp.json()
    print(f"  Init Progress: {status_data.get('progress')}% - Status: {status_data.get('status')}")
    
    # Get race state (should have predictions)
    resp = requests.get(f"{BASE_URL}/api/race/state", timeout=5)
    state = resp.json()
    
    print(f"\n[STATE] Race state:")
    print(f"  Current lap: {state.get('current_lap')}")
    print(f"  Drivers: {len(state.get('drivers', []))}")
    
    predictions = state.get('predictions', [])
    if predictions:
        print(f"\n[PREDICTIONS] Top 5 (lap {state.get('current_lap')}):")
        for pred in predictions[:5]:
            conf = pred.get('confidence', 0)
            driver = pred.get('driver_code', '?')
            print(f"    {driver}: {conf:.1f}%")
            if state.get('current_lap', 1) <= 5 and conf > 75.5:
                print(f"      ❌ Exceeds 75% cap for lap <= 5!")
            elif conf <= 75.5 or (state.get('current_lap', 1) > 5 and conf <= 82.5):
                print(f"      ✓ OK")
    else:
        print(f"\n[PREDICTIONS] None yet (race not started)")
    
    print(f"\n[TEST] Verification complete!")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
