#!/usr/bin/env python3
"""Quick full race test with new pace weighting"""
import sys
import os

# Select race 5
input_race = "5\n"

# Run race_predictor in non-interactive mode (pipe input)
os.system('echo 5 | python race_predictor.py > /dev/null 2>&1')

# Check the output file
import glob
output_files = sorted(glob.glob('outputs/race_05_*.txt'), reverse=True)

if output_files:
    latest = output_files[0]
    print(f"[LATEST OUTPUT] {latest}\n")
    print("[KEY LAPS COMPARISON]")
    print("="*70)
    
    with open(latest, 'r') as f:
        in_predictions = False
        for line in f:
            if 'PREDICTION EVOLUTION' in line:
                in_predictions = True
            elif 'FINAL CLASSIFICATION' in line:
                break
            elif in_predictions and 'LAP' in line:
                if 'LAP 4:' in line or 'LAP 10:' in line or 'LAP 21:' in line or 'LAP 40:' in line:
                    print(line.strip())
else:
    print("No output file found")
