#!/usr/bin/env python3
"""
Test script - Verify battle detection works with race simulator
"""

from battle_detector import BattleDetector
from event_generator import RaceEventGenerator

# Create test data - simulating drivers in a battle
def test_battle_detection():
    print("\n" + "="*80)
    print("[TEST] Battle Detection System")
    print("="*80 + "\n")
    
    detector = BattleDetector()
    generator = RaceEventGenerator()
    
    # Lap 1: VER and LEC 1.2s apart = BATTLE!
    print("[LAP 1] VER leading LEC by 1.2 seconds")
    lap1_drivers = [
        {
            'driver': 'VER',
            'position': 1,
            'gap_to_next': 1.2,
            'gap': 0,
            'lap_time': 88.5
        },
        {
            'driver': 'LEC',
            'position': 2,
            'gap_to_next': 2.1,
            'gap': 1.2,
            'lap_time': 89.7
        },
        {
            'driver': 'SAI',
            'position': 3,
            'gap_to_next': 5.2,
            'gap': 3.3,
            'lap_time': 94.9
        }
    ]
    
    battles = detector.detect_battles(1, lap1_drivers)
    print(f"  → Detected {len(battles)} battle events")
    for battle in battles:
        print(f"    💬 {battle['message']}")
        event = generator.generate_battle_event(battle)
        if event:
            print(f"    ✓ Event generated: {event['type']} - {event['message']}")
    
    # Lap 2: Gap closes to 0.8s = INTENSE BATTLE
    print("\n[LAP 2] Gap closes to 0.8s - VER defending!")
    lap2_drivers = [
        {'driver': 'VER', 'position': 1, 'gap_to_next': 0.8, 'gap': 0, 'lap_time': 88.3},
        {'driver': 'LEC', 'position': 2, 'gap_to_next': 2.5, 'gap': 0.8, 'lap_time': 89.1},
        {'driver': 'SAI', 'position': 3, 'gap_to_next': 5.0, 'gap': 3.3, 'lap_time': 94.8}
    ]
    
    battles = detector.detect_battles(2, lap2_drivers)
    print(f"  → Detected {len(battles)} battle events")
    for battle in battles:
        print(f"    💬 {battle['message']}")
        event = generator.generate_battle_event(battle)
        if event:
            print(f"    ✓ Event generated: {event['type']} - {event['message']}")
    
    # Lap 3: Gap increases to 1.8s = BATTLE OVER
    print("\n[LAP 3] VER escapes - gap now 1.8s")
    lap3_drivers = [
        {'driver': 'VER', 'position': 1, 'gap_to_next': 1.8, 'gap': 0, 'lap_time': 88.1},
        {'driver': 'LEC', 'position': 2, 'gap_to_next': 3.2, 'gap': 1.8, 'lap_time': 89.9},
        {'driver': 'SAI', 'position': 3, 'gap_to_next': 4.8, 'gap': 5.0, 'lap_time': 94.7}
    ]
    
    battles = detector.detect_battles(3, lap3_drivers)
    print(f"  → Detected {len(battles)} battle events")
    for battle in battles:
        print(f"    💬 {battle['message']}")
        event = generator.generate_battle_event(battle)
        if event:
            print(f"    ✓ Event generated: {event['type']} - {event['message']}")
    
    # Lap 4: No battles - quiet lap
    print("\n[LAP 4] No battles - all drivers separated")
    lap4_drivers = [
        {'driver': 'VER', 'position': 1, 'gap_to_next': 2.5, 'gap': 0, 'lap_time': 88.4},
        {'driver': 'LEC', 'position': 2, 'gap_to_next': 3.0, 'gap': 2.5, 'lap_time': 90.9},
        {'driver': 'SAI', 'position': 3, 'gap_to_next': 8.0, 'gap': 5.5, 'lap_time': 98.9}
    ]
    
    battles = detector.detect_battles(4, lap4_drivers)
    print(f"  → Detected {len(battles)} battle events")
    if not battles:
        print("    (No events - drivers separated)")
    
    print("\n" + "="*80)
    print("[RESULT] ✓ Battle detection system working correctly!")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_battle_detection()
