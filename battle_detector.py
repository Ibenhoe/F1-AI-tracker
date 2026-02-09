#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Battle Detector - Real-time battle detection between F1 drivers
Detecteert wanneer drivers dicht bij elkaar rijden (< 1.5s gap) = bataille!
"""

class BattleDetector:
    """Track battles tussen drivers per lap"""
    
    def __init__(self):
        self.previous_lap_data = {}  # {driver: {position, gap, ...}}
        self.active_battles = {}     # {(driver1, driver2): {'start_lap': X, 'gap': Y, ...}}
        self.battle_threshold = 0.8  # Seconden - onder dit = gevecht! (lowered from 1.5 for realistic F1 battles)
        
    def detect_battles(self, lap_num, drivers_data):
        """
        Detecteer battles ONLY in TOP 5 POSITIONS 
        
        Args:
            lap_num: Current lap number
            drivers_data: List of {driver, position, gap_to_leader, lap_time, ...}
        
        Returns:
            List of battle events for this lap (only top 5 battles)
        """
        events = []
        
        if not drivers_data or len(drivers_data) < 2:
            return events
        
        # Ensure all positions are floats (fix for string positions)
        for driver in drivers_data:
            if 'position' in driver:
                try:
                    driver['position'] = float(driver['position'])
                except (ValueError, TypeError):
                    driver['position'] = 999.0
        
        # Sort by position
        sorted_drivers = sorted(drivers_data, key=lambda x: float(x.get('position', 999)))
        
        # ONLY CONSIDER TOP 5 DRIVERS for battle detection to reduce notification spam
        sorted_drivers = sorted_drivers[:5]
        
        # Check each consecutive pair for battles
        for i in range(len(sorted_drivers) - 1):
            driver1 = sorted_drivers[i]
            driver2 = sorted_drivers[i + 1]
            
            d1_code = driver1.get('driver', 'UNK')
            d2_code = driver2.get('driver', 'UNK')
            
            # Extract gap (distance to next driver)
            # If no explicit gap, estimate from position difference
            gap = driver1.get('gap_to_next', None)
            
            if gap is None:
                # Fallback: use gap_to_leader difference
                gap1 = driver1.get('gap', 0)
                gap2 = driver2.get('gap', 0)
                
                # Ensure gap values are floats before subtraction
                try:
                    gap1 = float(gap1) if gap1 else 0.0
                    gap2 = float(gap2) if gap2 else 0.0
                except (ValueError, TypeError):
                    gap1, gap2 = 0.0, 0.0
                
                gap = abs(gap2 - gap1)
            
            # Convert to float and ensure valid
            try:
                gap = float(gap)
            except (ValueError, TypeError):
                gap = 0.0
            
            # DEBUG: Show gaps in TOP 5 only
            if gap < 2.0 and lap_num > 0 and lap_num % 3 == 0:
                status = "⚔️ BATTLE" if gap < self.battle_threshold else "📊 Close"
                print(f"[GAP-DEBUG-TOP5] Lap {lap_num} P{i+1}-P{i+2}: {d1_code} vs {d2_code} = {gap:.2f}s {status}")
            
            # Check if drivers are battling (gap < threshold)
            if gap < self.battle_threshold and gap > 0:
                battle_key = (d1_code, d2_code)
                
                # Check if this is a new battle or ongoing
                if battle_key not in self.active_battles:
                    # NEW BATTLE!
                    self.active_battles[battle_key] = {
                        'start_lap': lap_num,
                        'positions': [driver1.get('position', i+1), driver2.get('position', i+2)],
                        'gap': gap,
                        'type': 'battle_start'
                    }
                    
                    events.append({
                        'type': 'battle',
                        'subtype': 'battle_start',
                        'drivers': [d1_code, d2_code],
                        'positions': [driver1.get('position', i+1), driver2.get('position', i+2)],
                        'lap': lap_num,
                        'gap': round(gap, 2),
                        'message': f'⚔️ BATTLE BEGINS: {d1_code} vs {d2_code} - {gap:.2f}s gap at P{driver1.get("position", i+1)}'
                    })
                else:
                    # ONGOING BATTLE - only emit if gap changed significantly
                    old_gap = self.active_battles[battle_key]['gap']
                    
                    # Ensure both are floats before subtraction
                    try:
                        old_gap = float(old_gap)
                        gap = float(gap)
                    except (ValueError, TypeError):
                        continue
                    
                    gap_change = abs(old_gap - gap)
                    
                    if gap_change > 0.5:  # Gap changed by >0.5s
                        self.active_battles[battle_key]['gap'] = gap
                        
                        if gap < old_gap:
                            # Gap closing - driver1 is gaining
                            events.append({
                                'type': 'battle',
                                'subtype': 'gap_closing',
                                'drivers': [d1_code, d2_code],
                                'positions': [driver1.get('position', i+1), driver2.get('position', i+2)],
                                'lap': lap_num,
                                'gap': round(gap, 2),
                                'message': f'🔥 Gap closing! {d1_code} attacks {d2_code} - now {gap:.2f}s'
                            })
                        else:
                            # Gap increasing
                            events.append({
                                'type': 'battle',
                                'subtype': 'gap_increasing',
                                'drivers': [d1_code, d2_code],
                                'positions': [driver1.get('position', i+1), driver2.get('position', i+2)],
                                'lap': lap_num,
                                'gap': round(gap, 2),
                                'message': f'📊 {d2_code} defends: gap now {gap:.2f}s'
                            })
            else:
                # Not battling - check if battle ended
                battle_key = (d1_code, d2_code)
                if battle_key in self.active_battles:
                    # BATTLE ENDED
                    old_battle = self.active_battles.pop(battle_key)
                    gap_over_threshold = gap > self.battle_threshold
                    
                    if gap_over_threshold and gap > 0:  # Legitimate end (not DNF)
                        winner = d1_code if driver1.get('position', i+1) < driver2.get('position', i+2) else d2_code
                        events.append({
                            'type': 'battle',
                            'subtype': 'battle_end',
                            'drivers': [d1_code, d2_code],
                            'winner': winner,
                            'lap': lap_num,
                            'duration_laps': lap_num - old_battle['start_lap'],
                            'message': f'✓ Battle over: {winner} wins! ({lap_num - old_battle["start_lap"]} laps of battle)'
                        })
        
        return events
    
    def reset(self):
        """Reset battle detector for new race"""
        self.active_battles = {}
        self.previous_lap_data = {}
