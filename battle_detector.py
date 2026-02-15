
"""
Battle Detector - Real-time battle detection between F1 drivers
Detecteert wanneer drivers dicht bij elkaar rijden (< 1.5s gap) = bataille!
Geavanceerde features:
- Overtake kans berekening (bandenvoorraad, agressiviteit, circuit moeilijkheid)
- Overtake succesvol/mislukt detectie
- Gap trend analyse

COLOR CODES for Frontend Styling:
  - 'success' (Green): Overtake successful, battle end with clear winner
  - 'danger' (Red): Gap closing with high overtake probability, failed overtake attempt
  - 'warning' (Orange): Battle start, medium-risk situations
  - 'info' (Blue): Defensive actions, gap increasing, minor updates
"""

# Circuit overtaking difficulty (1.0 = neutral, < 1.0 = hard, > 1.0 = easy)
CIRCUIT_OVERTAKING_DIFFICULTY = {
    1: 0.6,   # Bahrain (moeilijk)
    2: 0.5,   # Saudi Arabia (erg moeilijk)
    3: 0.8,   # Australia (gemiddeld moeilijk)
    4: 0.4,   # Japan (erg moeilijk)
    5: 0.7,   # China (gemiddeld moeilijk)
    6: 0.9,   # Miami (gemiddeld)
    7: 0.7,   # Imola (moeilijk)
    8: 0.3,   # Monaco (BUITENGEWOON moeilijk)
    9: 1.0,   # Canada (neutraal/gemiddeld)
    10: 0.6,  # Spain (moeilijk)
    11: 1.1,  # Austria (relatief gemakkelijk)
    12: 0.8,  # UK (gemiddeld moeilijk)
    13: 0.9,  # Hungary (gemiddeld)
    14: 0.7,  # Belgium (moeilijk)
    15: 1.0,  # Netherlands (neutraal)
    16: 1.2,  # Italy (GEMAKKELIJK - Monza DRS-Heaven)
    17: 1.0,  # Azerbaijan (neutraal)
    18: 0.5,  # Singapore (erg moeilijk)
    19: 1.1,  # Austin (gemakkelijk)
    20: 1.0,  # Mexico (neutraal)
    21: 0.8,  # Brazil (gemiddeld moeilijk)
    22: 1.0,  # Abu Dhabi (neutraal)
}

# Driver aggression ratings (1.0 = neutral, > 1.0 = agressief, < 1.0 = defensief)
DRIVER_AGGRESSION = {
    'VER': 1.4,   # Max Verstappen - AGRESSIEF
    'LEC': 1.2,   # Charles Leclerc - Agressief
    'SAI': 1.0,   # Carlos Sainz - Neutraal
    'NOR': 1.1,   # Lando Norris - Licht agressief
    'PIA': 1.0,   # Oscar Piastri - Neutraal
    'HAM': 1.1,   # Lewis Hamilton - Licht agressief
    'RUS': 1.0,   # George Russell - Neutraal
    'ALO': 1.3,   # Fernando Alonso - Agressief (erfaring)
    'STR': 0.9,   # Lance Stroll - Licht defensief
    'GAS': 1.0,   # Pierre Gasly - Neutraal
    'OCO': 1.0,   # Esteban Ocon - Neutraal
    'MAG': 1.2,   # Kevin Magnussen - Agressief
    'HUL': 1.1,   # Nico Hulkenberg - Licht agressief
    'BOT': 1.0,   # Valtteri Bottas - Neutraal
    'ZHO': 0.9,   # Zhou Guanyu - Licht defensief
    'TSU': 1.0,   # Yuki Tsunoda - Neutraal
    'RIC': 1.1,   # Daniel Ricciardo - Licht agressief
    'ALB': 1.0,   # Alexander Albon - Neutraal
}

class BattleDetector:
    """Track battles tussen drivers per lap met geavanceerde overtake-analyse"""
    
    def __init__(self, race_number=21):
        self.previous_lap_data = {}  # {driver: {position, gap, ...}}
        self.active_battles = {}     # {(driver1, driver2): {'start_lap': X, 'gap': Y, ...}}
        self.overtake_attempts = {}  # {(driver1, driver2): {'started_lap': X, 'status': ...}}
        self.battle_threshold = 0.8  # Seconden - onder dit = gevecht!
        self.race_number = race_number
        self.circuit_difficulty = CIRCUIT_OVERTAKING_DIFFICULTY.get(race_number, 1.0)
        
        # Gap trend tracking voor betere overtake-detectie
        self.gap_history = {}  # {(driver1, driver2): [gap1, gap2, gap3, ...]}
        self.max_gap_history = 10  # Keep last 10 laps
        
        
    def detect_battles(self, lap_num, drivers_data, race_number=None):
        """
        Detecteer battles ONLY in TOP 5 POSITIONS 
        Nu met geavanceerde overtake-analyse!
        
        Args:
            lap_num: Current lap number
            drivers_data: List of {driver, position, gap_to_leader, lap_time, tire_age, tire_compound, ...}
            race_number: Optional race number to recalculate circuit difficulty
        
        Returns:
            List of battle events for this lap (inclusief overtakes en inhalingspoging analyses)
        """
        # Update circuit difficulty if race number provided
        if race_number and race_number != self.race_number:
            self.race_number = race_number
            self.circuit_difficulty = CIRCUIT_OVERTAKING_DIFFICULTY.get(race_number, 1.0)
        
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
            gap = driver1.get('gap_to_next', None)
            
            if gap is None:
                gap1 = driver1.get('gap', 0)
                gap2 = driver2.get('gap', 0)
                try:
                    gap1 = float(gap1) if gap1 else 0.0
                    gap2 = float(gap2) if gap2 else 0.0
                except (ValueError, TypeError):
                    gap1, gap2 = 0.0, 0.0
                gap = abs(gap2 - gap1)
            
            try:
                gap = float(gap)
            except (ValueError, TypeError):
                gap = 0.0
            
            # Track gap history voor trend analysis
            battle_key = (d1_code, d2_code)
            if battle_key not in self.gap_history:
                self.gap_history[battle_key] = []
            self.gap_history[battle_key].append(gap)
            if len(self.gap_history[battle_key]) > self.max_gap_history:
                self.gap_history[battle_key].pop(0)
            
            # Bepaal gap trend (stijgend, dalend, stabiel)
            gap_trend = 'stable'
            gap_change = 0.0
            old_gap = gap
            
            if len(self.gap_history[battle_key]) >= 2:
                old_gap = self.gap_history[battle_key][-2]
                gap_change = gap - old_gap
                if gap_change > 0.1:
                    gap_trend = 'increasing'
                elif gap_change < -0.1:
                    gap_trend = 'decreasing'
            
            # DEBUG output
            if gap < 2.0 and lap_num > 0 and lap_num % 3 == 0:
                status = "BATTLE" if gap < self.battle_threshold else "Close"
                trend_icon = "↑" if gap_trend == 'increasing' else "↓" if gap_trend == 'decreasing' else "="
                print(f"[GAP-DEBUG-TOP5] Lap {lap_num} P{i+1}-P{i+2}: {d1_code} vs {d2_code} = {gap:.2f}s {trend_icon} {status}")
            
            # ===== OVERTAKE PROBABILITY CALCULATION =====
            if gap < 1.5 and gap > 0:  # Only for close battles
                tire_info_d1 = {
                    'tire_age': driver1.get('tire_age', 10),
                    'tire_compound': driver1.get('tire_compound', 'MEDIUM')
                }
                tire_info_d2 = {
                    'tire_age': driver2.get('tire_age', 10),
                    'tire_compound': driver2.get('tire_compound', 'MEDIUM')
                }
                
                # d1 is probeert d2 in te halen
                overtake_prob = self.calculate_overtake_probability(
                    d1_code, d2_code, gap, tire_info_d1, tire_info_d2
                )
            else:
                overtake_prob = 0.0
            
            # ===== OVERTAKE SUCCESS DETECTION =====
            d1_current_pos = driver1.get('position', i+1)
            d2_current_pos = driver2.get('position', i+2)
            d1_prev_pos = self.previous_lap_data.get(d1_code, {}).get('position', d1_current_pos)
            d2_prev_pos = self.previous_lap_data.get(d2_code, {}).get('position', d2_current_pos)
            
            overtake_events = self.detect_overtake_success(
                lap_num, d1_current_pos, d2_current_pos, d1_prev_pos, d2_prev_pos, d1_code, d2_code
            )
            events.extend(overtake_events)
            
            # ===== BATTLE DETECTION =====
            if gap < self.battle_threshold and gap > 0:
                battle_key = (d1_code, d2_code)
                
                if battle_key not in self.active_battles:
                    # NEW BATTLE!
                    self.active_battles[battle_key] = {
                        'start_lap': lap_num,
                        'positions': [d1_current_pos, d2_current_pos],
                        'gap': gap,
                        'overtake_prob': overtake_prob,
                        'tire_diff': tire_info_d1.get('tire_age', 0) - tire_info_d2.get('tire_age', 0),
                        'type': 'battle_start'
                    }
                    
                    prob_text = f" ({overtake_prob:.0f}% kans)" if overtake_prob > 30 else ""
                    events.append({
                        'type': 'battle',
                        'subtype': 'battle_start',
                        'drivers': [d1_code, d2_code],
                        'positions': [d1_current_pos, d2_current_pos],
                        'lap': lap_num,
                        'gap': round(gap, 2),
                        'overtake_probability': round(overtake_prob, 1),
                        'message': f'BATTLE BEGINS: {d1_code} vs {d2_code} - {gap:.2f}s gap{prob_text}',
                        'color_code': 'warning',
                        'severity': 'high'
                    })
                else:
                    # ONGOING BATTLE - check voor veranderingen
                    old_gap = self.active_battles[battle_key]['gap']
                    
                    try:
                        old_gap = float(old_gap)
                        gap = float(gap)
                    except (ValueError, TypeError):
                        continue
                    
                    gap_change = abs(old_gap - gap)
                    
                    if gap_change > 0.5:  # Gap changed by >0.5s
                        self.active_battles[battle_key]['gap'] = gap
                        self.active_battles[battle_key]['overtake_prob'] = overtake_prob
                        
                        if gap < old_gap:
                            # Gap closing - driver1 is gaining
                            if overtake_prob > 50:
                                risk_msg = "GROTE KANS!"
                            elif overtake_prob > 30:
                                risk_msg = f"risico {overtake_prob:.0f}%"
                            else:
                                risk_msg = f"lastig {overtake_prob:.0f}%"
                            
                            events.append({
                                'type': 'battle',
                                'subtype': 'gap_closing',
                                'drivers': [d1_code, d2_code],
                                'positions': [d1_current_pos, d2_current_pos],
                                'lap': lap_num,
                                'gap': round(gap, 2),
                                'overtake_probability': round(overtake_prob, 1),
                                'message': f'Gap closing! {d1_code} attacks {d2_code} - nu {gap:.2f}s - {risk_msg}',
                                'color_code': 'danger' if overtake_prob > 50 else 'warning',
                                'severity': 'high' if overtake_prob > 50 else 'medium'
                            })
                        else:
                            # Gap increasing - check for failed overtake
                            failed_events = self.detect_failed_overtake_attempt(
                                lap_num, d1_code, d2_code, gap_change, old_gap, gap, gap_trend
                            )
                            events.extend(failed_events)
                            
                            if not failed_events:  # Als niet gedetecteerd als failure, gewoon gap update
                                events.append({
                                    'type': 'battle',
                                    'subtype': 'gap_increasing',
                                    'drivers': [d1_code, d2_code],
                                    'positions': [d1_current_pos, d2_current_pos],
                                    'lap': lap_num,
                                    'gap': round(gap, 2),
                                    'message': f'{d2_code} defends: gap nu {gap:.2f}s (trend: {gap_trend})',
                                    'color_code': 'info',
                                    'severity': 'low'
                                })
            else:
                # Not battling - check if battle ended
                battle_key = (d1_code, d2_code)
                if battle_key in self.active_battles:
                    # BATTLE ENDED
                    old_battle = self.active_battles.pop(battle_key)
                    gap_over_threshold = gap > self.battle_threshold
                    
                    if gap_over_threshold and gap > 0:  # Legitimate end (not DNF)
                        winner = d1_code if d1_current_pos < d2_current_pos else d2_code
                        duration = lap_num - old_battle['start_lap']
                        
                        events.append({
                            'type': 'battle',
                            'subtype': 'battle_end',
                            'drivers': [d1_code, d2_code],
                            'winner': winner,
                            'lap': lap_num,
                            'duration_laps': duration,
                            'message': f'Battle voorbij: {winner} wint! ({duration} laps intens gevecht)',
                            'color_code': 'success',
                            'severity': 'medium'
                        })
        
        # Werk previous lap data bij
        for driver in sorted_drivers:
            driver_code = driver.get('driver', 'UNK')
            self.previous_lap_data[driver_code] = {
                'position': driver.get('position', 999),
                'gap': driver.get('gap', 0),
                'tire_age': driver.get('tire_age', 0)
            }
        
        return events
    
    
    def calculate_overtake_probability(self, attacker, defender, gap, tire_info_attacker, tire_info_defender):
        """
        Bereken de kans dat attacker defender zal inhalen
        
        Factors:
        - Gap (hoe kleiner hoe beter)
        - Tire condition (verse banden = beter)
        - Driver aggression (agressieve drivers hebben betere kans)
        - Circuit difficulty (op Monaco moeilijker dan op Monza)
        
        Returns: float (0.0 - 100.0) representing overtake probability
        """
        overtake_prob = 50.0  # Base probability
        
        # 1. GAP FACTOR (zeer belangrijk)
        # Hoe kleiner de gap, hoe meer kans op inhalen
        if gap <= 0.3:
            overtake_prob += 30.0  # Zeer dicht - hoge kans
        elif gap <= 0.5:
            overtake_prob += 20.0  # Dicht - goede kans
        elif gap <= 0.8:
            overtake_prob += 10.0  # Gemiddeld - redelijke kans
        elif gap <= 1.2:
            overtake_prob += 5.0   # Ver uit elkaar - lage kans
        else:
            overtake_prob -= 15.0  # Te ver uit elkaar
        
        # 2. TIRE CONDITION FACTOR (ZEER BELANGRIJK!)
        # Verse banden = betere traction en performance
        attacker_tire_age = tire_info_attacker.get('tire_age', 10) if tire_info_attacker else 10
        defender_tire_age = tire_info_defender.get('tire_age', 10) if tire_info_defender else 10
        
        # Tire advantage: als attacker fresher banden heeft, grotere kans
        tire_age_diff = defender_tire_age - attacker_tire_age
        
        if tire_age_diff > 5:
            overtake_prob += 25.0  # Attacker veel fresher
        elif tire_age_diff > 2:
            overtake_prob += 15.0  # Attacker fresher
        elif tire_age_diff > 0:
            overtake_prob += 8.0   # Attacker iets fresher
        elif tire_age_diff < -5:
            overtake_prob -= 20.0  # Defender veel fresher
        elif tire_age_diff < -2:
            overtake_prob -= 12.0  # Defender fresher
        elif tire_age_diff < 0:
            overtake_prob -= 5.0   # Defender iets fresher
        
        # Tire wear: zeer versleten banden zijn slecht voor verdediger
        if defender_tire_age > 25:
            overtake_prob += 20.0  # Defender banden erg versleten
        elif defender_tire_age > 15:
            overtake_prob += 10.0  # Defender banden versleten
        
        # 3. DRIVER AGGRESSION FACTOR
        attacker_agg = DRIVER_AGGRESSION.get(attacker, 1.0)
        defender_agg = DRIVER_AGGRESSION.get(defender, 1.0)
        
        # Agressieve attackers hebben betere kans
        overtake_prob += (attacker_agg - 1.0) * 15.0  # Max +21% bonus voor zeer agressief
        
        # Agressieve defenders zijn moeilijker in te halen
        overtake_prob -= (defender_agg - 1.0) * 10.0  # Max -14% penalty
        
        # 4. CIRCUIT DIFFICULTY FACTOR
        # Op circuits waar inhalen gemakkelijk is, hogere kans
        circuit_factor = self.circuit_difficulty
        overtake_prob = overtake_prob * (0.7 + 0.3 * circuit_factor)
        
        # CLAMP: nooit hoger dan 95%, nooit lager dan 5%
        overtake_prob = max(5.0, min(95.0, overtake_prob))
        
        return overtake_prob
    
    def detect_overtake_success(self, lap_num, driver1_current_pos, driver2_current_pos, driver1_prev_pos, driver2_prev_pos, driver1, driver2):
        """
        Detecteer wanneer een overtake geslaagd is of mislukt
        Overtake sukces: driver inhaaltz andere (position veranderd)
        Overtake mislukking: gap wordt groter en geen position change
        """
        events = []
        
        # Positie change?
        driver1_gained = driver1_prev_pos > driver1_current_pos  # Lager getal = beter
        driver2_lost_pos = driver2_prev_pos < driver2_current_pos
        
        # Check voor succesvol overtake
        if driver1_gained and driver2_lost_pos:
            # driver1 is driver2 voorbij gerijden!
            events.append({
                'type': 'overtake',
                'subtype': 'overtake_success',
                'drivers': [driver1, driver2],
                'lap': lap_num,
                'new_position': driver1_current_pos,
                'message': f'OVERTAKE! {driver1} haalt {driver2} in op lap {lap_num}!',
                'color_code': 'success',
                'severity': 'high'
            })
        
        return events
    
    def detect_failed_overtake_attempt(self, lap_num, driver1, driver2, gap_change, old_gap, new_gap, gap_trend):
        """
        Detecteer wanneer een overtake-poging mislukt
        (gap wordt groter, driver valt terug)
        """
        events = []
        
        # Gap groeit (meer dan 0.3s stijging) = mislukking
        if gap_change > 0.3 and gap_trend == 'increasing':
            events.append({
                'type': 'overtake',
                'subtype': 'overtake_failed',
                'drivers': [driver1, driver2],
                'lap': lap_num,
                'gap_before': round(old_gap, 2),
                'gap_after': round(new_gap, 2),
                'message': f'{driver1} faalt in inhalingspoging! {driver2} breekt weg (+{gap_change:.2f}s)',
                'color_code': 'danger',
                'severity': 'medium'
            })
        
        return events
    
    def reset(self, race_number=None):
        """Reset battle detector for new race"""
        self.active_battles = {}
        self.previous_lap_data = {}
        self.overtake_attempts = {}
        self.gap_history = {}
        
        # Update race-specific parameters
        if race_number:
            self.race_number = race_number
            self.circuit_difficulty = CIRCUIT_OVERTAKING_DIFFICULTY.get(race_number, 1.0)
    
    def get_battle_summary(self):
        """
        Retourneer samenvatting van actieve battles met statistieken
        Handig voor dashboard/UI weergave
        """
        summary = []
        
        for (driver1, driver2), battle_data in self.active_battles.items():
            gap = battle_data.get('gap', 0)
            overtake_prob = battle_data.get('overtake_prob', 0)
            duration = max(1, getattr(self, 'current_lap', 0) - battle_data.get('start_lap', getattr(self, 'current_lap', 0)))
            
            # Tire advantage analysis
            tire_diff = battle_data.get('tire_diff', 0)
            tire_advantage = ""
            if tire_diff > 0:
                tire_advantage = f"🔴 {driver1} heeft frissere banden (+{tire_diff}L)"
            elif tire_diff < 0:
                tire_advantage = f"🔴 {driver2} heeft frissere banden (+{abs(tire_diff)}L)"
            else:
                tire_advantage = "⚪ Beide hebben gelijke bandenleeftijd"
            
            summary.append({
                'driver1': driver1,
                'driver2': driver2,
                'gap': round(gap, 2),
                'overtake_probability': round(overtake_prob, 1),
                'duration_laps': duration,
                'tire_advantage': tire_advantage,
                'circuit_difficulty': self.circuit_difficulty,
                'intensity': 'CRITICAL' if overtake_prob > 70 else 'HIGH' if overtake_prob > 50 else 'MEDIUM' if overtake_prob > 30 else 'LOW'
            })
        
        return summary
    
    def get_circuit_overtaking_difficulty(self):
        """Return readable circuit overtaking difficulty"""
        diff = self.circuit_difficulty
        if diff < 0.5:
            return "BUITENGEWOON MOEILIJK (Monaco-level)"
        elif diff < 0.8:
            return "Moeilijk"
        elif diff < 1.0:
            return "Gemiddeld moeilijk"
        elif diff == 1.0:
            return "Neutraal"
        elif diff < 1.2:
            return "Gemiddeld gemakkelijk"
        else:
            return "Gemakkelijk (Monza-style)"