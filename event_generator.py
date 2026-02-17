#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Race Event Generator - Genereer race events voor notificaties
Centralized event generation - minimale coupling met race_simulator
"""

import time
from datetime import datetime

class RaceEventGenerator:
    """Generate events voor notificatie system"""
    
    def __init__(self):
        self.event_queue = []
        self.last_event_time = {}  # {event_type: timestamp} - prevent spam
        self.event_throttle = 2.0   # Minimum seconds between same event types
        
    def generate_battle_event(self, battle_data):
        """Generate battle notification event"""
        event = {
            'id': int(time.time() * 1000),
            'timestamp': datetime.now().isoformat(),
            'type': battle_data.get('type', 'battle'),
            'subtype': battle_data.get('subtype', 'battle_start'),
            'drivers': battle_data.get('drivers', []),
            'lap': battle_data.get('lap', 0),
            'gap': battle_data.get('gap', 0),
            'message': battle_data.get('message', 'Battle event'),
            'color_code': battle_data.get('color_code', 'info'),
            'severity': battle_data.get('severity', 'normal'),
        }
        
        # Throttle to prevent spam
        event_key = f"battle_{battle_data.get('subtype', 'start')}"
        if self._should_throttle(event_key):
            return None
        
        self.last_event_time[event_key] = time.time()
        return event
    
    def generate_pit_event(self, pit_data):
        """Generate pit stop notification event"""
        event = {
            'id': int(time.time() * 1000),
            'timestamp': datetime.now().isoformat(),
            'type': 'pit_stop',
            'driver': pit_data.get('driver', 'UNK'),
            'lap': pit_data.get('lap', 0),
            'tire_old': pit_data.get('tire_old', 'UNKNOWN'),
            'tire_new': pit_data.get('tire_new', 'UNKNOWN'),
            'stops_total': pit_data.get('stops_total', 1),
            'message': pit_data.get('message', 'Pit stop'),
        }
        
        event_key = f"pit_{pit_data.get('driver', 'UNK')}"
        if self._should_throttle(event_key):
            return None
        
        self.last_event_time[event_key] = time.time()
        return event
    
    def generate_top5_update(self, top5_data):
        """Generate top 5 update event (less frequent)"""
        event = {
            'id': int(time.time() * 1000),
            'timestamp': datetime.now().isoformat(),
            'type': 'top5_update',
            'lap': top5_data.get('lap', 0),
            'top5': top5_data.get('top5', []),
        }
        
        # Top5 updates only every 5 laps to prevent spam
        if not self._should_throttle('top5_update', min_interval=5.0):
            return None
        
        self.last_event_time['top5_update'] = time.time()
        return event
    
    def generate_overtake_event(self, overtake_data):
        """Generate overtake event"""
        event = {
            'id': int(time.time() * 1000),
            'timestamp': datetime.now().isoformat(),
            'type': 'overtake',
            'driver': overtake_data.get('driver', 'UNK'),
            'position_new': overtake_data.get('position_new', 0),
            'position_old': overtake_data.get('position_old', 0),
            'lap': overtake_data.get('lap', 0),
            'on_driver': overtake_data.get('on_driver', 'UNK'),
            'message': overtake_data.get('message', 'Overtake'),
        }
        
        event_key = f"overtake_{overtake_data.get('driver', 'UNK')}"
        if self._should_throttle(event_key):
            return None
        
        self.last_event_time[event_key] = time.time()
        return event
    
    def generate_dnf_event(self, dnf_data):
        """Generate DNF/incident event"""
        event = {
            'id': int(time.time() * 1000),
            'timestamp': datetime.now().isoformat(),
            'type': 'incident',
            'driver': dnf_data.get('driver', 'UNK'),
            'lap': dnf_data.get('lap', 0),
            'reason': dnf_data.get('reason', 'DNF'),
            'message': dnf_data.get('message', 'Driver DNF'),
        }
        
        event_key = f"dnf_{dnf_data.get('driver', 'UNK')}"
        if self._should_throttle(event_key):
            return None
        
        self.last_event_time[event_key] = time.time()
        return event
    
    def _should_throttle(self, event_key, min_interval=None):
        """Check if event should be throttled to prevent spam"""
        if min_interval is None:
            min_interval = self.event_throttle
        
        if event_key not in self.last_event_time:
            return False
        
        time_since_last = time.time() - self.last_event_time[event_key]
        return time_since_last < min_interval
    
    def add_event(self, event):
        """Add event to queue"""
        if event:
            self.event_queue.append(event)
            return True
        return False
    
    def get_events(self):
        """Get all queued events (and clear queue)"""
        events = self.event_queue.copy()
        self.event_queue = []
        return events
    
    def reset(self):
        """Reset for new race"""
        self.event_queue = []
        self.last_event_time = {}