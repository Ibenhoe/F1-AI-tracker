import { useState } from 'react'
import './RaceSelector.css'

const RACES = {
  1: 'Bahrain',
  2: 'Saudi Arabia',
  3: 'Australia',
  4: 'Japan',
  5: 'China',
  6: 'Miami',
  7: 'Monaco',
  8: 'Canada',
  9: 'Spain',
  10: 'Austria',
  11: 'United Kingdom',
  12: 'Hungary',
  13: 'Belgium',
  14: 'Netherlands',
  15: 'Italy',
  16: 'Azerbaijan',
  17: 'Singapore',
  18: 'Austin',
  19: 'Mexico',
  20: 'Brazil',
  21: 'Abu Dhabi'
}

export default function RaceSelector({ selectedRace, onRaceChange, onSelectRace, disabled }) {
  // Support both onRaceChange (original) and onSelectRace (new PreRaceAnalysis)
  const handleChange = (e) => {
    const value = parseInt(e.target.value)
    if (onSelectRace) {
      onSelectRace(value)
    } else if (onRaceChange) {
      onRaceChange(value)
    }
  }

  return (
    <div className="race-selector">
      <label>Select Race:</label>
      <select 
        value={selectedRace || 21} 
        onChange={handleChange}
        disabled={disabled}
        className="race-select"
      >
        {Object.entries(RACES).map(([num, name]) => (
          <option key={num} value={num}>
            {num}. {name}
          </option>
        ))}
      </select>
    </div>
  )
}
