# Quick Start: ML Model Display

## What You Get
The frontend now displays the AI model's **continuous learning** in real-time. Every lap, the model trains itself more and the frontend shows:
- ✅ Model update count
- 📊 Model maturity progress (0-150 updates)
- 🧬 Which ML models are active
- 📈 Learning performance (MAE trend)

## Start the System

```bash
# Terminal 1: Backend
python app.py
# Runs on http://localhost:5000

# Terminal 2: Frontend  
cd frontend && npm run dev
# Runs on http://localhost:5173
```

## What to Look For

1. **Model Metrics Panel** appears between Race Info and Driver List
2. **Purple/violet design** with real-time updates
3. **Status badge**: Shows "Training..." or "Optimized"
4. **Component grid**: SGDRegressor ✅, MLPRegressor ⏳, RandomForest ✅
5. **Maturity bar**: Fills up from 0→100% (threshold at 150 updates)
6. **Model updates counter**: Increments every lap

## How It Works

### Backend
```python
# In race_simulator.py
lap_state['model_metrics'] = self.model.get_model_metrics_for_frontend(lap_number)

# In continuous_model_learner.py  
def get_model_metrics_for_frontend(self, lap_number=0) -> Dict:
    return {
        'total_updates': self.updates_count,
        'model_maturity_percentage': min(100, (self.updates_count / 150) * 100),
        'sgd_model_ready': self.sgd_model is not None,
        'mlp_model_ready': self.mlp_model is not None,
        'rf_classifier_ready': self.rf_classifier is not None,
        # ... more metrics
    }
```

### Frontend
```jsx
// In Dashboard.jsx
const [modelMetrics, setModelMetrics] = useState(null)

apiClient.on('lap/update', (data) => {
  if (data.model_metrics) {
    setModelMetrics(data.model_metrics)  // ← Updates here!
  }
})

// Render it
<ModelMetricsPanel 
  modelMetrics={modelMetrics}
  currentLap={raceData?.currentLap}
  totalLaps={raceData?.totalLaps}
/>
```

## Component Breakdown

### ModelMetricsPanel.jsx
- 💜 Purple gradient UI
- 📊 Grid layout for different sections
- 🔄 Smooth animations on updates
- 📱 Responsive for mobile

### What It Shows

| Section | Shows |
|---------|-------|
| **Status Badge** | Training / Optimized / Initializing |
| **Model Overview** | Type, Confidence cap (85%), Pre-trained |
| **Updates Section** | Total updates, Samples processed |
| **Maturity Bar** | 0-100% progress (0-150 updates) |
| **Components** | 3 models with status (✅/⏳) |
| **Features** | All features used in training |
| **Performance** | Recent MAE, Trend (improving/stable) |

## Example Output During Race

```
LAP 25/57 - 44% Complete

🧠 AI Model - Real-time Learning    ✅ Training

Model Type: Incremental SGD + MLP Ensemble
Confidence Cap: 85%
Pre-trained: ✅ Yes

MODEL UPDATES
┌─────────────────────────┐
│ Total Updates:    42    │
│ Samples Proc:    210    │
└─────────────────────────┘

Model Maturity [████████░░░░░░░░░] 28%

MODEL COMPONENTS
├─ 📊 SGDRegressor   ✅
├─ 🧬 MLPRegressor   ✅  
└─ 🌲 RandomForest   ✅

FEATURES: grid, driver_age, points_constructor, circuitId...

RECENT PERFORMANCE
Average MAE: 3.87        Trend: 📈 Improving
```

## Key Metrics Explained

### Total Updates
- Increments each lap the model trains
- Target: 150+ for "Optimized" status
- Shows model is actively learning

### Samples Processed
- Each lap processes all 20 drivers = 20 samples
- Total = lap_number × 20 (approximately)
- More samples = better learning

### Model Maturity %
- Formula: `(total_updates / 150) * 100`
- 0-50%: Early learning phase (red)
- 50-100%: Training phase (orange/yellow)
- 100%+: Optimized phase (green)

### MAE (Mean Absolute Error)
- Measures prediction accuracy
- Lower is better (closer to actual positions)
- Trend shows if model is improving lap-to-lap

### Component Status
- SGDRegressor: Online learning (always ✅ if training)
- MLPRegressor: Neural network backup (✅ when ready)
- RandomForest: Top-5 probability classifier (✅ when ready)

## Files Changed

```
✨ NEW FILES:
frontend/src/components/ModelMetricsPanel.jsx
frontend/src/components/ModelMetricsPanel.css
ML_MODEL_DISPLAY_GUIDE.md (this file)

📝 MODIFIED FILES:
continuous_model_learner.py        → Added get_model_metrics_for_frontend()
race_simulator.py                  → Added model_metrics to lap_state
frontend/src/pages/Dashboard.jsx   → Integrated ModelMetricsPanel
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Panel shows "Loading..." | Wait for race to start and lap 1 to complete |
| No metrics updating | Check browser console for errors (F12) |
| Models stuck at ⏳ | Model may not have enough samples yet (lap 5+) |
| Confidence shows 100% | Is not possible - should be max 85% (check display) |

## Next Steps

The model metrics display is now integrated! You can:
1. ✅ See real-time model training
2. ✅ Watch maturity progress increase
3. ✅ Monitor component activation
4. ✅ Track learning performance

Future enhancements could include:
- [ ] MAE trend graph (line chart)
- [ ] Feature importance heatmap
- [ ] Prediction accuracy tracker
- [ ] Historical comparison
- [ ] Model export button

---

**Status**: ✅ Production Ready  
**Last Updated**: January 2026  
**Tested On**: Windows PowerShell, Node 20+, Python 3.10+
