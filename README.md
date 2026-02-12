# 🏆 F1 Race Predictor

Een interactief F1 race prediction systeem dat per lap voorspellingen doet over de top 5 finishers.

## 🚀 Quick Start

```bash
python race_predictor.py
```

Volg vervolgens de interactieve menu's:
1. Kies een race uit de 21 races van 2024
2. Het systeem trainent het model per lap
3. Na lap 5 krijg je live predictions te zien
4. Output wordt opgeslagen in `outputs/` folder

## 📊 Features

✅ **Interactief Race Menu** - Kies uit 21 races van het 2024 F1 seizoen  
✅ **Per-Lap Predictions** - Ziet realtime voorspellingen na elke lap (starten vanaf lap 5)  
✅ **Top 5 Finishers** - Toont accuracy scores en visuele voortgangsbalk  
✅ **Output Opslag** - Alle predictions worden opgeslagen in `outputs/` folder  
✅ **Live Progress** - Voortgangsprocent en lap counter  
✅ **Final Classification** - Toont uiteindelijke race resultaten  

## 🗂️ Project Structure

```
F1-AI-tracker/
├── race_predictor.py              # Main entry point
├── continuous_model_learner.py    # ML model & training
├── continuous_learning_pipeline.py # Data pipeline
├── fastf1_data_fetcher.py         # F1 data from API
├── outputs/                       # Saved predictions
└── requirements.txt               # Dependencies
```

## 📈 How It Works

1. **Data Loading**: FastF1 API haalt real race data op
2. **Per-Lap Training**: Model wordt na elke lap bijgewerkt
3. **Predictions**: Top 5 finishers voorspeld met accuracy scores
4. **Output Saving**: Alle resultaten opgeslagen als `.txt` bestand

## 📝 Output Files

Voorspellingen worden opgeslagen als:
```
outputs/race_XX_YYYYMMDD_HHMMSS.txt
```

Bevat:
- Prediction evolution per lap
- Accuracy scores per driver
- Final classification
- Timestamp en race info

## 🔧 Requirements

```
xgboost
pandas
numpy
fastf1
```

Installeer:
```bash
pip install -r requirements.txt
```

## 🏎️ Available Races (2024)

**Eerste Helft:**
1. Bahrain, 2. Saudi Arabia, 3. Australia, 4. Japan, 5. China  
6. Miami, 7. Monaco, 8. Canada, 9. Spain, 10. Austria

**Tweede Helft:**
11. United Kingdom, 12. Hungary, 13. Belgium, 14. Netherlands, 15. Italy  
16. Azerbaijan, 17. Singapore, 18. Austin, 19. Mexico, 20. Brazil

**Final:**
21. Abu Dhabi

---

**Status**: ✅ Production Ready  
**Project**: Final Work 2026 - Erasmus Hogeschool Brussel
