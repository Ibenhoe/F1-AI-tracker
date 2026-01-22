import pandas as pd
import os

# Check existing CSV
if os.path.exists('processed_f1_training_data.csv'):
    df = pd.read_csv('processed_f1_training_data.csv')
    print(f"Existing CSV - Records: {len(df)}, Years: {df['year'].min()}-{df['year'].max()}")

# Check if f1_historical_5years.csv was created
if os.path.exists('f1_historical_5years.csv'):
    df = pd.read_csv('f1_historical_5years.csv')
    print(f"New CSV - Records: {len(df)}, Years: {df['year'].min()}-{df['year'].max()}")
else:
    print("f1_historical_5years.csv NOT created")

# Check cache directory for pkl files
import glob
pkl_files = glob.glob('.cache/**/*.pkl', recursive=True)
print(f"\nCache files: {len(pkl_files)} pkl files found")
