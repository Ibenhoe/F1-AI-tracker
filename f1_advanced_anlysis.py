# %% [markdown]
# # F1 Advanced Feature Engineering & Analysis
# Based on the Titanic Data Science workflow.
# This script performs advanced cleaning, feature creation (binning), and preparation for AI models.

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# %%
# 1. LOAD DATA
# We use the unprocessed data because we want to engineer features from scratch
print("Loading data...")
df = pd.read_csv('unprocessed_f1_training_data.csv')

# Ensure 'country' column exists (in case data_script.py wasn't re-run)
if 'country' not in df.columns:
    print("'country' column missing. Merging from circuits.csv...")
    circuits = pd.read_csv('../F1_data_mangement/circuits.csv')
    df = pd.merge(df, circuits[['circuitId', 'country']], on='circuitId', how='left')

# Ensure 'driverRef' column exists (for readability in analysis)
if 'driverRef' not in df.columns:
    print("'driverRef' column missing. Merging from drivers.csv...")
    drivers = pd.read_csv('../F1_data_mangement/drivers.csv')
    df = pd.merge(df, drivers[['driverId', 'driverRef']], on='driverId', how='left')

# Sort by date to ensure 'Experience' calculations are correct later
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# Split into Train (Historic) and Test (Recent/Future)
# In F1, we don't split randomly; we split by time. Let's say 2023+ is our "Test" set.
train_df = df[df['year'] < 2023].copy()
test_df = df[df['year'] >= 2023].copy()
combined_df = pd.concat([train_df, test_df], axis=0)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(combined_df.columns.values)

# %%
# 2. EXPLORATORY DATA ANALYSIS (EDA)
#
# Check correlations with the target (positionOrder)
# We want to see what correlates with a LOWER position (1 is better than 20)
numeric_df = train_df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
print("\n--- Correlation with Finishing Position ---")
print(corr["positionOrder"].sort_values(ascending=True).head(10))

# Visualization: Grid Position vs Survival (Finishing in Top 10)
# In Titanic this was 'Survived'. In F1, let's call 'Points Finish' (Top 10) our 'Survival'.
train_df['is_points_finish'] = train_df['positionOrder'].apply(lambda x: 1 if x <= 10 else 0)

print("\n--- Grid Position vs Points Finish Chance ---")
print(train_df[['grid', 'is_points_finish']].groupby(['grid'], as_index=False).mean().head(10))

g = sns.FacetGrid(train_df, col='is_points_finish')
g.map(plt.hist, 'grid', bins=20)
plt.show()

# %%
# 3. FEATURE ENGINEERING

# A. DRIVER EXPERIENCE (Similar to 'Age' or 'Family Size')
# Calculate how many races the driver had entered BEFORE this one.
print("\nGenerating 'Driver Experience' feature...")
combined_df['driver_experience'] = combined_df.groupby('driverId').cumcount()

# Update Train/Test
train_df = combined_df[combined_df['year'] < 2023]
test_df = combined_df[combined_df['year'] >= 2023]

# Visualize Experience vs Success
sns.lineplot(data=train_df, x='driver_experience', y='positionOrder')
plt.title("Driver Experience vs Finishing Position (Lower is Better)")
plt.show()

# B. HOME RACE (Similar to 'Title' extraction or 'Is_Alone')
# Does the driver's nationality match the circuit country?
print("Generating 'Home Race' feature...")

# Simple mapping for common F1 nations (expand this list for better accuracy)
nationality_map = {
    'British': 'UK', 'German': 'Germany', 'Spanish': 'Spain', 'French': 'France',
    'Italian': 'Italy', 'Dutch': 'Netherlands', 'Australian': 'Australia', 
    'Monegasque': 'Monaco', 'American': 'USA', 'Japanese': 'Japan', 'Canadian': 'Canada'
}

combined_df['mapped_nationality'] = combined_df['nationality'].map(nationality_map).fillna(combined_df['nationality'])
combined_df['is_home_race'] = np.where(combined_df['mapped_nationality'] == combined_df['country'], 1, 0)

print(combined_df[['driverRef', 'country', 'is_home_race']].tail(5))

# C. BINNING (Similar to AgeBin and FareBin)

# 1. Grid Bins
# Grid position is crucial. Let's group them: Pole, Front Row, Top 10, Midfield, Backmarker.
combined_df['grid_bin'] = pd.cut(combined_df['grid'], 
                                 bins=[-1, 1, 3, 10, 15, 25], 
                                 labels=['Pole', 'Top3', 'Points', 'Midfield', 'Back'])

# 2. Age Bins
combined_df['age_bin'] = pd.cut(combined_df['driver_age'], 
                                bins=[17, 24, 30, 36, 60], 
                                labels=['Rookie', 'Prime', 'Experienced', 'Veteran'])

# Check the impact of Bins
print("\n--- Age Bin vs Average Finish Position ---")
print(combined_df[['age_bin', 'positionOrder']].groupby(['age_bin'], as_index=False).mean().sort_values(by='positionOrder'))

# %%
# 4. ENCODING CATEGORICAL FEATURES
# Machine Learning models need numbers, not words like 'Prime' or 'McLaren'.

print("\nEncoding features...")
label = LabelEncoder()

combined_df['grid_bin_code'] = label.fit_transform(combined_df['grid_bin'])
combined_df['age_bin_code'] = label.fit_transform(combined_df['age_bin'])
combined_df['constructor_code'] = label.fit_transform(combined_df['constructorId'])
combined_df['circuit_code'] = label.fit_transform(combined_df['circuitId'])

# %%
# 5. PREPARING FOR MODEL (SCALING)
# Similar to the Titanic 'StandardScaler' step.

# Define the features we want to use for training
features = [
    'grid', 
    'driver_experience', 
    'is_home_race', 
    'grid_bin_code', 
    'age_bin_code', 
    'constructor_code', 
    'circuit_code',
    'driver_age'
]

target = 'positionOrder'

# Separate Train and Test again
train_df = combined_df[combined_df['year'] < 2023].copy()
test_df = combined_df[combined_df['year'] >= 2023].copy()

# Drop rows where target or features might be NaN
train_df = train_df.dropna(subset=features + [target])
test_df = test_df.dropna(subset=features + [target])

X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]

# Scale the data
# We fit the scaler ONLY on the training data to avoid data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for nice viewing
X_train_final = pd.DataFrame(X_train_scaled, columns=features)
X_test_final = pd.DataFrame(X_test_scaled, columns=features)

print("\n--- Final Training Data (First 5 rows) ---")
print(X_train_final.head())

# %%
# 6. QUICK MODEL TEST (Logistic Regression - Classification)
# Just like the Titanic example used LogisticRegression, let's try to predict if a driver finishes in Top 10.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create a binary target: 1 if Top 10, 0 if not
y_train_class = (y_train <= 10).astype(int)
y_test_class = (test_df[target] <= 10).astype(int)

logreg = LogisticRegression()
logreg.fit(X_train_final, y_train_class)
y_pred = logreg.predict(X_test_final)

acc = accuracy_score(y_test_class, y_pred)
print(f"\nLogistic Regression Accuracy (Predicting Top 10 Finish): {acc * 100:.2f}%")

# Feature Importance (Coefficient)
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

print("\n--- Feature Coefficients (Positive = More likely to be Top 10) ---")
print(pd.DataFrame({'Feature': features, 'Coefficient': logreg.coef_[0]}).sort_values(by='Coefficient', ascending=False))
# %%