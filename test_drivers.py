import fastf1
import warnings
warnings.filterwarnings('ignore')

session = fastf1.get_session(2024, 21, 'R')
session.load()

# Check available attributes
print('Session attributes:', [attr for attr in dir(session) if not attr.startswith('_')][:20])

# Get drivers
drivers = session.drivers
print('\nAll drivers:', drivers if isinstance(drivers, list) else drivers.tolist())
print('Total drivers:', len(drivers))

# Get laps
laps = session.laps
print('\nLaps shape:', laps.shape)
print('Laps columns:', laps.columns.tolist()[:10])
print('First few laps:\n', laps[['Driver', 'DriverNumber', 'LapNumber']].head())
