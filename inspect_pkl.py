import joblib
import os

# Find first pkl file
import glob
pkl_files = glob.glob(os.path.join('.cache', '**', '*.ff1pkl'), recursive=True)

if pkl_files:
    pkl_file = pkl_files[0]
    print(f"Checking: {pkl_file}\n")
    
    try:
        with open(pkl_file, 'rb') as f:
            data = joblib.load(f)
        
        print(f"Type: {type(data)}")
        print(f"Attributes/Keys: {dir(data) if not isinstance(data, dict) else list(data.keys())}")
        print(f"\nSample content (first 500 chars):")
        print(str(data)[:500])
    except Exception as e:
        print(f"Error: {e}")
else:
    print("No pkl files found")
