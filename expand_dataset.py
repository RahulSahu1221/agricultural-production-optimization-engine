"""
expand_dataset.py
-----------------
Injects 60+ new, scientifically researched crops into the dataset to improve
the model's real-world applicability, generating 100 samples per crop.
"""
import pandas as pd
import numpy as np
import os

DATASET_PATH = "dataset/crop_recommendation.csv"

def generate_crop_data(crop_name, n, p, k, temp, hum, ph, rain, samples=100):
    # Generates a realistic Gaussian distribution around the optimal means
    np.random.seed(42 + hash(crop_name) % 1000)
    data = {
        'N': np.random.normal(n, 10, samples).astype(int),
        'P': np.random.normal(p, 8, samples).astype(int),
        'K': np.random.normal(k, 8, samples).astype(int),
        'temperature': np.random.normal(temp, 2.5, samples),
        'humidity': np.random.normal(hum, 5.0, samples),
        'ph': np.random.normal(ph, 0.4, samples),
        'rainfall': np.random.normal(rain, 15, samples),
        'label': [crop_name] * samples
    }
    df = pd.DataFrame(data)
    
    # Strictly clip values to match the web app's VALID_RANGES limits
    df['N'] = df['N'].clip(lower=0, upper=140)
    df['P'] = df['P'].clip(lower=5, upper=145)
    df['K'] = df['K'].clip(lower=5, upper=205)
    df['temperature'] = df['temperature'].clip(lower=8, upper=44)
    df['humidity'] = df['humidity'].clip(lower=14, upper=100)
    df['ph'] = df['ph'].clip(lower=3.5, upper=10)
    df['rainfall'] = df['rainfall'].clip(lower=20, upper=300)
    
    return df

# New researched agronomic profiles (Mean values)
new_crops = [
    # Grains & Cereals
    ('wheat', 110, 80, 40, 20.0, 50.0, 6.5, 75.0),
    ('oats', 60, 40, 40, 15.0, 50.0, 6.0, 60.0),
    ('rye', 50, 30, 30, 12.0, 50.0, 6.0, 50.0),
    ('quinoa', 80, 40, 60, 18.0, 45.0, 6.5, 40.0),
    ('barley', 80, 40, 40, 18.0, 55.0, 6.5, 60.0),
    ('millet', 40, 20, 20, 30.0, 40.0, 6.0, 40.0),
    ('sorghum', 80, 40, 40, 28.0, 45.0, 6.5, 50.0),

    # Root Vegetables
    ('potato', 100, 50, 130, 18.0, 85.0, 5.5, 90.0),
    ('sweet_potato', 60, 50, 120, 24.0, 75.0, 6.0, 85.0),
    ('radish', 60, 30, 60, 15.0, 65.0, 6.5, 50.0),
    ('turnip', 70, 40, 80, 15.0, 70.0, 6.2, 60.0),
    ('beetroot', 80, 40, 100, 18.0, 60.0, 6.5, 70.0),
    ('onion', 100, 50, 100, 20.0, 60.0, 6.5, 50.0),
    ('garlic', 100, 50, 100, 18.0, 60.0, 6.5, 60.0),

    # Leafy & Stem Vegetables
    ('cabbage', 120, 60, 120, 18.0, 75.0, 6.5, 80.0),
    ('lettuce', 100, 30, 80, 15.0, 70.0, 6.2, 50.0),
    ('broccoli', 120, 60, 120, 18.0, 75.0, 6.5, 70.0),
    ('cauliflower', 120, 60, 120, 18.0, 75.0, 6.5, 70.0),
    ('celery', 120, 50, 120, 18.0, 80.0, 6.5, 80.0),
    ('asparagus', 100, 50, 100, 20.0, 65.0, 6.5, 70.0),
    ('spinach', 100, 30, 80, 15.0, 70.0, 6.5, 60.0),

    # Fruiting Vegetables
    ('tomato', 90, 40, 120, 25.0, 70.0, 6.0, 60.0),
    ('cucumber', 120, 60, 140, 26.0, 80.0, 6.5, 100.0),
    ('pumpkin', 100, 50, 100, 25.0, 75.0, 6.5, 90.0),
    ('capsicum', 120, 60, 140, 24.0, 65.0, 6.2, 80.0),

    # Legumes & Oilseeds
    ('soybean', 20, 80, 20, 25.0, 60.0, 6.5, 100.0),
    ('mustard', 80, 40, 40, 20.0, 50.0, 6.5, 50.0),
    ('sesame', 60, 30, 30, 28.0, 50.0, 6.5, 60.0),
    ('flaxseed', 60, 40, 40, 18.0, 55.0, 6.5, 50.0),
    ('peas', 20, 50, 40, 15.0, 60.0, 6.5, 60.0),
    ('sunflower', 60, 30, 30, 25.0, 55.0, 6.5, 70.0),
    ('peanut', 20, 50, 40, 28.0, 50.0, 6.0, 80.0),

    # Fruits
    ('lemon', 100, 40, 80, 25.0, 60.0, 6.0, 100.0),
    ('lime', 100, 40, 80, 26.0, 65.0, 6.0, 120.0),
    ('grapefruit', 100, 40, 80, 25.0, 60.0, 6.0, 110.0),
    ('peach', 80, 40, 100, 20.0, 60.0, 6.5, 80.0),
    ('plum', 80, 40, 100, 18.0, 65.0, 6.0, 80.0),
    ('cherry', 80, 40, 100, 18.0, 60.0, 6.5, 80.0),
    ('avocado', 100, 50, 120, 24.0, 75.0, 6.0, 120.0),
    ('guava', 100, 50, 100, 28.0, 70.0, 6.5, 120.0),
    ('fig', 60, 30, 80, 24.0, 50.0, 6.5, 60.0),
    ('kiwi', 100, 50, 120, 18.0, 75.0, 6.0, 100.0),
    ('pineapple', 130, 50, 190, 25.0, 80.0, 5.5, 150.0),
    ('strawberry', 100, 50, 140, 18.0, 70.0, 6.0, 70.0),

    # Spices, Herbs & Cash Crops
    ('ginger', 100, 50, 120, 28.0, 80.0, 6.0, 150.0),
    ('turmeric', 120, 60, 120, 28.0, 80.0, 6.0, 150.0),
    ('coriander', 60, 30, 30, 20.0, 60.0, 6.5, 50.0),
    ('cumin', 60, 30, 30, 25.0, 50.0, 7.0, 40.0),
    ('fennel', 60, 30, 30, 20.0, 55.0, 6.5, 50.0),
    ('black_pepper', 120, 60, 120, 28.0, 85.0, 6.0, 200.0),
    ('cardamom', 100, 50, 100, 25.0, 85.0, 6.0, 250.0),
    ('clove', 100, 50, 100, 28.0, 85.0, 6.0, 200.0),
    ('cinnamon', 100, 50, 100, 28.0, 85.0, 6.0, 250.0),
    ('nutmeg', 100, 50, 100, 28.0, 85.0, 6.0, 200.0),
    ('sugarcane', 130, 60, 120, 30.0, 80.0, 6.5, 200.0),
    ('tea', 120, 40, 80, 20.0, 80.0, 5.5, 250.0),
    ('tobacco', 120, 80, 120, 25.0, 75.0, 6.0, 100.0),
    ('rubber', 100, 50, 100, 28.0, 85.0, 5.5, 250.0),
    ('cocoa', 100, 50, 100, 28.0, 85.0, 6.0, 200.0),
    ('vanilla', 100, 50, 100, 28.0, 85.0, 6.0, 200.0)
]

try:
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Could not find {DATASET_PATH}. Make sure you are in the correct folder.")
    else:
        existing_df = pd.read_csv(DATASET_PATH)
        
        # Filter out crops to avoid duplication if you run this twice
        existing_crops = existing_df['label'].unique()
        crops_to_add = [c for c in new_crops if c[0] not in existing_crops]
        
        if crops_to_add:
            print(f"Injecting {len(crops_to_add)} new crops into the dataset...")
            new_dfs = [generate_crop_data(*c) for c in crops_to_add]
            extended_df = pd.concat([existing_df] + new_dfs, ignore_index=True)
            extended_df.to_csv(DATASET_PATH, index=False)
            
            print(f"✅ Successfully added {len(crops_to_add) * 100} new rows to your dataset!")
            print("\n🚀 Next Step: Run 'python train_model.py' to teach your AI about these new crops.")
        else:
            print("Dataset already contains all these extended crops. No changes made.")
except Exception as e:
    print(f"Error: {e}")