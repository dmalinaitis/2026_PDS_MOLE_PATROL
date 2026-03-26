import os
import cv2
import pandas as pd
from feature_A import get_asymmetry
from utils import MASK_DIR, OUTPUT_CSV_PATH

def extract_all_features():
    results = []
    
    # Iterate through all masks in the directory
    for filename in os.listdir(MASK_DIR):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_id = filename.replace("_mask", "").split(".")[0]
            
            # Read the mask in grayscale
            filepath = os.path.join(MASK_DIR, filename)
            mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            # 1. Extract Asymmetry
            asym_score = get_asymmetry(mask)
            
            # 2. Append to our row dictionary
            results.append({
                "img_id": img_id,
                "asymmetry_score": asym_score
                # You will add 'border_score', etc. here later
            })
            
    # Convert list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(results)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Features successfully extracted and saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    extract_all_features()